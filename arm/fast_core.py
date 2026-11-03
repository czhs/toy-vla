"""Shared core of the fast SO-101 SmolVLA pipeline, used by build_feature_cache.py and train_policy.py.

Everything here computes the same functions as lerobot 0.6.1 SmolVLA (expert-only fine-tune, frozen SigLIP /
connector / VLM text layers, cross_attn with self-attn every 2nd layer); differences are bf16 reduction order only.

Training step (split_forward):
  image features  = connector(SigLIP(image))    read from a per-dataset cache (build_feature_cache.py)
  prefix K/V      = VLM layers over [image, lang] no grad; language pad positions dropped
  grad pass       = state token through the VLM layers + 50 action tokens through the expert
Inference (FastChunker): prefix incl. state once, expert K/V of the prefix hoisted out of the 10 Euler steps,
optionally the 10 steps captured as one CUDA graph (torch.compile mode reduce-overhead).
Eval (run_condition_batched): a pool of envs steps in lockstep; every env that needs a new chunk joins ONE
batched policy call; the overhead camera renders only on the control step the policy sees; the unused wrist
camera is not rendered. Per-env flow noise comes from a per-env CUDA generator seeded like the stock wrapper
(bit-identical to eval_closed_loop.SmolVLAWrapper at pool size 1)."""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lerobot.policies.smolvla import modeling_smolvla as MS
from lerobot.policies.smolvla.smolvlm_with_expert import apply_rope

import paths  # noqa: E402
FEAT_ROOT = paths.FEATURE_CACHE
FEAT_TOKENS, FEAT_DIM = 64, 960


# =============================================================================================== config

def make_base_config(out_dir: Path, cameras: dict) -> Path:
    """Local copy of lerobot/smolvla_base declaring observation.state + the given cameras {key: [3, H, W]} (in that
    order = the prefix order) and a 6-dim action; weights symlinked. For cameras={overhead: [3,512,512]} this is"""
    import os
    from huggingface_hub import snapshot_download
    src = Path(snapshot_download("lerobot/smolvla_base", revision=paths.SMOLVLA_BASE_REVISION))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    feats = {"observation.state": {"type": "STATE", "shape": [6]}}
    for k, shp in cameras.items():
        feats[k] = {"type": "VISUAL", "shape": list(shp)}
    cfg["input_features"] = feats
    cfg["output_features"] = {"action": {"type": "ACTION", "shape": [6]}}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    for f in src.iterdir():
        if f.name.endswith(".safetensors") and not (out_dir / f.name).exists():
            os.symlink(f.resolve(), out_dir / f.name)
    return out_dir


def dataset_cameras(root: Path, keys) -> dict:
    """{key: [3, H, W]} from the dataset's info.json, in the given order."""
    info = json.loads((Path(root).expanduser() / "meta/info.json").read_text())
    out = {}
    for k in keys:
        h, w, c = info["features"][k]["shape"]
        out[k] = [c, h, w]
    return out


# =============================================================================================== feature cache

def dataset_fingerprint(root: Path) -> dict:
    """Cheap identity of a LeRobot dataset: frame / episode counts and the size + mtime of every video file."""
    root = Path(root).expanduser()
    info = json.loads((root / "meta/info.json").read_text())
    vids = sorted((root / "videos").rglob("*.mp4"))
    h = hashlib.sha1()
    for v in vids:
        st = v.stat()
        h.update(f"{v.relative_to(root)}:{st.st_size}:{int(st.st_mtime)}".encode())
    return dict(total_frames=info["total_frames"], total_episodes=info["total_episodes"], videos_sha1=h.hexdigest())


def cache_dir_for(dataset: str | Path, image_key="observation.images.overhead") -> Path:
    root = Path(dataset).expanduser().resolve()
    return FEAT_ROOT / f"{root.name}__{image_key.split('.')[-1]}"


class FeatureCache:
    """[N, 64, 960] bf16 SigLIP+connector features (stored as uint16) of one dataset's image key."""

    def __init__(self, path: Path, mode="r"):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        n = self.meta["n_frames"]
        self.arr = np.memmap(self.path / "features.u16", dtype=np.uint16, mode=mode, shape=(n, FEAT_TOKENS, FEAT_DIM))
        self.ram = None

    def check(self, dataset):
        fp = dataset_fingerprint(dataset)
        assert fp == self.meta["dataset_fingerprint"], f"feature cache {self.path} is for another dataset version: {fp} vs {self.meta['dataset_fingerprint']}"
        done = np.load(self.path / "done.npy")
        assert done.all(), f"feature cache incomplete: {int(done.sum())}/{done.size} blocks; run build_feature_cache.py"

    def load_to_ram(self):
        self.ram = torch.from_numpy(np.ascontiguousarray(self.arr)).view(torch.bfloat16)
        return self

    def gather(self, idx: np.ndarray) -> torch.Tensor:
        if self.ram is not None:
            return self.ram[torch.as_tensor(idx)]
        order = np.argsort(idx)                      # sequential-ish reads, then restore order
        out = np.empty((len(idx), FEAT_TOKENS, FEAT_DIM), np.uint16)
        out[order] = self.arr[idx[order]]
        return torch.from_numpy(out).view(torch.bfloat16)


# =============================================================================================== data (no video)

class TableBatcher:
    """State / 50-action chunk / action_is_pad for dataset indices, from the parquet, already normalised by the
    policy preprocessor. Equal to LeRobotDataset.__getitem__ + preprocessor."""

    def __init__(self, ds, pre, chunk=50, device="cuda"):
        import pandas as pd
        root = Path(ds.root)
        df = pd.concat([pd.read_parquet(p, columns=["index", "episode_index", "observation.state", "action"])
                        for p in sorted((root / "data").rglob("*.parquet"))]).sort_values("index")
        assert np.array_equal(df["index"].to_numpy(), np.arange(len(df)))
        st = torch.tensor(np.stack(df["observation.state"].to_numpy()), dtype=torch.float32)
        ac = torch.tensor(np.stack(df["action"].to_numpy()), dtype=torch.float32)
        ep = torch.as_tensor(df["episode_index"].to_numpy())
        ep_to = torch.as_tensor(np.asarray(ds.meta.episodes["dataset_to_index"]))
        self.hi = (ep_to[ep] - 1).to(device)
        # normalise with the SAME preprocessor (elementwise, so tables == per-batch results bit for bit)
        from scene import TASK
        out = pre({"observation.state": st, "action": ac[:, None, :], "task": [TASK] * 1})
        self.st = out["observation.state"].to(device).float().reshape(len(df), -1)
        self.ac = out["action"].to(device).float().reshape(len(df), -1)
        tok = pre({"observation.state": st[:1], "action": ac[:1, None, :], "task": [TASK]})
        self.tokens = tok["observation.language.tokens"].to(device)[:1]
        self.tmask = tok["observation.language.attention_mask"].to(device)[:1]
        self.k = torch.arange(chunk, device=device)
        self.device = device

    def batch(self, idx) -> dict:
        i = torch.as_tensor(idx, device=self.device)
        j = i[:, None] + self.k[None]
        hi = self.hi[i][:, None]
        B = len(i)
        return {"observation.state": self.st[i], "action": self.ac[torch.minimum(j, hi)], "action_is_pad": j > hi,
                "observation.language.tokens": self.tokens.expand(B, -1),
                "observation.language.attention_mask": self.tmask.expand(B, -1)}


# =============================================================================================== model pieces

def image_features(policy, batch):
    """[connector(SigLIP(img)) per camera, in config.image_features order], exactly as embed_prefix computes them
    (before the sqrt(d) scale). All cameras must be present (no empty-camera padding)."""
    images, masks = policy.prepare_images(batch)
    assert len(images) == len(policy.config.image_features), "every configured camera must be in the batch"
    for mk in masks:
        assert bool(mk.all()), "image padding masks are not supported by the fast path"
    return [policy.model.vlm_with_expert.embed_image(im) for im in images]


def prefix_kv_layers(vwe, embs, att2d, pos):
    """Prefix prefill (as SmolVLMWithExpertModel.forward_attn_layer, prefix only); returns [(K, V)] per layer,
    [B, P, H, D], post-RoPE. The last layer's o_proj / MLP are skipped (never used)."""
    layers = vwe.get_vlm_model().text_model.layers
    hd = vwe.vlm.config.text_config.head_dim
    B = embs.shape[0]
    h = embs
    kv = []
    for l, layer in enumerate(layers):
        x = layer.input_layernorm(h).to(dtype=layer.self_attn.q_proj.weight.dtype)
        shp = (*x.shape[:-1], -1, layer.self_attn.head_dim)
        q = apply_rope(layer.self_attn.q_proj(x).view(shp), pos)
        k = apply_rope(layer.self_attn.k_proj(x).view(shp), pos)
        v = layer.self_attn.v_proj(x).view(shp)
        kv.append((k, v))
        if l == len(layers) - 1:
            break
        a = vwe.eager_attention_forward(att2d, B, hd, q, k, v)
        if a.dtype != layer.self_attn.o_proj.weight.dtype:
            a = a.to(layer.self_attn.o_proj.weight.dtype)
        o = layer.self_attn.o_proj(a)
        o += h
        r = o.clone()
        o = layer.post_attention_layernorm(o)
        o = layer.mlp(o)
        o += r
        h = o
    return kv


def _prefix_embs(model, img_feats, lang_tokens, lang_masks, state_emb=None):
    """[cam0 64 | cam1 64 | ... | lang (valid only) | state?] as VLAFlowMatching.embed_prefix (pads dropped)."""
    vwe = model.vlm_with_expert
    if torch.is_tensor(img_feats):
        img_feats = [img_feats]
    d = img_feats[0].shape[-1]
    img_emb = torch.cat([f * torch.tensor(d ** 0.5, dtype=f.dtype, device=f.device) for f in img_feats], 1)
    lm = lang_masks.bool()
    L = int(lm.sum(1).max())
    assert bool(lm[:, :L].all()) and not bool(lm[:, L:].any()), "expects right-padded, equal-length prompts"
    lang_emb = vwe.embed_language_tokens(lang_tokens[:, :L])
    lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
    parts = [img_emb, lang_emb.to(img_emb.dtype)]
    if state_emb is not None:
        parts.append(state_emb.to(img_emb.dtype))
    embs = torch.cat(parts, 1)
    B, n = embs.shape[:2]
    pad = torch.ones(B, n, dtype=torch.bool, device=embs.device)
    att = torch.zeros(B, n, dtype=torch.bool, device=embs.device)
    if state_emb is not None:
        att[:, -1] = True                    # image / lang never attend to the state token
    return embs, pad, MS.make_att_2d_masks(pad, att), torch.cumsum(pad, 1) - 1


def split_forward(policy, kv, pre_pad, state, actions, noise, time_):
    """Per-element flow-matching losses (B, 50, 32): the state token (VLM layers) + the action tokens (expert)
    against the no-grad [image, lang] prefix K/V. Same math as VLAFlowMatching.forward."""
    m = policy.model
    vwe = m.vlm_with_expert
    B = state.shape[0]
    te = time_[:, None, None]
    x_t = te * noise + (1 - te) * actions
    u_t = noise - actions
    h_s = m.state_proj(state)[:, None, :]
    h_a, _, _ = m.embed_suffix(x_t, time_)
    P, Ls, dev = pre_pad.shape[1], h_a.shape[1], h_a.device
    n_valid = pre_pad.sum(1, keepdim=True)
    pos_s = n_valid
    pos_a = n_valid + 1 + torch.arange(Ls, device=dev)[None]
    causal = torch.tril(torch.ones(Ls, Ls, dtype=torch.bool, device=dev))
    m_state = torch.cat([pre_pad, torch.ones(B, 1, dtype=torch.bool, device=dev)], 1)[:, None, :]
    m_sa = torch.cat([
        torch.cat([m_state, torch.zeros(B, 1, Ls, dtype=torch.bool, device=dev)], 2),
        torch.cat([pre_pad[:, None, :].expand(B, Ls, P), torch.ones(B, Ls, 1, dtype=torch.bool, device=dev),
                   causal[None].expand(B, Ls, Ls)], 2)], 1)
    m_cross = m_state.expand(B, Ls, P + 1)
    vlm_layers, exp_layers = vwe.get_vlm_model().text_model.layers, vwe.lm_expert.layers
    hd = vwe.vlm.config.text_config.head_dim
    attn = vwe.eager_attention_forward
    nL = vwe.num_vlm_layers
    for l in range(nL):
        vl, el = vlm_layers[l], exp_layers[l]
        Kp, Vp = kv[l]
        xs = vl.input_layernorm(h_s).to(dtype=vl.self_attn.q_proj.weight.dtype)
        shp = (B, 1, -1, vl.self_attn.head_dim)
        q_s = apply_rope(vl.self_attn.q_proj(xs).view(shp), pos_s)
        k_s = apply_rope(vl.self_attn.k_proj(xs).view(shp), pos_s)
        v_s = vl.self_attn.v_proj(xs).view(shp)
        xa = el.input_layernorm(h_a).to(dtype=el.self_attn.q_proj.weight.dtype)
        eshp = (B, Ls, -1, el.self_attn.head_dim)
        if vwe.self_attn_every_n_layers > 0 and l % vwe.self_attn_every_n_layers == 0:
            q_a = apply_rope(el.self_attn.q_proj(xa).view(eshp), pos_a)
            k_a = apply_rope(el.self_attn.k_proj(xa).view(eshp), pos_a)
            v_a = el.self_attn.v_proj(xa).view(eshp)
            att = attn(m_sa, B, hd, torch.cat([q_s, q_a], 1), torch.cat([Kp.to(k_s.dtype), k_s, k_a], 1),
                       torch.cat([Vp.to(v_s.dtype), v_s, v_a], 1))
            att_s, att_a = att[:, :1], att[:, 1:]
        else:
            Ks = torch.cat([Kp.to(k_s.dtype), k_s], 1)
            Vs = torch.cat([Vp.to(v_s.dtype), v_s], 1)
            att_s = attn(m_state, B, hd, q_s, Ks, Vs)
            q_a = el.self_attn.q_proj(xa).view(eshp)
            k_e = el.self_attn.k_proj(Ks.to(dtype=el.self_attn.k_proj.weight.dtype).reshape(B, P + 1, -1)).view(
                B, P + 1, -1, el.self_attn.head_dim)
            v_e = el.self_attn.v_proj(Vs.to(dtype=el.self_attn.v_proj.weight.dtype).reshape(B, P + 1, -1)).view(
                B, P + 1, -1, el.self_attn.head_dim)
            q_a = apply_rope(q_a, pos_a - pos_a.min(1, keepdim=True).values)
            att_a = attn(m_cross, B, hd, q_a, k_e, v_e)
        outs = []
        for h, layer, a in ((h_s, vl, att_s), (h_a, el, att_a)):
            if l == nL - 1 and layer is vl:
                outs.append(None)
                continue
            if a.dtype != layer.self_attn.o_proj.weight.dtype:
                a = a.to(layer.self_attn.o_proj.weight.dtype)
            o = layer.self_attn.o_proj(a)
            o += h
            r = o.clone()
            o = layer.post_attention_layernorm(o)
            o = layer.mlp(o)
            o += r
            outs.append(o)
        h_s, h_a = outs
    out = vwe.lm_expert.norm(h_a)[:, -policy.config.chunk_size:].to(torch.float32)
    return F.mse_loss(u_t, m.action_out_proj(out), reduction="none")


class FastLoss:
    """loss = FastLoss(policy)(batch, img_feat) with the same noise / time sampling calls (same RNG order) and the
    same reduction as SmolVLAPolicy.forward. compile=True compiles the prefix pass and the grad pass."""

    def __init__(self, policy, compile=True):
        c = policy.config
        assert c.freeze_vision_encoder and c.train_expert_only, "the fast path requires a frozen VLM (expert-only)"
        assert c.attention_mode == "cross_attn" and not c.add_image_special_tokens and not c.adapt_to_pi_aloha
        self.policy = policy
        self._prefix = torch.compile(prefix_kv_layers) if compile else prefix_kv_layers
        self._grad = torch.compile(split_forward) if compile else split_forward

    def __call__(self, batch, img_feat, noise=None, time_=None):
        """img_feat: tensor (one camera) or list of tensors (config.image_features order)."""
        pol, m = self.policy, self.policy.model
        state = pol.prepare_state(batch)
        actions = pol.prepare_action(batch)
        if noise is None:
            noise = m.sample_noise(actions.shape, actions.device)
        if time_ is None:
            time_ = m.sample_time(actions.shape[0], actions.device)
        with torch.no_grad():
            embs, pad, att2d, pos = _prefix_embs(m, img_feat, batch["observation.language.tokens"],
                                                 batch["observation.language.attention_mask"])
            kv = self._prefix(m.vlm_with_expert, embs, att2d, pos)
            kv = [(k.detach(), v.detach()) for k, v in kv]
        losses = self._grad(pol, kv, pad, state, actions, noise, time_)
        losses = losses[:, :, : pol.config.action_feature.shape[0]]
        pad_a = batch.get("action_is_pad")
        if pad_a is None:
            return losses.mean()
        losses = losses * (~pad_a).unsqueeze(-1)
        return losses.sum() / ((~pad_a).sum() * losses.shape[-1]).clamp_min(1)


class FastChunker:
    """actions = FastChunker(policy)(batch_after_preprocessor, noise): same as model.sample_actions (normalised,
    (B, chunk, max_action_dim)). compile=True captures the 10 Euler steps in a CUDA graph (fixed batch shapes)."""

    def __init__(self, policy, compile=False):
        self.policy, self.m, self.cfg = policy, policy.model, policy.config
        self.vwe = self.m.vlm_with_expert
        self._euler = torch.compile(self.euler, mode="reduce-overhead", dynamic=False) if compile else self.euler

    @torch.no_grad()
    def prefix(self, batch):
        pol, vwe = self.policy, self.vwe
        feat = image_features(pol, batch)          # list over cameras
        state_emb = self.m.state_proj(pol.prepare_state(batch))[:, None, :]
        embs, pad, att2d, pos = _prefix_embs(self.m, feat, batch["observation.language.tokens"],
                                             batch["observation.language.attention_mask"], state_emb)
        kv = prefix_kv_layers(vwe, embs, att2d, pos)
        B, n = embs.shape[:2]
        out = []
        for l, (K, V) in enumerate(kv):
            el = vwe.lm_expert.layers[l]
            if vwe.self_attn_every_n_layers > 0 and l % vwe.self_attn_every_n_layers == 0:
                out.append((K, V))
            else:           # expert k/v projections of the (fixed) prefix: once per chunk, not once per step
                out.append((el.self_attn.k_proj(K.to(el.self_attn.k_proj.weight.dtype).reshape(B, n, -1)).view(B, n, -1, el.self_attn.head_dim),
                            el.self_attn.v_proj(V.to(el.self_attn.v_proj.weight.dtype).reshape(B, n, -1)).view(B, n, -1, el.self_attn.head_dim)))
        return out

    def denoise(self, kv, x_t, t):
        m, vwe = self.m, self.vwe
        h, _, _ = m.embed_suffix(x_t, t)
        B, Ls = h.shape[:2]
        P, dev = kv[0][0].shape[1], h.device
        pos_a = (P + torch.arange(Ls, device=dev))[None].expand(B, Ls)
        pos_e = torch.arange(Ls, device=dev)[None].expand(B, Ls)
        causal = torch.tril(torch.ones(Ls, Ls, dtype=torch.bool, device=dev))
        m_self = torch.cat([torch.ones(B, Ls, P, dtype=torch.bool, device=dev), causal[None].expand(B, Ls, Ls)], 2)
        m_cross = torch.ones(B, Ls, P, dtype=torch.bool, device=dev)
        hd = vwe.vlm.config.text_config.head_dim
        for l, el in enumerate(vwe.lm_expert.layers):
            x = el.input_layernorm(h).to(dtype=el.self_attn.q_proj.weight.dtype)
            shp = (B, Ls, -1, el.self_attn.head_dim)
            Kp, Vp = kv[l]
            if vwe.self_attn_every_n_layers > 0 and l % vwe.self_attn_every_n_layers == 0:
                q = apply_rope(el.self_attn.q_proj(x).view(shp), pos_a)
                k = apply_rope(el.self_attn.k_proj(x).view(shp), pos_a)
                v = el.self_attn.v_proj(x).view(shp)
                att = vwe.eager_attention_forward(m_self, B, hd, q, torch.cat([Kp, k], 1), torch.cat([Vp, v], 1))
            else:
                q = apply_rope(el.self_attn.q_proj(x).view(shp), pos_e)
                att = vwe.eager_attention_forward(m_cross, B, hd, q, Kp, Vp)
            if att.dtype != el.self_attn.o_proj.weight.dtype:
                att = att.to(el.self_attn.o_proj.weight.dtype)
            o = el.self_attn.o_proj(att)
            o += h
            r = o.clone()
            o = el.post_attention_layernorm(o)
            o = el.mlp(o)
            o += r
            h = o
        out = vwe.lm_expert.norm(h)[:, -self.cfg.chunk_size:].to(torch.float32)
        return m.action_out_proj(out)

    def euler(self, kv, noise):
        n = self.cfg.num_steps
        dt = -1.0 / n
        x = noise
        for s in range(n):
            t = torch.full((noise.shape[0],), 1.0 + s * dt, dtype=torch.float32, device=noise.device)
            x = x + dt * self.denoise(kv, x, t)
        return x

    @torch.no_grad()
    def __call__(self, batch, noise):
        return self._euler(self.prefix(batch), noise).clone()


# =============================================================================================== batched eval

class LazyRenderer:
    """Renders only when .want is True, otherwise returns the last image (which the harness never reads)."""

    def __init__(self, r):
        self.r, self.want, self.last = r, True, None

    def update_scene(self, *a, **k):
        if self.want or self.last is None:
            self.r.update_scene(*a, **k)

    def render(self, *a, **k):
        if self.want or self.last is None:
            self.last = self.r.render(*a, **k)
        return self.last

    def __getattr__(self, n):
        return getattr(self.r, n)


class NullRenderer:
    def __init__(self, h, w):
        self.z = np.zeros((h, w, 3), np.uint8)

    def update_scene(self, *a, **k):
        pass

    def render(self, *a, **k):
        return self.z

    def enable_depth_rendering(self):
        pass

    def disable_depth_rendering(self):
        pass

    def close(self):
        pass


def prep_env(env, spec, render10hz=True, no_wrist=True):
    """Returns the list of LazyRenderers (cameras the policy reads, rendered only on the policy's control step).
    no_wrist: the wrist camera is not a policy input -> not rendered at all."""
    lz = []
    if no_wrist and env._wrist_renderer is not None and not isinstance(env._wrist_renderer, NullRenderer):
        env._wrist_renderer.close()                 # frees its EGL context and host memory
        env._wrist_renderer = NullRenderer(spec.wr_h, spec.wr_w)
    if render10hz:
        for attr in ("_overhead_obs_renderer", "_wrist_renderer"):
            r = getattr(env, attr)
            if r is None or isinstance(r, NullRenderer):
                continue
            if not isinstance(r, LazyRenderer):
                r = LazyRenderer(r)
                setattr(env, attr, r)
            lz.append(r)
    return lz


def episode_gen(env, seed, max_sec=12.0, exec_mode="interp", stop_on_success=True, lazy=None, frame_hooks=(),
                save_frames=False):
    """eval_closed_loop.run_episode as a generator (same protocol and record): yields a frame, receives the action."""
    import eval_closed_loop as E
    import scene as S
    from so101_nexus.lerobot_dataset import dataset_row_to_sim_qpos
    gl = S.grip_limits(env)
    low, high = env.action_space.low, env.action_space.high
    lazy = list(lazy or [])
    for lz in lazy:
        lz.want = True
    obs, info = env.reset(seed=seed)
    u = env
    cube0 = u._get_object_pose()[:3].copy()
    tray = u._get_target_pos()[:3].copy()
    max_steps = int(max_sec * S.SIM_HZ)
    prev_q = obs["state"].astype(np.float64).copy()
    t_success, grasped, lifted, min_tcp_cube = -1.0, False, False, 1e9
    first_close = None
    low_xy, reach_xy_min = None, 1e9
    frames, traj = [], []
    t = 0
    while t < max_steps:
        fr = E.make_frame(obs, gl)
        for h in frame_hooks:
            fr = h(fr)
        if save_frames:
            frames.append(fr["observation.images.overhead"])
        a = yield fr
        q = dataset_row_to_sim_qpos(np.asarray(a, np.float64), gripper_limits_rad=gl)
        q = np.clip(q, low, high)
        if first_close is None and a[5] < 30.0 and t > 0:
            first_close = u._get_tcp_pose()[:3].copy()
        done = False
        for i in range(S.RECORD_EVERY):
            for lz in lazy:
                lz.want = (i == S.RECORD_EVERY - 1)
            qi = q if exec_mode == "hold" else prev_q + (q - prev_q) * (i + 1) / S.RECORD_EVERY
            obs, _r, _te, _tr, info = env.step(qi.astype(np.float32))
            t += 1
            tcp = u._get_tcp_pose()[:3]
            cube = u._get_object_pose()[:3]
            min_tcp_cube = min(min_tcp_cube, float(np.linalg.norm(tcp - cube)))
            if low_xy is None and tcp[2] < E.LOW_Z:
                low_xy = tcp[:2].copy()
            if not grasped and tcp[2] < 0.06:
                reach_xy_min = min(reach_xy_min, float(np.linalg.norm(tcp[:2] - cube0[:2])))
            grasped |= bool(info.get("is_grasped", 0.0))
            lifted |= float(info.get("lift_height", 0.0)) > 0.02
            if info.get("success", False) and t_success < 0:
                t_success = t / S.SIM_HZ
                if stop_on_success:
                    done = True
                    break
        traj.append(u._get_tcp_pose()[:3].round(4).tolist())
        prev_q = q
        if done:
            break
    cube = u._get_object_pose()[:3]
    success = t_success >= 0
    phase = None if success else ("reach" if min_tcp_cube > 0.02 else "grasp" if not grasped else
                                  "lift" if not lifted else "place")
    rec = dict(seed=int(seed), success=success, t_success=t_success, sim_sec=t / S.SIM_HZ,
               final_dist=float(np.linalg.norm(cube[:2] - tray[:2])), cube0_xy=cube0[:2].round(4).tolist(),
               min_tcp_cube=min_tcp_cube, grasped=grasped, lifted=lifted, fail_phase=phase,
               start_qpos=env.extra_info.get("start_qpos"))
    if first_close is not None:
        fc = first_close[:2]
        rec["first_close_xy"] = fc.round(4).tolist()
        rec["first_close_to_cube"] = float(np.linalg.norm(fc - cube0[:2]))
        ds = [float(np.linalg.norm(fc - np.array(d["xy"]))) for d in env.extra_info.get("distractors", []) if d.get("placed")]
        if ds:
            rec["first_close_to_distractors"] = ds
    rec["reach_xy_min"] = reach_xy_min if reach_xy_min < 1e8 else None
    if low_xy is not None:
        e = low_xy - cube0[:2]
        rec["low_xy"] = low_xy.round(4).tolist()
        rec["reach_xy_low"] = float(np.linalg.norm(e))
        objs = {f"distr{i}": np.array(d["xy"]) for i, d in enumerate(env.extra_info.get("distractors", [])) if d.get("placed")}
        objs.update(E.landmarks(env.spec, low_xy))
        near = {}
        for k, o in objs.items():
            v = o - cube0[:2]
            near[k] = dict(dist_low=float(np.linalg.norm(low_xy - o)), dist_cube=float(np.linalg.norm(v)),
                           cos_err=float(e @ v / (np.linalg.norm(e) * np.linalg.norm(v) + 1e-9)))
        if near:
            rec["objects"] = near
    if env.extra_info.get("distractors"):
        rec["distractors"] = env.extra_info["distractors"]
    rec["tcp_traj_10hz"] = traj
    if save_frames:
        rec["_frames"] = frames
    return rec


class BatchedPolicy:
    """Batched SmolVLAWrapper.act + select_action queue. `w` needs .policy .pre .post .device .image_keys."""

    def __init__(self, w, fast_chunk=False, compile=False, pad_to=0):
        self.w, self.policy, self.pre, self.post = w, w.policy, w.pre, w.post
        self.n_action_steps = self.policy.config.n_action_steps
        self.fc = FastChunker(self.policy, compile=compile) if fast_chunk else None
        self.pad_to = pad_to if (fast_chunk and compile) else 0
        self.stats = dict(chunk_calls=0, chunk_rows=0, chunk_s=0.0)

    def chunk(self, frames, gens):
        import eval_closed_loop as E
        import scene as S
        from lerobot.policies.utils import prepare_observation_for_inference
        t0 = time.perf_counter()
        cfg = self.policy.config
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            obs = [prepare_observation_for_inference(
                {k: v for k, v in f.items() if k == "observation.state" or k in self.w.image_keys},
                self.w.device, S.TASK, E.ROBOT_TYPE) for f in frames]
            batch = {k: (torch.cat([o[k] for o in obs]) if torch.is_tensor(obs[0][k]) else
                         [o[k] for o in obs] if k == "task" else obs[0][k]) for k in obs[0]}
            batch = self.pre(batch)
            noise = torch.cat([torch.normal(0.0, 1.0, size=(1, cfg.chunk_size, cfg.max_action_dim), dtype=torch.float32,
                                            device=self.w.device, generator=g) for g in gens])
            K = len(frames)
            if self.fc is None:
                acts = self.policy._get_action_chunk(batch, noise)
            else:
                if self.pad_to and K < self.pad_to:     # fixed shapes for the CUDA graph
                    rep = self.pad_to - K
                    batch = {k: (torch.cat([v, v[:1].expand(rep, *v.shape[1:])]) if torch.is_tensor(v) and v.ndim > 0
                                 and v.shape[0] == K else v) for k, v in batch.items()}
                    noise = torch.cat([noise, noise[:1].expand(rep, *noise.shape[1:])])
                acts = self.fc(batch, noise)[:K, :, : cfg.action_feature.shape[0]]
            acts = self.post(acts[:, : self.n_action_steps])
            out = acts.float().cpu().numpy().astype(np.float64)
        self.stats["chunk_calls"] += 1
        self.stats["chunk_rows"] += K
        self.stats["chunk_s"] += time.perf_counter() - t0
        return out


def run_condition_batched(bpol: BatchedPolicy, spec, seeds, pool=10, render10hz=True, no_wrist=None, sync=True,
                          envs=None, frame_hooks=(), gif_n=0, **kw):
    """Same outputs as eval_closed_loop.run_condition (summary with 'sec', per-episode records in seed order).
    Episodes run in a pool of envs; with sync, new episodes start only on re-plan ticks so batches stay full."""
    import eval_closed_loop as E
    import scene as S
    own = envs is None
    if own:
        envs = [S.make_env(spec) for _ in range(max(1, min(pool, len(seeds))))]
    if no_wrist is None:
        no_wrist = "observation.images.wrist" not in bpol.w.image_keys
    lazies = [prep_env(e, spec, render10hz, no_wrist) for e in envs]
    t0 = time.perf_counter()
    pending = list(enumerate(seeds))
    slots, recs, idle, tick = {}, {}, [], 0

    def start(sl):
        i, sd = pending.pop(0)
        g = torch.Generator(device=bpol.w.device)
        g.manual_seed(int(sd) % (2 ** 31))              # = torch.manual_seed(seed) in SmolVLAWrapper.reset
        eg = episode_gen(envs[sl], sd, lazy=lazies[sl], frame_hooks=frame_hooks, save_frames=i < gif_n, **kw)
        slots[sl] = dict(i=i, g=g, eg=eg, frame=next(eg), q=deque())

    for sl in range(len(envs)):
        if pending:
            start(sl)
    while slots or pending:
        if idle and pending and (not sync or tick % bpol.n_action_steps == 0):
            for sl in list(idle):
                if pending:
                    start(sl)
                    idle.remove(sl)
        if not slots:
            tick += 1
            continue
        need = [sl for sl, d in slots.items() if not d["q"]]
        if need:
            acts = bpol.chunk([slots[sl]["frame"] for sl in need], [slots[sl]["g"] for sl in need])
            for sl, a in zip(need, acts):
                slots[sl]["q"].extend(a)
        for sl in list(slots):
            d = slots[sl]
            try:
                d["frame"] = d["eg"].send(d["q"].popleft())
            except StopIteration as st:
                recs[d["i"]] = st.value
                del slots[sl]
                idle.append(sl)
        tick += 1
    if own:
        for e in envs:
            e.close()
    recs = [recs[k] for k in range(len(seeds))]
    s = E.summarise(recs)
    s["sec"] = time.perf_counter() - t0
    return s, recs
