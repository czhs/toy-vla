"""Shared tools for the arm's fold: a scene that renders a layout (seed) and computes forward kinematics of an action chunk, and a model
wrapper around the policy (prefix K/V of a batch of frames, expert K/V projection, 10-step Euler with per-case noise). Everything acts on the
checkpoint in paths.POLICY unless another is given.

Layout of the prefix: post-RoPE K/V rows [B, P, 5, 64], P = 64 image tokens (8 x 8 tokens of 64 x 64 px, token k = 8*row + col) + valid language
tokens + 1 state token. The expert's odd (cross-attention) layers read them through their own k_proj / v_proj."""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch

import mujoco  # noqa: E402

import eval_closed_loop as E  # noqa: E402
import fast_core as FC  # noqa: E402
import paths  # noqa: E402
import scene as S  # noqa: E402
from lerobot.policies.smolvla.smolvlm_with_expert import apply_rope  # noqa: E402
from so101_nexus.lerobot_dataset import dataset_row_to_sim_qpos, sim_qpos_to_dataset_row  # noqa: E402

CAM = paths.ASSETS / "camera.json"
ODD = list(range(1, 16, 2))
NIMG, GRID, TOKPX = 64, 8, 64
LOW_Z = E.LOW_Z


def layouts():
    return E.load_layouts(None, str(CAM), "shadow")


def spec_of(cond, L=None, **over):
    L = L or layouts()
    d = dict(L["conditions"][cond])
    d.update(over)
    return S.SceneSpec.from_dict(d)


# =============================================================================================== scenes

class Scene:
    """One env for a condition. render(seed, cube, distractors) -> dict(img, state, cube_xy, distractors,
    cov_cube [64], cov_distr [nd, 64], cov_arm [64], cube_uv)."""

    def __init__(self, spec):
        self.spec = spec
        self.env = S.make_env(spec)
        e = self.env
        self.gl = S.grip_limits(e)
        self.rend = e._overhead_obs_renderer
        self.cam = e._overhead_obs_cam
        self.seg = mujoco.Renderer(e.model, height=spec.ov_h, width=spec.ov_w)
        self.seg.enable_segmentation_rendering()
        self.robot_geom = np.asarray(e._robot_geom_mask, bool)
        self.fk = mujoco.MjData(e.model)

    def _img(self):
        self.rend.update_scene(self.env.data, camera=self.cam)
        self.rend.disable_depth_rendering()
        return self.rend.render().copy()

    def _segids(self):
        self.seg.update_scene(self.env.data, camera=self.cam)
        s = self.seg.render()
        gid = s[..., 0].astype(np.int64)
        gid[s[..., 1] != int(mujoco.mjtObj.mjOBJ_GEOM)] = -1
        return gid

    @staticmethod
    def token_cov(mask):
        return mask.reshape(GRID, TOKPX, GRID, TOKPX).mean((1, 3)).reshape(-1)

    def render(self, seed, cube=True, distractors=True, arm_cov=True, cube_rgb=None):
        e = self.env
        obs, _ = e.reset(seed=seed)
        m, d = e.model, e.data
        slot = e._slots[e._target_slot_idx]
        cube_gids = list(slot.geom_ids)
        if cube_rgb is not None:
            for g in cube_gids:
                m.geom_rgba[g] = [*S._c(cube_rgb), 1.0]
        dinfo = [x for x in e.extra_info.get("distractors", [])]
        dg = [sl["gid"] for sl in e._dslots]
        placed = [x.get("placed", False) for x in dinfo]
        changed = cube_rgb is not None
        if not cube:
            d.qpos[slot.qpos_addr + 2] = -1.0
            changed = True
        saved = None
        if not distractors and e._dslots:
            saved = [(sl["bid"], m.body_pos[sl["bid"]].copy()) for sl in e._dslots]
            for sl in e._dslots:
                m.body_pos[sl["bid"]] = [0, 0, -1]
            changed = True
        if changed:
            mujoco.mj_forward(m, d)
            img = self._img()
        else:
            img = obs["overhead_camera"].copy()
        gid = self._segids()
        out = dict(img=img, seed=int(seed),
                   state=sim_qpos_to_dataset_row(obs["state"].astype(np.float64), gripper_limits_rad=self.gl).astype(np.float32),
                   cube_xy=np.array(e.extra_info["cube_xy"], float),
                   distractors=[x for x in dinfo if x.get("placed")])
        out["cov_cube"] = self.token_cov(np.isin(gid, cube_gids).astype(float))
        out["cov_distr"] = np.stack([self.token_cov((gid == g).astype(float)) for g, p in zip(dg, placed) if p]) \
            if any(placed) and distractors else np.zeros((0, NIMG))
        if arm_cov:
            rg = np.where(self.robot_geom)[0]
            out["cov_arm"] = self.token_cov(np.isin(gid, rg).astype(float))
        # cube pixel centre from the segmentation (visible pixels); None if hidden
        cm = np.isin(gid, cube_gids)
        out["cube_uv"] = np.array([np.nonzero(cm)[1].mean(), np.nonzero(cm)[0].mean()]) if cm.any() else \
            self.world_uv(np.r_[out["cube_xy"], 0.0125])
        if saved:
            for bid, p in saved:
                m.body_pos[bid] = p
        return out

    def world_uv(self, xyz):
        """pixel (u, v) of a world point through the policy camera."""
        m, d = self.env.model, self.env.data
        cid = self.env._cam_id
        mujoco.mj_forward(m, d)
        p, R = d.cam_xpos[cid], d.cam_xmat[cid].reshape(3, 3)
        c = R.T @ (np.asarray(xyz, float) - p)        # camera frame: x right, y up, -z forward
        f = (self.spec.ov_h / 2) / math.tan(math.radians(m.cam_fovy[cid] / 2))
        return np.array([self.spec.ov_w / 2 + f * c[0] / -c[2], self.spec.ov_h / 2 - f * c[1] / -c[2]])

    def tcp_of(self, acts):
        """TCP xyz for each action row (dataset units)."""
        e, m, d = self.env, self.env.model, self.fk
        d.qpos[:] = e.data.qpos
        out = []
        for a in acts:
            q = dataset_row_to_sim_qpos(np.asarray(a, np.float64), gripper_limits_rad=self.gl)
            d.qpos[e._qpos_addrs] = q
            mujoco.mj_kinematics(m, d)
            out.append(d.site_xpos[e._tcp_site_id].copy())
        return np.array(out)

    def close(self):
        self.seg.close()
        self.env.close()


def reach_point(tcp):
    """xy where the open-loop TCP path first goes below LOW_Z (else at its lowest point)."""
    i = np.nonzero(tcp[:, 2] < LOW_Z)[0]
    j = int(i[0]) if len(i) else int(np.argmin(tcp[:, 2]))
    return tcp[j, :2], bool(len(i))


# =============================================================================================== model

class Model:
    def __init__(self, ckpt=str(paths.POLICY), device="cuda"):
        self.w = E.SmolVLAWrapper(ckpt, device)
        self.pol = self.w.policy
        self.m = self.pol.model
        self.vwe = self.m.vlm_with_expert
        self.cfg = self.pol.config
        self.dev = torch.device(device)

    @torch.no_grad()
    def prefix(self, imgs, states):
        """imgs uint8 [B,H,W,3], states [B,6] dataset units -> kv list [(K,V)] per VLM layer, [B,P,5,64] post-RoPE,
        P = 64 image + valid language + 1 state (pads dropped, as fast_core; image positions 0-63).
        $TOYVLA_PREFIX_CHUNK (default 32) splits the SigLIP/VLM pass into micro-batches (memory on small GPUs)."""
        ch = int(os.environ.get("TOYVLA_PREFIX_CHUNK", "32"))
        if len(imgs) > ch:
            parts = [self.prefix(imgs[i:i + ch], states[i:i + ch]) for i in range(0, len(imgs), ch)]
            return [(torch.cat([p[l][0] for p in parts]), torch.cat([p[l][1] for p in parts])) for l in range(len(parts[0]))]
        from lerobot.policies.utils import prepare_observation_for_inference
        B = len(imgs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            obs = [prepare_observation_for_inference({"observation.state": np.asarray(st, np.float32),
                                                      "observation.images.overhead": im}, self.dev, S.TASK,
                                                     E.ROBOT_TYPE) for im, st in zip(imgs, states)]
            batch = {k: (torch.cat([o[k] for o in obs]) if torch.is_tensor(obs[0][k]) else
                         [o[k] for o in obs] if k == "task" else obs[0][k]) for k in obs[0]}
            b = self.w.pre(batch)
            feat = FC.image_features(self.pol, b)
            state_emb = self.m.state_proj(self.pol.prepare_state(b))[:, None, :]
            embs, pad, att2d, pos = FC._prefix_embs(self.m, feat, b["observation.language.tokens"],
                                                    b["observation.language.attention_mask"], state_emb)
            kv = FC.prefix_kv_layers(self.vwe, embs, att2d, pos)
        return [(k.clone(), v.clone()) for k, v in kv]

    def expert_kv(self, kv):
        vwe = self.vwe
        out = []
        for l, (K, V) in enumerate(kv):
            el = vwe.lm_expert.layers[l]
            B, n = K.shape[:2]
            if l % vwe.self_attn_every_n_layers == 0:
                out.append((K, V))
            else:
                out.append((el.self_attn.k_proj(K.to(el.self_attn.k_proj.weight.dtype).reshape(B, n, -1)).view(B, n, -1, 64),
                            el.self_attn.v_proj(V.to(el.self_attn.v_proj.weight.dtype).reshape(B, n, -1)).view(B, n, -1, 64)))
        return out

    @torch.no_grad()
    def denoise(self, ekv, x_t, t, rec=None):
        """FastChunker.denoise with optional recording: rec = dict(layers=[...]) gets rec['att'][l] = probs
        [B, heads, Ls, P] (float32) and rec['logit'][l] (pre-mask logits) for the cross layers listed."""
        m, vwe = self.m, self.vwe
        h, _, _ = m.embed_suffix(x_t, t)
        B, Ls = h.shape[:2]
        P, dev = ekv[0][0].shape[1], h.device
        pos_a = (P + torch.arange(Ls, device=dev))[None].expand(B, Ls)
        pos_e = torch.arange(Ls, device=dev)[None].expand(B, Ls)
        causal = torch.tril(torch.ones(Ls, Ls, dtype=torch.bool, device=dev))
        m_self = torch.cat([torch.ones(B, Ls, P, dtype=torch.bool, device=dev), causal[None].expand(B, Ls, Ls)], 2)
        m_cross = torch.ones(B, Ls, P, dtype=torch.bool, device=dev)
        hd = 64
        for l, el in enumerate(vwe.lm_expert.layers):
            x = el.input_layernorm(h).to(dtype=el.self_attn.q_proj.weight.dtype)
            shp = (B, Ls, -1, el.self_attn.head_dim)
            Kp, Vp = ekv[l]
            if l % vwe.self_attn_every_n_layers == 0:
                q = apply_rope(el.self_attn.q_proj(x).view(shp), pos_a)
                k = apply_rope(el.self_attn.k_proj(x).view(shp), pos_a)
                v = el.self_attn.v_proj(x).view(shp)
                att = vwe.eager_attention_forward(m_self, B, hd, q, torch.cat([Kp, k], 1), torch.cat([Vp, v], 1))
            else:
                q = apply_rope(el.self_attn.q_proj(x).view(shp), pos_e)
                if rec is not None and l in rec["layers"]:
                    nk = Kp.shape[2]
                    qq = q.float().transpose(1, 2)                                  # B, H, Ls, D
                    kk = Kp.float().repeat_interleave(qq.shape[1] // nk, dim=2).transpose(1, 2)
                    lg = (qq @ kk.transpose(2, 3)) * hd ** -0.5
                    rec.setdefault("att", {})[l] = torch.softmax(lg, -1)
                    rec.setdefault("logit", {})[l] = lg
                    rec.setdefault("q", {})[l] = qq
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

    @torch.no_grad()
    def act(self, kv, noise, rec=None):
        """kv (possibly edited VLM K/V) -> unnormalised actions [B,50,6] (dataset units). rec: attention record at
        step 0 (t = 1)."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ekv = self.expert_kv(kv)
            n = self.cfg.num_steps
            dt = -1.0 / n
            x = noise
            for s in range(n):
                t = torch.full((noise.shape[0],), 1.0 + s * dt, dtype=torch.float32, device=noise.device)
                x = x + dt * self.denoise(ekv, x, t, rec=rec if s == 0 else None)
            a = self.post(x[:, :, :6])
        return a

    def post(self, a):
        return self.w.post(a).float()

    def noise(self, seeds):
        out = []
        for s in seeds:
            g = torch.Generator(device=self.dev).manual_seed(int(s) % (2 ** 31))
            out.append(torch.normal(0.0, 1.0, size=(1, self.cfg.chunk_size, self.cfg.max_action_dim),
                                    dtype=torch.float32, device=self.dev, generator=g))
        return torch.cat(out)


def suffix_emb(Mdl, x_t, t_emb):
    from lerobot.policies.smolvla.modeling_smolvla import create_sinusoidal_pos_embedding
    m = Mdl.m
    ae = m.action_in_proj(x_t)
    te = create_sinusoidal_pos_embedding(t_emb, Mdl.vwe.expert_hidden_size, Mdl.cfg.min_period, Mdl.cfg.max_period,
                                         device=ae.device).to(ae.dtype)
    te = te[:, None, :].expand_as(ae)
    h = m.action_time_mlp_in(torch.cat([ae, te], 2))
    return m.action_time_mlp_out(torch.nn.functional.silu(h))


def tvals(Mdl):
    n = Mdl.cfg.num_steps
    return np.array([1.0 - s / n for s in range(n)])


def unnorm_scale(Mdl):
    """per-dim std and mean of the action normaliser (dataset units per normalised unit)."""
    z = torch.zeros(1, 1, 6, device=Mdl.dev)
    o = torch.ones(1, 1, 6, device=Mdl.dev)
    mu = Mdl.post(z)[0, 0]
    sd = Mdl.post(o)[0, 0] - mu
    return mu.cpu().numpy(), sd.cpu().numpy()
