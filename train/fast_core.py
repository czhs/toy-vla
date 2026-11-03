"""Pieces of the fast toy trainer. Same functions as VLAFlowMatching.forward / sample_actions of lerobot 0.3.3 (frozen SigLIP + connector + 16 vision-language layers, trainable expert, cross-attention with
self-attention every second layer, eager fp32-softmax attention), reorganised:
  image features   connector(SigLIP(img)) of every pool scene, read from a feature cache (build_feature_cache.py) or computed on the fly under no_grad (same call, same batch 32).
  prefix K/V       the 16 layers over [64 image tokens, 9 prompt tokens] under no_grad; the 39 language pads are dropped (they are masked out of every valid row).
  grad pass        the state token through the 16 layers (attending to the no-grad K/V) + the 50 action tokens through the expert. Positions, masks, RoPE, dtypes and op order follow lerobot.
  eval             model.sample_actions with cached panel features (exact), or a split / hoisted / compiled chunker.
Only bf16 reduction order (batch shape) differs from lerobot's own training step."""
from __future__ import annotations

import hashlib
import json
import math
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from lerobot.policies.smolvla.smolvlm_with_expert import apply_rope
from toyvla.paths import FEATURE_CACHE

FEAT_TOKENS, FEAT_DIM = 64, 960
FEAT_ROOT = FEATURE_CACHE


# ============================================================================================ legacy-exact sample order

class LegacyIndexStream:
    """Pool-index batches in exactly the order the plain lerobot training step's DataLoader(Stream(seed), batch_size=B, num_workers=W) yields
    them: batch k comes from worker k % W, and worker w draws default_rng([seed, w]).integers(POOL) item by item."""

    def __init__(self, seed, workers, batch, pool):
        self.rngs = [np.random.default_rng([seed, w]) for w in range(max(1, workers))]
        self.batch, self.pool, self.k = batch, pool, 0

    def __iter__(self): return self

    def __next__(self):
        r = self.rngs[self.k % len(self.rngs)]; self.k += 1
        return np.fromiter((int(r.integers(self.pool)) for _ in range(self.batch)), np.int64, self.batch)


# ============================================================================================ feature cache

def cache_key(block_colour='red'):
    return f'toy_plain_{block_colour}'


class FeatCache:
    """[POOL, 64, 960] bf16 connector(SigLIP(render(pool_scene(i)))) (stored as uint16) + angles [POOL, 4] float32."""

    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / 'meta.json').read_text())
        n = self.meta['n']
        self.arr = np.memmap(self.path / 'features.u16', dtype=np.uint16, mode='r', shape=(n, FEAT_TOKENS, FEAT_DIM))
        a = self.path / 'angles.npy'
        self.angles = np.load(a if a.exists() else self.path / 'angles_partial.npy')

    def check(self, expect):
        bad = {k: (self.meta.get(k), v) for k, v in expect.items() if self.meta.get(k) != v}
        assert not bad, f'feature cache {self.path} does not match this run: {bad}'
        done = np.load(self.path / 'done.npy')
        assert done.all(), f'feature cache incomplete: {int(done.sum())}/{done.size} blocks; run build_feature_cache.py'

    def gather(self, idx):
        order = np.argsort(idx, kind='stable')
        out = np.empty((len(idx), FEAT_TOKENS, FEAT_DIM), np.uint16)
        out[order] = self.arr[idx[order]]
        return torch.from_numpy(out).view(torch.bfloat16)


class Prefetch:
    """Background reader: index batches -> (idx, pinned bf16 features) in order, `threads` batches read concurrently."""

    def __init__(self, cache, batches, depth=8, threads=3):
        self.q = queue.Queue(depth)
        self.cache, self.it = cache, batches
        self.pool = ThreadPoolExecutor(threads)
        self.th = threading.Thread(target=self._run, daemon=True); self.th.start()

    def _read(self, idx): return idx, self.cache.gather(idx).pin_memory()

    def _run(self):
        pend = []
        for idx in self.it:
            pend.append(self.pool.submit(self._read, idx))
            if len(pend) >= 4: self.q.put(pend.pop(0).result())

    def __next__(self): return self.q.get()


def sha256_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ============================================================================================ model pieces

def image_features(vwe, img):
    """= SmolVLMWithExpertModel.embed_image (connector(SigLIP(img))), bf16 under autocast. img: [B, 3, 512, 512] in [-1, 1]."""
    return vwe.embed_image(img)


class PrefixEmb:
    """[image 64 | prompt 9] prefix embeddings as VLAFlowMatching.embed_prefix builds them (pads dropped): the image
    features times sqrt(960) in bf16, the language embedding times sqrt(960) in fp32, concatenated -> fp32."""

    def __init__(self, model, lang_ids, lang_mask):
        vwe = model.vlm_with_expert
        L = int(lang_mask.sum())
        assert bool(lang_mask[0, :L].all()) and not bool(lang_mask[0, L:].any()), 'expects a right-padded prompt'
        with torch.no_grad():
            le = vwe.embed_language_tokens(lang_ids[:1, :L].to('cuda'))
            self.lang = le * math.sqrt(le.shape[-1])                       # fp32 [1, L, 960]
        self.L = L
        self.scale = torch.tensor(FEAT_DIM ** 0.5, dtype=torch.bfloat16, device='cuda')

    def __call__(self, feat):
        B = feat.shape[0]
        img = feat * self.scale                                             # bf16, as legacy img_emb * tensor(sqrt d, bf16)
        return torch.cat([img, self.lang.expand(B, -1, -1)], 1)             # promotes to fp32, as the legacy cat


def _layer_tail(layer, a, h):
    if a.dtype != layer.self_attn.o_proj.weight.dtype:
        a = a.to(layer.self_attn.o_proj.weight.dtype)
    o = layer.self_attn.o_proj(a)
    o += h
    r = o.clone()
    o = layer.post_attention_layernorm(o)
    o = layer.mlp(o)
    o += r
    return o


def prefix_kv(vwe, embs):
    """VLM prefill of the all-valid [image, prompt] prefix (full bidirectional attention); [(K, V)] per layer, K post-RoPE,
    [B, P, 5, 64]. The last layer's attention output / MLP are skipped (never read)."""
    layers = vwe.get_vlm_model().text_model.layers
    hd = vwe.vlm.config.text_config.head_dim
    B, P = embs.shape[:2]
    pos = torch.arange(P, device=embs.device)[None].expand(B, P)
    full = torch.ones(B, P, P, dtype=torch.bool, device=embs.device)
    h, kv = embs, []
    for l, layer in enumerate(layers):
        x = layer.input_layernorm(h).to(dtype=layer.self_attn.q_proj.weight.dtype)
        shp = (B, P, -1, layer.self_attn.head_dim)
        k = apply_rope(layer.self_attn.k_proj(x).view(shp), pos)
        v = layer.self_attn.v_proj(x).view(shp)
        kv.append((k, v))
        if l == len(layers) - 1: break
        q = apply_rope(layer.self_attn.q_proj(x).view(shp), pos)
        h = _layer_tail(layer, vwe.eager_attention_forward(full, B, hd, q, k, v), h)
    return kv


def split_losses(model, kv, state, actions, noise, time_):
    """Per-element flow-matching losses (B, 50, 32) = VLAFlowMatching.forward with the [image, prompt] K/V given (no grad).
    Grad flows through state_proj, the state token's VLM path (frozen weights) and the expert."""
    vwe = model.vlm_with_expert
    B = state.shape[0]
    te = time_[:, None, None]
    x_t = te * noise + (1 - te) * actions
    u_t = noise - actions
    h_s = model.state_proj(state)[:, None, :].float()          # the legacy prefix cat promotes the state token to fp32
    h_a, _, _ = model.embed_suffix(x_t, time_)
    P, Ls, dev = kv[0][0].shape[1], h_a.shape[1], h_a.device
    pos_s = torch.full((B, 1), P, dtype=torch.long, device=dev)
    pos_a = (P + 1 + torch.arange(Ls, device=dev))[None].expand(B, Ls)
    pos_e = torch.arange(Ls, device=dev)[None].expand(B, Ls)
    causal = torch.tril(torch.ones(Ls, Ls, dtype=torch.bool, device=dev))
    m_sa = torch.cat([torch.cat([torch.ones(B, 1, P + 1, dtype=torch.bool, device=dev), torch.zeros(B, 1, Ls, dtype=torch.bool, device=dev)], 2),
                      torch.cat([torch.ones(B, Ls, P + 1, dtype=torch.bool, device=dev), causal[None].expand(B, Ls, Ls)], 2)], 1)
    m_state = torch.ones(B, 1, P + 1, dtype=torch.bool, device=dev)
    m_cross = torch.ones(B, Ls, P + 1, dtype=torch.bool, device=dev)
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
            k_e = el.self_attn.k_proj(Ks.to(dtype=el.self_attn.k_proj.weight.dtype).reshape(B, P + 1, -1)).view(B, P + 1, -1, el.self_attn.head_dim)
            v_e = el.self_attn.v_proj(Vs.to(dtype=el.self_attn.v_proj.weight.dtype).reshape(B, P + 1, -1)).view(B, P + 1, -1, el.self_attn.head_dim)
            q_a = apply_rope(q_a, pos_e)
            att_a = attn(m_cross, B, hd, q_a, k_e, v_e)
        h_s = None if l == nL - 1 else _layer_tail(vl, att_s, h_s)
        h_a = _layer_tail(el, att_a, h_a)
    out = vwe.lm_expert.norm(h_a)[:, -Ls:].to(torch.float32)
    return F.mse_loss(u_t, model.action_out_proj(out), reduction='none')


class SplitLoss:
    """losses = SplitLoss(model, prefix_emb)(feat, state, actions, noise, time_) -> (B, 50, 32), same as
    model.forward(*inputs, noise, time) with the image features given. compile=True compiles both passes."""

    def __init__(self, model, pe, compile=True):
        self.model, self.pe = model, pe
        self.vwe = model.vlm_with_expert
        self._prefix = torch.compile(prefix_kv) if compile else prefix_kv
        self._grad = torch.compile(split_losses) if compile else split_losses

    def kv(self, feat):
        with torch.no_grad():
            kv = self._prefix(self.vwe, self.pe(feat))
        return [(k.detach(), v.detach()) for k, v in kv]

    def from_kv(self, kv, state, actions, noise, time_):
        return self._grad(self.model, kv, state, actions, noise, time_)

    def __call__(self, feat, state, actions, noise, time_):
        return self.from_kv(self.kv(feat), state, actions, noise, time_)


# ============================================================================================ fast inference (eval)

class FastChunker:
    """= model.sample_actions(images, masks, ids, mask, state, noise) with image features given: [image, prompt] prefix
    without pads, the state token appended (it attends to the prefix and itself), the expert's k/v projections of the
    prefix hoisted out of the 10 Euler steps (computed once per chunk instead of 10 times), optionally compiled."""

    def __init__(self, model, pe, compile=False):
        self.m, self.pe, self.vwe = model, pe, model.vlm_with_expert
        self.n = model.config.num_steps
        self._euler = torch.compile(self.euler, dynamic=False) if compile else self.euler

    @torch.no_grad()
    def prefix(self, feat, state):
        vwe, m = self.vwe, self.m
        embs = torch.cat([self.pe(feat), m.state_proj(state)[:, None, :].float()], 1)
        layers = vwe.get_vlm_model().text_model.layers
        hd = vwe.vlm.config.text_config.head_dim
        B, P = embs.shape[:2]
        pos = torch.arange(P, device=embs.device)[None].expand(B, P)
        msk = torch.ones(B, P, P, dtype=torch.bool, device=embs.device); msk[:, :-1, -1] = False   # img/lang never see the state
        h, out = embs, []
        for l, layer in enumerate(layers):
            x = layer.input_layernorm(h).to(dtype=layer.self_attn.q_proj.weight.dtype)
            shp = (B, P, -1, layer.self_attn.head_dim)
            q = apply_rope(layer.self_attn.q_proj(x).view(shp), pos)
            k = apply_rope(layer.self_attn.k_proj(x).view(shp), pos)
            v = layer.self_attn.v_proj(x).view(shp)
            el = vwe.lm_expert.layers[l]
            if vwe.self_attn_every_n_layers > 0 and l % vwe.self_attn_every_n_layers == 0:
                out.append((k, v))
            else:
                out.append((el.self_attn.k_proj(k.to(el.self_attn.k_proj.weight.dtype).reshape(B, P, -1)).view(B, P, -1, el.self_attn.head_dim),
                            el.self_attn.v_proj(v.to(el.self_attn.v_proj.weight.dtype).reshape(B, P, -1)).view(B, P, -1, el.self_attn.head_dim)))
            if l < len(layers) - 1:
                h = _layer_tail(layer, vwe.eager_attention_forward(msk, B, hd, q, k, v), h)
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
            h = _layer_tail(el, att, h)
        out = vwe.lm_expert.norm(h)[:, -Ls:].to(torch.float32)
        return m.action_out_proj(out)

    def euler(self, kv, noise):
        # legacy: dt = tensor(-1/n, fp32); time = tensor(1.0); while time >= -dt/2: x_t += dt * v_t; time += dt
        dt = torch.tensor(-1.0 / self.n, dtype=torch.float32)
        ts, t = [], torch.tensor(1.0, dtype=torch.float32)
        while t >= -dt / 2: ts.append(float(t)); t = t + dt
        dtg = dt.to(noise.device)
        x = noise.clone()
        for tv in ts:
            t = torch.full((noise.shape[0],), tv, dtype=torch.float32, device=noise.device)
            x += dtg * self.denoise(kv, x, t)
        return x

    @torch.no_grad()
    def __call__(self, feat, state, noise):
        return self._euler(self.prefix(feat, state), noise).clone()
