"""Helpers for the policy's prefix pass (the frozen vision-language model's residual stream and its key/value cache).
The stream entering layer L is inputs_embeds[0] of forward_attn_layer(layer_idx = L), L = 16 = input of the final norm; edits replace it there.
Scenes: the gradient fit grid is phi = -180..175 step 5 (72 scenes), theta_s = 30, shared noise_for(B, 0), batch 12."""
import json, time, resource, numpy as np, torch
from toyvla import policy_tools, readouts as R
from toyvla.paths import DEVICE, POLICY, GRADIENT_DIRECTIONS, VLM_WEIGHTS, peak_gpu_bytes
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from lerobot.policies.smolvla.smolvlm_with_expert import apply_rope
NS = 10; NL = 16; NI = 64; D = 960; ODD = list(range(1, 16, 2))
t0 = time.time()


def log(*a): print(f"[{time.time() - t0:8.1f}s]", *a, flush=True)
def peak(): return f"GPU peak {peak_gpu_bytes() / 2**30:.2f} GiB, host peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.2f} GiB"


class St: fn = None


def setup(colour="red"):
    S = policy_tools.S
    pol, m = policy_tools.load(POLICY[colour]); H = policy_tools.Hooks(m); ve = H.ve
    for p in m.parameters(): p.requires_grad_(False)
    st = St()
    def pwrap(old):
        def call(*a, **kw):
            if H.phase == "prefix" and st.fn is not None:
                emb = a[1]; new = st.fn(a[2], emb[0])
                if new is not None: emb[0] = new
            return old(*a, **kw)
        return call
    ve.forward_attn_layer = pwrap(ve.forward_attn_layer); ve.forward_cross_attn_layer = pwrap(ve.forward_cross_attn_layer)
    def npre(mod, a):
        if H.phase == "prefix" and st.fn is not None:
            new = st.fn(16, a[0])
            if new is not None: return (new,)
    H.vlm.norm.register_forward_pre_hook(npre)
    return S, pol, m, H, st


# ------------------------------------------------------------------ scenes
def grid_scenes():
    S = policy_tools.S
    phis = np.arange(-180, 180, 5.0)
    return [S.scene_at(float(p), 30.0) for p in phis]


def rand_scenes(seed, n):
    S = policy_tools.S; return [S.make_scene(np.random.default_rng([seed, i]), "full") for i in range(n)]


def wrapd(a): return (np.asarray(a, float) + 180) % 360 - 180
def adiff(a, b): return wrapd(np.asarray(a, float) - np.asarray(b, float))


# ------------------------------------------------------------------ readouts
def y_pullback(x):
    """the differentiated answer: mean over the 50 rows of coordinate 0, in degrees."""
    return x[:, :, 0].mean(1) * 180


def y_eval_np(x):
    """the evaluated answer: circular mean of x*180 over the 50 rows (numpy, float64)."""
    r = x[:, :, 0].float().cpu().numpy().astype(np.float64)
    a = np.radians(r * 180); return np.degrees(np.arctan2(np.sin(a).mean(1), np.cos(a).mean(1)))


# ------------------------------------------------------------------ prefix passes
def prefix_fp32(H, pol, scs, grad=True):
    """the policy's prefix pass in fp32 (no autocast), with autograd through the VLM layers when grad (embed_prefix stays no-grad)."""
    m = H.m; inp = policy_tools.make_inputs(pol, scs)
    with torch.no_grad():
        embs, pm, am = m.embed_prefix(inp["images"], inp["img_masks"], inp["ids"], inp["mk"], state=inp["state"])
    a2 = make_att_2d_masks(pm, am); pos = torch.cumsum(pm, dim=1) - 1
    H.phase = "prefix"
    try:
        with torch.set_grad_enabled(grad):
            _, kv = H.ve.forward(attention_mask=a2, position_ids=pos, past_key_values=None, inputs_embeds=[embs, None], use_cache=True, fill_kv_cache=True)
    finally: H.phase = None
    return pm, kv, pos


class Embs:
    """bf16-autocast prefix (as Hooks.prefix) with the SigLIP embeddings computed once per batch and reused for every edit."""
    def __init__(self, H, pol, scs, imgs=None):
        self.H = H; m = H.m; inp = policy_tools.make_inputs(pol, scs, imgs=imgs)
        with torch.no_grad(), torch.autocast(DEVICE.split(":")[0], dtype=torch.bfloat16):
            self.embs, self.pm, am = m.embed_prefix(inp["images"], inp["img_masks"], inp["ids"], inp["mk"], state=inp["state"])
        self.a2 = make_att_2d_masks(self.pm, am); self.pos = torch.cumsum(self.pm, dim=1) - 1; self.B = self.pm.shape[0]
    def kv(self, st=None, fn=None):
        H = self.H
        if st is not None: st.fn = fn
        H.phase = "prefix"
        try:
            with torch.no_grad(), torch.autocast(DEVICE.split(":")[0], dtype=torch.bfloat16):
                _, kv = H.ve.forward(attention_mask=self.a2, position_ids=self.pos, past_key_values=None, inputs_embeds=[self.embs.clone(), None],
                                     use_cache=True, fill_kv_cache=True)
        finally:
            H.phase = None
            if st is not None: st.fn = None
        return kv


def flow_eval(m, pm, kv, noise, keep_xt=False):
    """fp32 x, no autocast, cache as given: final answer (numpy, eval readout), y_k per step, x_t."""
    x = noise.clone(); B = x.shape[0]; t = torch.tensor(1.0, device=DEVICE); dt = torch.tensor(-1.0 / NS, device=DEVICE); xs, ts, ys = [], [], []
    with torch.no_grad():
        for k in range(NS):
            if keep_xt: xs.append(x.clone()); ts.append(t.clone())
            v = m.denoise_step(pm, kv, x, t.expand(B)); ys.append(R.y_of((x - t * v)[:, :, 0]).float().cpu().numpy())
            x = x + dt * v; t = t + dt
    return y_eval_np(x), np.stack(ys), xs, ts


def one_step(m, pm, kv, xt, tk):
    B = xt.shape[0]
    with torch.no_grad():
        v = m.denoise_step(pm, kv, xt, tk.expand(B))
    return R.y_of((xt - tk * v)[:, :, 0]).float().cpu().numpy()


# ------------------------------------------------------------------ pulled-back cache directions, all layers
def rope_back(D):
    """(n, 320) post-RoPE key directions -> (64, n, 320) pre-RoPE directions per image position."""
    d = torch.tensor(np.asarray(D), dtype=torch.float32).reshape(len(D), 1, 5, 64).expand(len(D), 64, 5, 64); pos = torch.arange(64)[None].float()
    return apply_rope(d, -pos.expand(len(D), 64)).reshape(len(D), 64, 320).permute(1, 0, 2).numpy().astype(np.float64)


def kept(G, L, kind, frac=0.90):
    s = np.array(G["s" + kind][str(L)]); tot = G["grad_share"][kind][L] * G["grad_energy_total"]
    return int(np.searchsorted(np.cumsum(s ** 2) / tot, frac) + 1)


def pulled_back(layers=range(16), frac=0.90):
    """per layer: S (64, kV + kK, 960) = [gam * (DV W_v) broadcast over tokens ; gam * (rope_back(DK) W_k)], plus kV, kK."""
    G = json.load(open(GRADIENT_DIRECTIONS)); z = np.load(VLM_WEIGHTS); out = {}
    for L in layers:
        kv, kk = kept(G, L, "V", frac), kept(G, L, "K", frac); DV = np.array(G["DV"][str(L)][:kv]); DK = np.array(G["DK"][str(L)][:kk]); gam = z[f"gam{L}"]
        sV = (DV @ z[f"Wv{L}"]) * gam; sK = (rope_back(DK) @ z[f"Wk{L}"]) * gam
        out[L] = dict(S=np.concatenate([np.broadcast_to(sV, (64, kv, 960)), sK], 1), kV=kv, kK=kk, DV=DV, DK=DK)
    return out


def orth_rows(A):
    A = np.asarray(A, np.float64)
    if len(A) == 0: return np.zeros((0, A.shape[-1]))
    q, r = np.linalg.qr(A.T); keep = np.abs(np.diag(r)) > 1e-10 * np.abs(np.diag(r)).max(); return q[:, keep].T


def mean_cos2(A, B):
    """mean cos^2 of the min(kA, kB) principal angles between the row spaces of A and B (orthonormalised here)."""
    qa, qb = orth_rows(A), orth_rows(B); s = np.linalg.svd(qa @ qb.T, compute_uv=False); return float(np.mean(s ** 2))


def kfrac(evals, frac):
    ev = np.clip(np.asarray(evals, np.float64), 0, None); return int(np.searchsorted(np.cumsum(ev) / ev.sum(), frac) + 1)
