"""Gradients of the answer with respect to the frozen model's residual stream at the 64 image rows, entering layer L = 0..15 and L = 16 (the input of the final norm). A zero leaf is added to the image rows of
the stream entering every layer in one fp32 prefix pass, so its gradient is the total derivative along every path. Objectives: (a) y = whole-flow final answer; (b) y_k = the readout of xhat_k at step k from
the stored intact x_t, step k only; (c) pooled over steps. Local term of (a) at layer L: the gradient with every path through the residual leaving layer L stopped; noJ = the same without the RMSNorm
Jacobian, gamma * (W_v^T g_V + W_k^T R^-1 g_K). Variants: grid (the fit scenes, shared noise) and rand (seed [62001, i], noise 8000 + i).
Output: data/stream_gradients/ (5.4 GB of arrays), read by experiments/stream_gradient_overlap.py. Runtime about 1 min; needs about 10 GB of GPU memory."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, numpy as np, torch
from toyvla import stream_tools as V, policy_tools
from toyvla.paths import DEVICE, DATA, SMOKE, empty_gpu_cache
R = V.R
OD = DATA / "stream_gradients"; OD.mkdir(parents=True, exist_ok=True)
S, pol, m, H, st = V.setup(); vlm = H.vlm; NS, NI, D = V.NS, V.NI, V.D
EPS = [float(vlm.layers[L].input_layernorm.variance_epsilon) for L in range(16)]

# ------------------------------------------------------------------ gradients
def kv_img(lay, x, pos):
    """layer's own K (post-RoPE) and V at the image rows from the stream x (B, 64, 960), as forward_attn_layer computes them."""
    hn = lay.input_layernorm(x); B = x.shape[0]
    k = lay.self_attn.k_proj(hn).view(B, NI, -1, lay.self_attn.head_dim); v = lay.self_attn.v_proj(hn).view(B, NI, -1, lay.self_attn.head_dim)
    return V.apply_rope(k, pos[:, :NI]), v


def run_variant(name, scs, noise_fn, BS=12, smoke=False):
    n = len(scs) if not smoke else min(BS, len(scs)); scs = scs[:n]
    mk = (lambda f, shp, dt: np.lib.format.open_memmap(f"{OD}/{f}_{name}.npy", mode="w+", dtype=dt, shape=shp)) if not smoke else (lambda f, shp, dt: np.zeros(shp, dt))
    AF = mk("A_full", (17, n, NI, D), np.float16); AL = mk("A_loc", (16, n, NI, D), np.float16); AN = mk("A_noJ", (16, n, NI, D), np.float16)
    BB = mk("B", (17, n, NS, NI, D), np.float16); GKs = mk("gK", (16, n, NI, 320), np.float32); GVs = mk("gV", (16, n, NI, 320), np.float32)
    scA = np.zeros((17, n), np.float32); scL = np.zeros((16, n), np.float32); scN = np.zeros((16, n), np.float32); scB = np.zeros((17, n, NS), np.float32)
    GA = torch.zeros(17, D, D, dtype=torch.float64, device=DEVICE); GL = torch.zeros(16, D, D, dtype=torch.float64, device=DEVICE)
    GN = torch.zeros(16, D, D, dtype=torch.float64, device=DEVICE); GB = torch.zeros(17, NS, D, D, dtype=torch.float64, device=DEVICE)
    yA, yB, yB0 = np.zeros(n), np.zeros((NS, n)), np.zeros((NS, n)); chk = dict(kv_recompute=[], local_formula_cos=[], cos_loc_full=[])
    def put(arr, sc, idx, g):
        mx = g.abs().flatten(1).amax(1).clamp_min(1e-30); sc[idx] = mx.cpu().numpy()
        arr[idx] = (g / mx.view(-1, *([1] * (g.dim() - 1)))).cpu().numpy().astype(np.float16)
    for j in range(0, n, BS):
        sub = scs[j:j + BS]; B = len(sub); noise = noise_fn(j, B); sl = slice(j, j + B)
        z = [torch.zeros(B, NI, D, device=DEVICE, requires_grad=True) for _ in range(17)]; xin = {}
        def fn(L, h):
            xin[L] = h[:, :NI].detach().clone(); return torch.cat([h[:, :NI] + z[L], h[:, NI:]], 1)
        st.fn = fn
        try: pm, kv, pos = V.prefix_fp32(H, pol, sub, grad=True)
        finally: st.fn = None
        assert len(xin) == 17, sorted(xin)
        kvK = [kv[L]["key_states"] for L in range(16)]; kvV = [kv[L]["value_states"] for L in range(16)]
        # (a) whole flow with grad 
        x = noise.clone(); t = torch.tensor(1.0, device=DEVICE); dt = torch.tensor(-1.0 / NS, device=DEVICE); xs, ts = [], []
        with torch.enable_grad():
            for k in range(NS):
                xs.append(x.detach().clone()); ts.append(t.clone())
                v = m.denoise_step(pm, kv, x, t.expand(B))
                with torch.no_grad(): yB0[k, sl] = R.y_of((x - t * v)[:, :, 0]).cpu().numpy()
                x = x + dt * v; t = t + dt
            y = V.y_pullback(x)
            g = torch.autograd.grad(y.sum(), z + kvK + kvV, retain_graph=True, allow_unused=True)
        yA[sl] = y.detach().cpu().numpy(); del x, y, v
        gz = [gg if gg is not None else torch.zeros(B, NI, D, device=DEVICE) for gg in g[:17]]
        gK = [g[17 + L][:, :NI].reshape(B, NI, -1) for L in range(16)]; gV = [g[33 + L][:, :NI].reshape(B, NI, -1) for L in range(16)]
        for L in range(17):
            g2 = gz[L].reshape(-1, D).double(); GA[L] += g2.T @ g2; put(AF[L], scA[L], sl, gz[L])
        for L in range(16):
            GKs[L, sl] = gK[L].cpu().numpy(); GVs[L, sl] = gV[L].cpu().numpy()
            lay = vlm.layers[L]; xd = xin[L].float().requires_grad_(True)
            with torch.enable_grad():
                K_, V_ = kv_img(lay, xd, pos)
                loc, = torch.autograd.grad([K_, V_], [xd], [g[17 + L][:, :NI], g[33 + L][:, :NI]])
                r = xd.detach().pow(2).mean(-1, keepdim=True).add(EPS[L]).sqrt(); u = (xd.detach() / r).requires_grad_(True)
                hn = lay.input_layernorm.weight * u; k2 = lay.self_attn.k_proj(hn).view(B, NI, -1, 64); v2 = lay.self_attn.v_proj(hn).view(B, NI, -1, 64)
                noJ, = torch.autograd.grad([V.apply_rope(k2, pos[:, :NI]), v2], [u], [g[17 + L][:, :NI], g[33 + L][:, :NI]])
            chk["kv_recompute"].append(max(float((K_ - kv[L]["key_states"][:, :NI]).abs().max()), float((V_ - kv[L]["value_states"][:, :NI]).abs().max())))
            xx = xd.detach(); form = (noJ - xx * (xx * noJ).sum(-1, keepdim=True) / (D * r ** 2)) / r
            nz = loc.norm(dim=-1) > 0
            chk["local_formula_cos"].append(float(torch.nn.functional.cosine_similarity(form, loc, dim=-1)[nz].min()))
            chk["cos_loc_full"].append(float((loc * gz[L]).sum() / (loc.norm() * gz[L].norm()).clamp_min(1e-30)))
            for G_, arr, sc, gg in ((GL, AL, scL, loc), (GN, AN, scN, noJ)):
                g2 = gg.reshape(-1, D).double(); G_[L] += g2.T @ g2; put(arr[L], sc[L], sl, gg)
        del g, gz, gK, gV
        # (b) per step, from the stored intact x_t, step k only
        for k in range(NS):
            with torch.enable_grad():
                v = m.denoise_step(pm, kv, xs[k], ts[k].expand(B)); yk = R.y_of((xs[k] - ts[k] * v)[:, :, 0])
                gb = torch.autograd.grad(yk.sum(), z, retain_graph=(k < NS - 1), allow_unused=True)
            yB[k, sl] = yk.detach().cpu().numpy()
            for L in range(17):
                gg = gb[L] if gb[L] is not None else torch.zeros(B, NI, D, device=DEVICE)
                g2 = gg.reshape(-1, D).double(); GB[L, k] += g2.T @ g2
                mx = gg.abs().flatten(1).amax(1).clamp_min(1e-30); scB[L, sl, k] = mx.cpu().numpy()
                BB[L, sl, k] = (gg / mx[:, None, None]).cpu().numpy().astype(np.float16)
            del gb, v, yk
        d_ = np.abs(yB[:, sl] - yB0[:, sl]).max(); assert d_ < 1e-3, ("step re-run differs from flow", d_)
        del kv, kvK, kvV, xs, z, xin, pm; empty_gpu_cache()
        V.log(f"{name}: {j + B}/{n}  ({V.peak()})")
    if not smoke:
        for a in (AF, AL, AN, BB, GKs, GVs): a.flush()
        np.savez(f"{OD}/scales_{name}.npz", A_full=scA, A_loc=scL, A_noJ=scN, B=scB)
    # eigen-decompositions (descending)
    def eig(G):
        w, Q = torch.linalg.eigh(G); return w.flip(-1).cpu().numpy(), Q.flip(-1).transpose(-1, -2).float().cpu().numpy()
    out = {}
    for nm, G_ in (("A", GA), ("L", GL), ("N", GN), ("P", GB.sum(1))):
        w, Q = eig(G_); out[f"ev{nm}"] = w; out[f"vec{nm}"] = Q
    w, Q = eig(GB); out["evB"] = w; out["vecB"] = Q
    eA = np.array([GA[L].trace().item() for L in range(17)]); eB = np.array([[GB[L, k].trace().item() for k in range(NS)] for L in range(17)])
    eL = np.array([GL[L].trace().item() for L in range(16)])
    cnt = lambda w: (V.kfrac(w, 0.90), V.kfrac(w, 0.99)) if w.clip(0).sum() > 0 else (0, 0)
    meta = dict(name=name, n=n, phi=[s["phi"] for s in scs], y_a=yA.tolist(), y_b=yB.tolist(), energy_a=eA.tolist(), energy_b=eB.tolist(), energy_loc=eL.tolist(),
                k_a=[cnt(out["evA"][L]) for L in range(17)], k_loc=[cnt(out["evL"][L]) for L in range(16)], k_noJ=[cnt(out["evN"][L]) for L in range(16)],
                k_pool=[cnt(out["evP"][L]) for L in range(17)], k_b=[[cnt(out["evB"][L, k]) for k in range(NS)] for L in range(17)],
                checks=dict(kv_recompute_maxabs=float(max(chk["kv_recompute"])), local_formula_min_cos=float(min(chk["local_formula_cos"])),
                            cos_loc_full_batches=chk["cos_loc_full"]))
    V.log(name, "energy share (a) per layer:", np.round(eA / eA.sum(), 4).tolist())
    V.log(name, "k90/k99 (a):", meta["k_a"]); V.log(name, "k90/k99 pooled:", meta["k_pool"])
    V.log(name, "k90 (b) per layer (steps 0..9):", ["".join(str(c[0]) if c[0] < 10 else f"({c[0]})" for c in row) for row in meta["k_b"]])
    V.log(name, "checks:", meta["checks"]["kv_recompute_maxabs"], meta["checks"]["local_formula_min_cos"])
    if not smoke:
        np.savez(f"{OD}/eig_{name}.npz", **out); json.dump(meta, open(f"{OD}/meta_{name}.json", "w"))
    del GA, GL, GN, GB; empty_gpu_cache()
    return yA, yB


grid = V.grid_scenes()[::24] if SMOKE else V.grid_scenes(); rnd = V.rand_scenes(62001, 3 if SMOKE else 60)
shared = lambda j, B: policy_tools.noise_for(B, 0)
per = lambda j, B: R.noise_per_scene(range(8000 + j, 8000 + j + B))
run_variant("grid", grid, shared); run_variant("rand", rnd, per)
V.log(V.peak()); V.log("done")
