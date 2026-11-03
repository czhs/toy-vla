"""Keep-only and erase of the read directions at all eight odd layers, keys and values. Edits relative to the per-slot mean MU over a separate fit
panel of 256 scenes: keep X' = MU + (X - MU) P; erase X' = X - (X - MU) P; mean X' = MU. P = projector on the directions; controls of the same dimension:
random (QR of Gaussian) and the top principal components of (X - MU) on the fit panel. 256 held-out scenes, noise 7000 + scene.
Output: results/keep_erase.npz (phi and the per-scene answer of every condition). Runtime about 1 min on a GPU."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time, numpy as np, torch
from toyvla import cache_edits as C, policy_tools
from toyvla.paths import DEVICE, RESULTS, SMOKE
N = 4 if SMOKE else 256; BS = 32; t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
S, pol, m, H = C.setup(); D, KD, _ = C.read_directions(); log("direction counts", {f"L{L}{k}": v for (L, k), v in KD.items()})
fit_sc = C.scenes(61001, N); ev_sc = C.scenes(61002, N); phi = np.array([s["phi"] for s in ev_sc])
# ---- slot means and PCA on the fit panel
acc = {(L, k): [] for L in C.ODD for k in "VK"}
for j in range(0, N, BS):
    pm, kv = C.prefix(H, pol, fit_sc[j:j + BS])
    for L in C.ODD:
        for k in "VK": acc[(L, k)].append(C.rows(kv, L, k).cpu())
    del kv
MU, PCA = {}, {}
for key, v in acc.items():
    X = torch.cat(v).double(); MU[key] = X.mean(0).float().to(DEVICE)
    Xc = (X - X.mean(0)).reshape(-1, X.shape[-1]); _, s, Vt = torch.linalg.svd(Xc, full_matrices=False); PCA[key] = Vt[:16].numpy()
    PCA[key + ("var",)] = (s ** 2).numpy()
del acc
ev = []
for j in range(0, N, BS):
    pm, kv = C.prefix(H, pol, ev_sc[j:j + BS]); ev.append((pm, kv))
rng = np.random.default_rng(0)
def projm(B): Q = torch.tensor(np.asarray(B), dtype=torch.float32, device=DEVICE); return Q.T @ Q
RAND = {(L, k): np.linalg.qr(rng.standard_normal((320, KD[(L, k)])))[0].T for L in C.ODD for k in "VK"}
def basis(src, L, k):
    if src == "gradient": return D[(L, k)]
    if src == "rand": return RAND[(L, k)]
    if src == "pca": return PCA[(L, k)][:KD[(L, k)]]
def run(edits):
    """edits: list of (L, kind, mode, P or None)"""
    out = []
    for b, (pm, kv0) in enumerate(ev):
        kv = C.clone_kv(kv0) if edits else kv0
        for L, k, mode, P in edits:
            X = C.rows(kv, L, k); mu = MU[(L, k)][None]
            if mode == "keep": Xn = mu + (X - mu) @ P
            elif mode == "erase": Xn = X - (X - mu) @ P
            elif mode == "mean": Xn = mu.expand_as(X)
            C.set_rows(kv, L, k, Xn)
        out.append(C.readout(m, pm, kv, policy_tools.noise_for(pm.shape[0], 7000 + b * BS, shared=False)))
    return np.concatenate(out)
th0 = run([]); e0 = np.abs(C.ang_diff(th0, phi)); log(f"intact median err {np.median(e0):.3f} mean {e0.mean():.3f}")
Y = dict(intact=th0)
for mode in ("keep", "erase"):
    for src in ("gradient", "rand", "pca"):
        th = run([(L, k, mode, projm(basis(src, L, k))) for L in C.ODD for k in "VK"]); Y[f"{mode}_{src}"] = th; log(mode, src, round(float(np.median(np.abs(C.ang_diff(th, phi)))), 2))
Y["mean"] = run([(L, k, "mean", None) for L in C.ODD for k in "VK"])
np.savez(RESULTS / "keep_erase.npz", phi=phi, **Y); log("written")
