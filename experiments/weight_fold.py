"""Fold a fixed estimate into the query weights of layers 5-15 (the equation in the paper's weight-fold section). With n = input_layernorm(x) the row entering q_proj, B (2 x 720) the directions of n-space that
move with the estimate coordinate c (least squares n_rowmean ~ nbar + c B over the 96 fit scenes at the last flow step), P = B^+ B and g = nbar / |nbar|^2 (g . n ~ 1 for every row):
    M = (I - P) + g^T (nbar P + c_A B),   and in torch's convention W' = W M^T.
A rank-2 removal plus a rank-1 addition that acts at every flow step and action row. Conditions: A on a grid, estimate removed (no angle written), random-plane control (the same construction in
a random 2-D subspace), identity. 288 scenes (seed [61030, i], noise 14000 + i), disjoint from the fit scenes. No hooks at evaluation.
Output: results/weight_fold.npz. Runtime about 1 min on a GPU; needs data/estimate_fit.npz from experiments/query_replacement.py."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time, numpy as np, torch
from toyvla import query_tools as Q
from toyvla.paths import DEVICE, RESULTS, ESTIMATE_FIT, SMOKE
V, R = Q.V, Q.V.R
BS = 16
S, pol, m, H, st = V.setup()
NS, ODD = Q.NS, Q.ODD; dev = DEVICE; EL = [5, 7, 9, 11, 13, 15]; ex = H.ex
fz = np.load(ESTIMATE_FIT); BL = {L: fz[f"B_L{L}"] for L in EL}; MU = {L: fz[f"MU_L{L}"] for L in EL}; fphi = fz["fit_phi"]
W0 = {L: ex.layers[L].self_attn.q_proj.weight.detach().clone() for L in EL}
def restore():
    for L in EL: ex.layers[L].self_attn.q_proj.weight.data.copy_(W0[L])
def tgt_of(scs): return np.array([s_["phi"] for s_ in scs])
def wrap(a): return (a + 180.0) % 360.0 - 180.0
# ---- calibration with hooks (removed afterwards): x (layer input) and n (q_proj input) at step 9, row means, 96 fit scenes
CAL = {L: {"x": [], "n": []} for L in EL}; ON = {"v": False}
hs = []
for L in EL:
    lay = ex.layers[L]
    hs.append(lay.input_layernorm.register_forward_pre_hook(lambda mod, a, L=L: CAL[L]["x"].append(a[0][:, -50:].float().mean(1).cpu()) if (ON["v"] and H.step == 9 and a[0].shape[1] == 50) else None))
    hs.append(lay.self_attn.q_proj.register_forward_pre_hook(lambda mod, a, L=L: CAL[L]["n"].append(a[0][:, -50:].float().mean(1).cpu()) if (ON["v"] and H.step == 9 and a[0].shape[1] == 50) else None))
NF = 6 if SMOKE else 96; fit = [S.make_scene(np.random.default_rng([61004, i]), "full") for i in range(NF)]; fseeds = list(range(11000, 11000 + NF))
N = 2 if SMOKE else 288; scs = [S.make_scene(np.random.default_rng([61030, i]), "full") for i in range(N)]; seeds = list(range(14000, 14000 + N))
phi = np.array([s_["phi"] for s_ in scs]); tgt = tgt_of(scs)
def run(scs_, seeds_):
    n = len(scs_); Y = np.zeros(n, np.float32)
    for j in range(0, n, BS):
        sub = scs_[j:j + BS]; B = len(sub); noise = R.noise_per_scene(seeds_[j:j + B]); E_ = V.Embs(H, pol, sub); kv = E_.kv()
        Y[j:j + B], _ = Q.flow_eval(m, H, E_.pm, kv, noise); del kv
    return Y
t0 = time.time(); ON["v"] = True; run(fit, fseeds); ON["v"] = False
for h_ in hs: h_.remove()
X = {L: torch.cat(CAL[L]["x"]).numpy() for L in EL}; Nn = {L: torch.cat(CAL[L]["n"]).numpy() for L in EL}
assert X[5].shape == (NF, 720) and np.allclose(np.array([s_["phi"] for s_ in fit]), fphi)
# estimate coordinate (x-space) and its n-space image
C, BN, NBAR, G, R2 = {}, {}, {}, {}, {}
for L in EL:
    c = (X[L] - MU[L].mean(0)) @ BL[L].T                       # (96, 2) row-mean coordinate
    A_ = np.concatenate([np.ones((NF, 1)), c], 1); coef = np.linalg.lstsq(A_, Nn[L], rcond=None)[0]   # (3, 720): nbar_fit, Bn
    NBAR[L] = Nn[L].mean(0); BN[L] = coef[1:]; C[L] = c; G[L] = NBAR[L] / (NBAR[L] @ NBAR[L])
    pred = A_ @ coef; R2[L] = 1 - ((Nn[L] - pred) ** 2).sum() / ((Nn[L] - Nn[L].mean(0)) ** 2).sum()
    gn = Nn[L] @ G[L]; Q.log(f"L{L}: plane -> n-space R2 {R2[L]:.3f} (fraction of n's scene variance explained by the 2 coords); |Bn| {np.linalg.norm(BN[L], axis=1).round(2)}; g.n in [{gn.min():.2f}, {gn.max():.2f}]")
def template(L, A):
    dd = np.abs(wrap(fphi - A)); sel = dd <= 15
    if sel.sum() < 3: sel = np.argsort(dd)[:3]
    return C[L][sel].mean(0)
rng = np.random.default_rng(5); RB = {L: np.linalg.qr(rng.standard_normal((720, 2)))[0].T for L in EL}
def apply(kind, A=None):
    restore()
    for L in EL:
        Bn = BN[L] if kind != "rand" else RB[L]; Pm = np.linalg.pinv(Bn) @ Bn               # (720, 720) projector onto span(Bn)
        if kind == "identity": M = np.eye(720)
        elif kind == "erase": M = np.eye(720) - Pm + np.outer(G[L], NBAR[L] @ Pm)
        else:
            if kind == "tmpl": cA = template(L, A); v = NBAR[L] @ Pm + cA @ Bn
            else:   # rand: the random subspace's coordinate of the fit scenes near A, as a template
                cr = (Nn[L] - NBAR[L]) @ np.linalg.pinv(Bn); dd = np.abs(wrap(fphi - A)); sel = dd <= 15
                if sel.sum() < 3: sel = np.argsort(dd)[:3]
                v = NBAR[L] @ Pm + cr[sel].mean(0) @ Bn
            M = np.eye(720) - Pm + np.outer(G[L], v)
        Mt = torch.tensor(M, dtype=W0[L].dtype, device=dev); ex.layers[L].self_attn.q_proj.weight.data.copy_(W0[L] @ Mt.T)   # q = W n' with n' = n M  =>  W' = W M^T
GRID = [120] if SMOKE else [-150, -120, -90, -60, 60, 90, 120, 150]
conds = [("identity", None), ("erase", None)] + [("tmpl", A) for A in GRID] + [("rand", A) for A in (-120, 120)]
pos, neg = phi > 0, phi < 0; bands = [(-180, -120), (-120, -90), (-90, 0), (0, 90), (90, 120), (120, 180)]
lines = [f"estimate folded into q_proj of layers 5-15 (N {N}); median error phi>0 (n {pos.sum()}) / phi<0 (n {neg.sum()}) [frac < 10 deg]; then by band " + " ".join(f"[{a},{b})" for a, b in bands)]
restore(); Y0 = run(scs, seeds); e0 = np.abs(V.adiff(Y0, tgt)); lines.append(f"intact (weights untouched): {np.median(e0[pos]):5.1f} / {np.median(e0[neg]):5.1f}")
RES = {"intact": Y0}
for kind, A in conds:
    apply(kind, A); Y = run(scs, seeds); e = np.abs(V.adiff(Y, tgt)); name = kind if A is None else f"{kind}_{A:+d}"; RES[name] = Y
    bl = " ".join(f"{np.median(e[(phi >= a) & (phi < b)]):5.1f}" if ((phi >= a) & (phi < b)).sum() >= 5 else "    -" for a, b in bands)
    lines.append(f"{name:10s}: {np.median(e[pos]):5.1f} [{(e[pos] < 10).mean():.2f}] / {np.median(e[neg]):5.1f} [{(e[neg] < 10).mean():.2f}]   bands {bl}   [{time.time() - t0:.0f} s]")
restore()
Q.log("\n".join(lines))
np.savez(RESULTS / "weight_fold.npz", phi=phi, tgt=tgt, **{f"y_{k}": v for k, v in RES.items()}, **{f"Bn_L{L}": BN[L] for L in EL}, **{f"nbar_L{L}": NBAR[L] for L in EL}, **{f"C_L{L}": C[L] for L in EL}, fit_phi=fphi, r2=np.array([R2[L] for L in EL]))
Q.log("done")
