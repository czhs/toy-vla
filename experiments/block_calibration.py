"""Calibration T of the read directions: 400 scenes (seed [271027, i], split 'full'; red block + link). T_L = mean over scenes with block-centre token red
fraction > 0.08 of (projection at the block-centre token - slot mean over the 400 scenes). Output: results/block_calibration.json. Runtime about 1 min on a GPU."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, time, numpy as np
from toyvla import vlm_pointer as C, policy_tools
from toyvla.paths import DEVICE, POLICY, BLOCK_CALIBRATION, SMOKE
t0 = time.time(); EPS = 1e-5
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
S = policy_tools.S; BGN = "plain"
pol, m = policy_tools.load(POLICY["red"]); H = policy_tools.Hooks(m)
dirs = C.directions(); Sd = C.to_dev(dirs)
def base_res(scs, imgs):
    H.capture_prefix(layers=C.ODD, slots=slice(0, 64)); pm, kv = H.prefix(policy_tools.make_inputs(pol, scs, imgs=imgs))
    X = {L: H.cap[f"pre_resid_l{L}"].to(DEVICE).float() for L in C.ODD}; H.clear(); return X, kv
def table_mask(a, b): return (np.abs(a.astype(int) - b.astype(int)).sum(-1) > 30).astype(float)
NCAL = 8 if SMOKE else 400; scs = [S.make_scene(np.random.default_rng([271027, i]), "full") for i in range(NCAL)]; Cc = {L: [] for L in C.ODD}; fr = []; chk = {}
for j in range(0, NCAL, 32):
    sub = scs[j:j + 32]; imgs = [S.render(sc, BGN) for sc in sub]
    for sc in sub:
        a = S.render(sc, BGN, arm=False); b = S.render(sc, BGN, arm=False, blocks=False); fr.append(table_mask(a, b).reshape(8, 64, 8, 64).mean((1, 3)).reshape(64))
    X, kv = base_res(sub, imgs)
    for L in C.ODD:
        c = C.proj(Sd, L, X[L], EPS).cpu().numpy(); Cc[L].append(c)
        if j == 0:
            Vv = kv[L]["value_states"][:, :64].float().reshape(len(sub), 64, -1).cpu().numpy() @ dirs[L]["DV"].T
            Kk = kv[L]["key_states"][:, :64].float().reshape(len(sub), 64, -1).cpu().numpy() @ dirs[L]["DK"].T
            ref = np.concatenate([Vv, Kk], -1); chk[str(L)] = dict(max_abs_diff=float(np.abs(ref - c).max()), max_abs=float(np.abs(ref).max()), corr=float(np.corrcoef(ref.ravel(), c.ravel())[0, 1]))
    del kv
fr = np.array(fr); ub, vb = S.w2uv(np.array([s["px"] for s in scs]), np.array([s["py"] for s in scs]))
jb = (np.floor(vb / 64) * 8 + np.floor(ub / 64)).astype(int); b = np.arange(NCAL); sel = fr[b, jb] > 0.08; T = {}
for L in C.ODD:
    c = np.concatenate(Cc[L]); mu = c.mean(0); T[L] = (c[b[sel], jb[sel]] - mu[jb[sel]]).mean(0)
json.dump(dict(T={str(L): T[L].tolist() for L in C.ODD}, n_selected=int(sel.sum()), cache_check=chk), open(BLOCK_CALIBRATION, "w"))
log("calibrated", int(sel.sum()), chk["1"])
