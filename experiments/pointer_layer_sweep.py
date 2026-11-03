import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Layer x strength sweep of the pointer test: one odd layer, keys and values, strength 1 to 8 at every odd layer and 12, 16 at layers 5, 7, 9, with the random
control of the same norm, the no-edit and drawn-box references and the all-odd-layers plant at strength 2. 200 scenes (seed [20261200, i], ring offset 0), both
questions; the paper's layer and strength table reads the selection question. Output: results/pointer_layer_sweep.npz. Runtime about 40 min on an 8 GB GPU (4.4 GB used)."""
import json, time, numpy as np
from toyvla import pointer_plants as GM
from toyvla.paths import RESULTS, SMOKE
C = GM.C; ODD = GM.ODD; N = 2 if SMOKE else 200; t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
E = GM.Env(bs=int(os.environ.get("BS", "40")))
def single(L, m, toks): return GM.calib(toks, [(L, L)], m)
scs = GM.ring8_scenes(N); ims = GM.images(scs); t1 = [[s["tok"]] for s in scs]
conds = GM.ref_conds(scs)
for L in (7,) if SMOKE else ODD:
    for m in (8,) if SMOKE else (1, 2, 4, 8): conds += GM.with_rand(f"L{L}_m{m}", single(L, m, t1))
for L in () if SMOKE else (5, 7, 9):
    for m in (12, 16): conds += GM.with_rand(f"L{L}_m{m}", single(L, m, t1))
res = GM.run(E, ims, conds, scs, None, None, log)
np.savez(RESULTS / "pointer_layer_sweep.npz", digit=np.array([s["digit"] for s in scs]), **{f"{k}|{p}": np.array([str(g) for g in r[p]["got"]]) for k, r in res.items() for p in E.P})
log("summary", {k: {p: GM.acc(r, scs, p) for p in E.P} for k, r in res.items()})
