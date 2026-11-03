import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""The pointer test at layer 7 on fresh scenes (seed [20261300, i], not used to choose the layer or strength; 1000-scene chunks bound host memory). Conditions: no
edit, drawn box, all odd layers at strength 2, values only at strength 10 / 12 / 14 / 16, keys and values at 8, keys only at 8, each edit with its random control.
Output: results/pointer_test.npz. Runtime about 20 min on a 24 GB GPU (4.3 GB used)."""
import json, time, numpy as np
from toyvla import pointer_plants as GM
from toyvla.paths import RESULTS, SMOKE
C = GM.C; ODD = GM.ODD; N = 2 if SMOKE else 4000; SEED = 20261300; CH = 1000; t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
E = GM.Env(bs=int(os.environ.get("BS", "40")))
def single(L, m, toks, rows="all"): return GM.calib(toks, [(L, L)], m, rows)
def conds_for(scs):
    t1 = [[s["tok"]] for s in scs]; own = [(L, L) for L in ODD]
    c = [("none", dict(img="plain")), ("box", dict(img="box"))] + GM.with_rand("all_odd_m2", GM.calib(t1, own, 2))
    for m in (10, 12, 14, 16): c += GM.with_rand(f"L7V_m{m}", single(7, m, t1, "V"))
    c += GM.with_rand("L7_m8", single(7, 8, t1)) + GM.with_rand("L7K_m8", single(7, 8, t1, "K"))
    return c
scs_all = GM.ring8_scenes(N, seed=SEED); got = {}
for j in range(0, N, CH):
    scs = scs_all[j:j + CH]; ims = GM.images(scs); res = GM.run(E, ims, conds_for(scs), scs, None, None, log)
    for k, r in res.items():
        for p in E.P: got.setdefault(f"{k}|{p}", []).extend(str(g) for g in r[p]["got"])
    log(f"chunk {j} done")
digit = np.array([s["digit"] for s in scs_all])
np.savez(RESULTS / "pointer_test.npz", digit=digit, **{k: np.array(v) for k, v in got.items()})
log("accuracy, paper question", {k.split("|")[0]: round(float((np.array(v) == digit).mean()), 4) for k, v in got.items() if k.endswith("|paper")})
