"""Policies trained on a green or a blue block, no edit, on the first 128 scenes of the colour-switch panel (seed [61060, i], noise 7000 + i). usage: colour_native_reference.py green|blue
Output: results/colour_native_<colour>.npz. Runtime under a minute on a GPU."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time, numpy as np
from toyvla import query_tools as Q, colour_scenes as CG
from toyvla.paths import RESULTS, SMOKE
V, R = Q.V, Q.V.R
NAT = sys.argv[1]; N = 2 if SMOKE else 128; BS = 16
S, pol, m, H, st = V.setup(NAT)
AR = (-180.0, 180.0)
PAL = CG.palette(S); COLS = list(PAL)
scenes = [CG.compose(S, np.random.default_rng([61060, i]), PAL, k=2, lo=AR[0], hi=AR[1]) for i in range(N)]
angle = np.full((N, 3), np.nan)
for i, (sc_, sq_, objs) in enumerate(scenes):
    angle[i, 0] = sc_["phi"]
    for (c, a) in sq_: angle[i, COLS.index(c)] = a
Y0 = np.zeros(N); t0 = time.time()
for j in range(0, N, BS):
    sub = scenes[j:j + BS]; B = len(sub)
    E = V.Embs(H, pol, [s_ for s_, _, _ in sub], imgs=[CG.render_objs(S, s_, "plain", objs=o) for s_, _, o in sub]); kv = E.kv()
    noise = R.noise_per_scene(range(7000 + j, 7000 + j + B)); Y0[j:j + B], _ = Q.flow_eval(m, H, E.pm, kv, noise)
    Q.log(f"{NAT} panel {j + B}/{N} {time.time() - t0:.0f}s", Q.peak())
np.savez(RESULTS / f"colour_native_{NAT}.npz", cols=np.array(COLS), Y0=Y0, angle=angle)
