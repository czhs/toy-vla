import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Per head at layers 5-15: block-tile attention mass as a function of the absolute estimate A written by experiments/estimate_gate_scan.py. A head fires on a scene when its intact
block-tile mass is at least 0.5; its sector is the circular mean of the block angles on which it fires. Heads analysed: those firing on at least 8 of the 176 scenes, plus the
heads of results/head_ranking_top16.json (a ranking from a single-head query scan, kept as a fixed list). tmpl[i, a] is the median block mass over head i's firing scenes at A.
CPU only, seconds. Output: results/head_gating.npz."""
import json, numpy as np
from toyvla.paths import DATA, RESULTS
OD = DATA / "estimate_gate_scan"
ATT = np.load(OD / "attention.npy").astype(np.float32); Y = np.load(OD / "answers.npy"); M = json.load(open(OD / "meta.json"))
C, N = Y.shape; EL = M["edited_layers"]; phi = np.array(M["phi"]); tok = np.array(M["tok"]); conds = [tuple(c) for c in M["conds"]]; ci = {c: i for i, c in enumerate(conds)}; GRID = M["grid"]
blk = np.stack([ATT[:, n, :, :, tok[n]] for n in range(N)], 1); fire = blk[ci[("intact", 0)]] >= 0.5                 # (C, N, 6, 15)
top = [tuple(x) for x in json.load(open(RESULTS / "head_ranking_top16.json"))["top16"]]
def cmean(a): return np.degrees(np.arctan2(np.sin(np.radians(a)).mean(), np.cos(np.radians(a)).mean()))
heads = []
for li, l in enumerate(EL):
    for h in range(15):
        n = int(fire[:, li, h].sum())
        if n >= 8 or (l, h) in top: heads.append((l, h, li, n, cmean(phi[fire[:, li, h]]) if n else np.nan))
heads.sort(key=lambda x: (top.index((x[0], x[1])) if (x[0], x[1]) in top else 99, -x[3]))
np.savez(RESULTS / "head_gating.npz", heads=np.array([(l, h, n) for l, h, li, n, s_ in heads]), sectors=np.array([s_ for l, h, li, n, s_ in heads]), grid=np.array(GRID),
         tmpl=np.array([[np.median(blk[ci[("tmpl", a)], fire[:, li, h], li, h]) if n >= 3 else np.nan for a in GRID] for l, h, li, n, s_ in heads]))
print(len(heads), "heads written")
