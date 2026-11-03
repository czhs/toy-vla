import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Calibration of the colour switch from 32 scenes (seed [61050, 0, k]): an arm-only render and three renders of the same scene with one red / green / blue square each at
ring angles at least 40 degrees apart. Gives the reference rows Kref / Vref of every odd layer (mean arm-only rows per slot), per-colour layer-3 key directions x_X
(coverage-weighted mean key delta of X's tokens with RoPE undone, minus its projection on the span of the other two, normalised), their arm-only medians m_x, and the
scale theta0 (99th percentile of the arm-only tokens' coordinate on the top singular vector of the three deltas) that sets the detector's ramp.
Output: results/colour_calibration.npz. Runtime under a minute on a GPU."""
import numpy as np, torch
from toyvla import query_tools as Q, colour_scenes as CG
from toyvla.paths import COLOUR_CALIBRATION, SMOKE
V = Q.V; ODD = CG.ODD
BS = 16; COV_T = 0.05; NCAL = 2 if SMOKE else 32; SEED = 0
S, pol, m, H, st = V.setup()
AR = (-180.0, 180.0)
PAL = CG.palette(S); COLS = list(PAL)
scs_c, arm_c, imgs_c, cov_c, angs_c = [], [], {X: [] for X in COLS}, {X: [] for X in COLS}, []
for k in range(NCAL):
    rng = np.random.default_rng([61050, SEED, k]); sc = S.make_scene(rng, 'full'); angs = []
    while len(angs) < 3:
        a = float(rng.uniform(*AR))
        if all(abs(CG.wrap(a - b)) >= 40 for b in angs): angs.append(a)
    arm_only = CG.render_objs(S, sc, 'plain', objs=(), red=False); scs_c.append(sc); arm_c.append(arm_only); angs_c.append(angs)
    for X, a in zip(COLS, angs):
        im = CG.render_objs(S, sc, 'plain', objs=[CG.obj('square', *S.block_xy(a), PAL[X][1])], red=False); imgs_c[X].append(im)
        cov_c[X].append(np.any(im != arm_only, -1).reshape(8, 64, 8, 64).mean((1, 3)).reshape(64))
# captures: per scene the arm-only render and the three square renders -> post-RoPE keys / values at all 113 slots, per odd layer
ORDER = ['arm'] + COLS; allsc = [sc for sc in scs_c for _ in ORDER]; allim = [im for k in range(NCAL) for im in [arm_c[k]] + [imgs_c[X][k] for X in COLS]]
Kpost, Vv = {L: [] for L in ODD}, {L: [] for L in ODD}
for j in range(0, len(allsc), BS):
    kv = V.Embs(H, pol, allsc[j:j + BS], imgs=allim[j:j + BS]).kv(); B = len(allsc[j:j + BS])
    for L in ODD:
        Kpost[L].append(kv[L]['key_states'][:, :CG.NSL].float().reshape(B, CG.NSL, 320).cpu().numpy().astype(np.float64))
        Vv[L].append(kv[L]['value_states'][:, :CG.NSL].float().reshape(B, CG.NSL, 320).cpu().numpy().astype(np.float64))
Kpost = {L: np.concatenate(Kpost[L]).reshape(NCAL, 4, CG.NSL, 320) for L in ODD}; Vv = {L: np.concatenate(Vv[L]).reshape(NCAL, 4, CG.NSL, 320) for L in ODD}
Kc = {L: CG.rope_back(Kpost[L].reshape(NCAL * 4, CG.NSL, 320)[:, :64]).reshape(NCAL, 4, 64, 320) for L in ODD}
cov = {X: np.stack(cov_c[X]) for X in COLS}
unit = lambda v: v / np.linalg.norm(v); L = CG.GL; ix = {X: 1 + COLS.index(X) for X in COLS}
mK = {X: np.mean([(cov[X][k][:, None] * (Kc[L][k, ix[X]] - Kc[L][k, 0])).sum(0) / cov[X][k].sum() for k in range(NCAL)], 0) for X in COLS}
s3 = np.linalg.svd(np.stack([mK[X] for X in COLS]), full_matrices=False)[2][0]; s3 = s3 if s3 @ mK['red'] > 0 else -s3
x3, mKx = {}, {}; c0 = Kc[L][:, 0].reshape(-1, 320)
for X in COLS:
    Bk = np.linalg.svd(np.stack([mK[Y] for Y in COLS if Y != X]), full_matrices=False)[2]; x3[X] = unit(mK[X] - Bk.T @ (Bk @ mK[X])); mKx[X] = float(np.median(c0 @ x3[X]))
sig0 = c0 @ s3; theta0 = float(np.percentile(sig0 - np.median(sig0), 99))
np.savez(COLOUR_CALIBRATION, cols=np.array(COLS), **{f'x3_{X}': x3[X] for X in COLS}, **{f'mKx_{X}': mKx[X] for X in COLS}, theta0=theta0,
         Kref=np.stack([Kpost[q][:, 0].mean(0) for q in ODD]), Vref=np.stack([Vv[q][:, 0].mean(0) for q in ODD]))
Q.log('theta0', theta0, 'mKx', mKx)
