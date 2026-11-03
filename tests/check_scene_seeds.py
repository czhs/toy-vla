"""Regenerate the scenes of every toy experiment from their seeds and compare them with the stored outputs in results/. CPU, about a minute; no model is loaded."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, numpy as np
from toyvla import scene as S, colour_scenes as CG, pointer_plants as GM
from toyvla.paths import RESULTS
def panel(seed, n): return [S.make_scene(np.random.default_rng([seed, i]), 'full') for i in range(n)]
def check(name, a, b): a, b = np.asarray(a, float), np.asarray(b, float); assert a.shape == b.shape and np.allclose(a, b, equal_nan=True), name; print('ok', name, a.shape)
PAL = CG.palette(S); COLS = list(PAL)
k = np.load(RESULTS / 'keep_erase.npz'); check('keep_erase', [s['phi'] for s in panel(61002, 256)], k['phi'])
g = json.load(open(RESULTS / 'gradient_directions.json')); check('gradient fit grid', np.arange(-180, 180, 5.0), g['fit_phi'])
d = np.load(RESULTS / 'distractor_one_square.npz'); DIST = ['hue120', 'hue240']
sc = [CG.compose(S, np.random.default_rng([61070, i]), PAL, k=1, cols=[DIST[i % 2]]) for i in range(256)]
check('distractor block', [s['phi'] for s, _, _ in sc], d['phi']); check('distractor square', [q[0][1] for _, q, _ in sc], d['dang'])
z = np.load(RESULTS / 'colour_switch.npz'); scs = [CG.compose(S, np.random.default_rng([61060, i]), PAL, k=2) for i in range(2048)]; ang = np.full((2048, 3), np.nan)
for i, (s, q, _) in enumerate(scs):
    ang[i, 0] = s['phi']
    for c, a in q: ang[i, COLS.index(c)] = a
check('colour switch', ang, z['angle']); check('colour switch, red only', [CG.compose(S, np.random.default_rng([61061, i]), PAL, k=0)[0]['phi'] for i in range(512)], z['ptgt'])
q = np.load(RESULTS / 'query_replacement.npz'); B, i = [], 0
while len(B) < 128:
    s = S.make_scene(np.random.default_rng([61003, i]), 'full'); i += 1
    if abs(s['phi']) >= 90: B.append(s)
check('query replacement', [s['phi'] for s in B + panel(61002, 48)], q['phi'])
w = np.load(RESULTS / 'weight_fold.npz'); check('weight fold', [s['phi'] for s in panel(61030, 288)], w['phi']); check('weight fold, fit', [s['phi'] for s in panel(61004, 96)], w['fit_phi'])
p = np.load(RESULTS / 'pointer_test.npz'); assert [s['digit'] for s in GM.ring8_scenes(4000, seed=20261300)] == p['digit'].tolist(); print('ok pointer test digits')
p = np.load(RESULTS / 'pointer_layer_sweep.npz'); assert [s['digit'] for s in GM.ring8_scenes(200)] == p['digit'].tolist(); print('ok pointer sweep digits')
