"""Recompute every number of the paper from the per-scene outputs in results/ and compare it with the value printed in the paper (numbers/paper_values.json). No model is run. python numbers/recompute_numbers.py"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / 'results'
wrap = lambda a: (np.asarray(a, float) + 180) % 360 - 180
err = lambda y, t: np.abs(wrap(np.asarray(y) - np.asarray(t)))
med = lambda y, t: float(np.median(err(y, t)))
N = {}

# ---- even layers
e = json.load(open(RES / 'even_layer_skip.json'))
N['even.n_scenes'] = e['n_scenes']; N['even.intact'] = e['intact'][0]; N['even.even_removed'] = e['results']['all_even|zero|attn+mlp'][0]; N['even.odd_removed'] = e['results']['all_odd|zero|attn+mlp'][0]

# ---- directions, keep and erase
g = json.load(open(RES / 'gradient_directions.json')); ODD = list(range(1, 16, 2))
def count(s, frac): e_ = np.array(s, float) ** 2; return int(np.searchsorted(np.cumsum(e_) / e_.sum(), frac) + 1)
for kind in 'KV':
    for pct, frac in ((90, 0.90), (99, 0.99)):
        c = [count(g['s' + kind][str(L)], frac) for L in ODD]; N[f'dirs.{kind}{pct}'] = c; N[f'dirs.{kind}{pct}.total'] = sum(c)
N['dirs.total90'] = N['dirs.K90.total'] + N['dirs.V90.total']; N['dirs.total99'] = N['dirs.K99.total'] + N['dirs.V99.total']
share = np.array(g['grad_share']['K']) + np.array(g['grad_share']['V'])
N['dirs.share_odd'] = float(share[1::2].sum()); N['dirs.share_even'] = float(share[0::2].sum()); N['dirs.share_per_odd_layer'] = share[1::2].tolist()
N['dirs.fit_scenes'] = len(g['fit_phi']); N['dirs.fit_error'] = g['err_fit_median']
c = json.load(open(RES / 'stream_gradient_overlap.json'))['layers']
N['stream.share_union_odd'] = [c[str(L)]['share_union_odd'] for L in ODD]; N['stream.share_union_odd_random'] = [c[str(L)]['share_union_odd_rand'] for L in ODD]; N['stream.ov_local_noJ'] = [c[str(L)]['ov_local_noJ'] for L in ODD]
k = np.load(RES / 'keep_erase.npz')
for name in ('intact', 'keep_gradient', 'erase_gradient', 'keep_rand', 'erase_rand', 'keep_pca', 'erase_pca', 'mean'): N[f'keep.{name}'] = med(k[name], k['phi'])

# ---- pointer test table
acc = lambda z, key: float((z[key] == z['digit']).mean())
z = np.load(RES / 'pointer_test.npz')
for c_, key in (('none', 'none'), ('box', 'box'), ('kv8', 'L7_m8'), ('kv8_random', 'rand_L7_m8'), ('v10', 'L7V_m10'), ('v12', 'L7V_m12'), ('v14', 'L7V_m14'), ('v16', 'L7V_m16'), ('v14_random', 'rand_L7V_m14'), ('k8', 'L7K_m8')):
    N[f'pointer4000.{c_}'] = acc(z, f'{key}|paper')
N['pointer4000.n'] = len(z['digit'])
s = np.load(RES / 'pointer_layer_sweep.npz')
for key in ('none', 'box', 'all_odd_m2', 'rand_all_odd_m2'): N[f'sweep.{key}'] = acc(s, f'{key}|selection')
for L in ODD:
    for m in (1, 2, 4, 8) + ((12, 16) if L in (5, 7, 9) else ()):
        N[f'sweep.L{L}.m{m}'] = acc(s, f'L{L}_m{m}|selection'); N[f'sweep.L{L}.m{m}.random'] = acc(s, f'rand_L{L}_m{m}|selection')
N['sweep.n'] = len(s['digit'])

# ---- distractor and colour switch
d = np.load(RES / 'distractor_one_square.npz')
N['distractor.n'] = len(d['phi'])
for name in ('intact', 'nodist_cache', 'erase_read_dirs', 'keep_read_dirs', 'keep_random'): N[f'distractor.{name}'] = med(d[f'y_{name}'], d['phi'])
z = np.load(RES / 'colour_switch.npz'); cols = [str(x) for x in z['cols']]; name = {'red': 'red', 'hue120': 'green', 'hue240': 'blue'}
N['switch.n'] = len(z['Y0']); N['switch.n_red_only'] = len(z['P0'])
for ci, c_ in enumerate(cols):
    N[f'switch.to_{name[c_]}'] = med(z['Y'][ci], z['angle'][:, ci]); N[f'switch.untouched_to_{name[c_]}'] = med(z['Y0'], z['angle'][:, ci])
N['switch.red_only_edit'] = med(z['PG'], z['ptgt']); N['switch.red_only_intact'] = med(z['P0'], z['ptgt'])
for ci, c_ in enumerate(cols[1:], 1):
    y = np.load(RES / f'colour_native_{name[c_]}.npz'); assert np.allclose(y['angle'], z['angle'][:128], equal_nan=True)
    N[f'switch.native_{name[c_]}'] = med(y['Y0'], y['angle'][:, ci])

# ---- the estimate in the query input
q = np.load(RES / 'query_replacement.npz'); conds = [str(x) for x in q['conds']]; ends = q['panel'] == 'ends'; phi = q['phi']; pos, neg = ends & (phi > 0), ends & (phi < 0)
ee = lambda c_: err(q['y'][conds.index(c_)], q['tgt'])
N['gate.n_ends'] = int(ends.sum()); N['gate.n_pos'] = int(pos.sum()); N['gate.n_neg'] = int(neg.sum()); N['gate.n_all'] = len(phi)
for label, c_ in (('intact', 'intact|0'), ('mean', 'mean_only|0')): N[f'gate.{label}'] = [float(np.median(ee(c_)[pos])), float(np.median(ee(c_)[neg]))]
for A in (-150, -105, -60, -30, 0, 30, 60, 90, 105, 120, 135, 150, 165, 180): N[f'gate.A{A}'] = [float(np.median(ee(f'mean_plus_A|{A}')[pos])), float(np.median(ee(f'mean_plus_A|{A}')[neg]))]
for A in (-150, 0, 90, 150): N[f'gate.random_A{A}'] = [float(np.median(ee(f'mean_plus_A_random|{A}')[pos])), float(np.median(ee(f'mean_plus_A_random|{A}')[neg]))]
f = json.load(open(RES / 'estimate_fit.json')); N['estimate.fit_n'] = f['fit_N']
N['estimate.variance_fraction'] = [f['fit'][f'L{L}']['ev_frac_top2'] for L in f['edited_layers']]; N['estimate.line_r2'] = [f['fit'][f'L{L}']['line_r2'] for L in f['edited_layers']]
h = np.load(RES / 'head_gating.npz'); gated = open_ = rev = n_h = 0                                    # A' = A * sign(sector): far side A' <= -90, own side A' >= 60
for sec, t_ in zip(h['sectors'], h['tmpl']):
    if np.isnan(sec): continue
    Ap = h['grid'] * np.sign(sec); fa, ow = np.nanmedian(t_[Ap <= -90]), np.nanmedian(t_[Ap >= 60]); n_h += 1
    gated += (fa <= 0.1) & (ow >= 0.5); open_ += (fa >= 0.5) & (ow >= 0.5); rev += (fa >= 0.5) & (ow <= 0.1)
N['heads.n'] = n_h; N['heads.gated'] = int(gated); N['heads.open'] = int(open_); N['heads.reversed'] = int(rev)

# ---- toy weight fold
w = np.load(RES / 'weight_fold.npz'); p, t_ = w['phi'], w['tgt']
N['fold.n_pos'] = int((p > 0).sum()); N['fold.n_neg'] = int((p < 0).sum()); N['fold.plane_r2'] = w['r2'].tolist()
for label, key in (('unedited', 'y_identity'), ('A+120', 'y_tmpl_+120'), ('A-120', 'y_tmpl_-120'), ('random_A+120', 'y_rand_+120'), ('removed', 'y_erase')):
    N[f'fold.{label}'] = [float(np.median(err(w[key], t_)[p > 0])), float(np.median(err(w[key], t_)[p < 0]))]

# ---- the arm
arm = RES / 'arm' / 'closed_loop.npz'
if arm.exists():
    a = np.load(arm)
    def band(s, b): return [float(s[(b >= lo) & (b < hi)].mean()) for lo, hi in ((-60, -40), (-40, -20), (-20, 0), (0, 20), (20, 40), (40, 61))]
    def region(tag):
        s = a[f'{tag}_success'].astype(float); xy = a[f'{tag}_cube_xy']; return s, np.degrees(np.arctan2(xy[:, 1], xy[:, 0]))
    for label, tag, kept in (('unedited', 'big_unedited', None), ('A+60', 'big_plus60', (-60, -20)), ('A-60', 'big_minus60', (20, 60))):
        s_, b = region(tag); N[f'arm.{label}.all'] = float(s_.mean()); N[f'arm.{label}.n'] = len(s_); N[f'arm.{label}.bands'] = band(s_, b)
        if kept: m = (b >= kept[0]) & (b <= kept[1]); N[f'arm.{label}.kept'] = float(s_[m].mean()); N[f'arm.{label}.elsewhere'] = float(s_[~m].mean()); N[f'arm.{label}.n_kept'] = int(m.sum()); N[f'arm.{label}.n_elsewhere'] = int((~m).sum())
    for label, tag in (('dev_unedited', 'dev_unedited'), ('dev_fold', 'dev_fold'), ('fresh_unedited', 'fresh_unedited'), ('fresh_fold', 'fresh_fold')):
        s_, b = region(tag); m = (b >= -60) & (b <= -20); N[f'arm.{label}'] = [float(s_[m].mean()), float(s_[~m].mean())]

# ---- compare
EXP = json.load(open(ROOT / 'numbers' / 'paper_values.json'))
def same(x, want, tol):
    if isinstance(want, list): return isinstance(x, list) and len(x) == len(want) and all(same(a_, b_, tol) for a_, b_ in zip(x, want))
    return abs(x - want) <= tol
bad = missing = 0
for k_, spec in EXP.items():
    if k_ not in N: print(f'MISSING  {k_}'); missing += 1; continue
    ok = same(N[k_], spec['value'], spec['tol']); bad += not ok
    print(f"{'ok      ' if ok else 'MISMATCH'} {k_:32s} recomputed {np.round(N[k_], 4).tolist() if isinstance(N[k_], (list, float)) else N[k_]}   paper {spec['value']}")
extra = sorted(set(N) - set(EXP))
json.dump(N, open(ROOT / 'numbers' / 'recomputed.json', 'w'), indent=1)
print(f'\n{len(EXP) - bad - missing} of {len(EXP)} match, {bad} mismatch, {missing} missing; computed but not listed in the paper values: {len(extra)}')
sys.exit(1 if bad else 0)
