"""Figure 3a: error of the toy policy by block angle for the unedited weights, the query weights folded to a fixed estimate of +120 and of -120 degrees, and the same construction in a random plane;
288 scenes. Points are scenes; lines are medians in 15-degree bins."""
from common import OUT, RES
import numpy as np, json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
z = np.load(RES / 'weight_fold.npz'); phi, tgt = z['phi'], z['tgt']; wrap = lambda a: (a + 180) % 360 - 180
SER = [('y_identity', 'unedited', '#9a9a9a'), ('y_tmpl_+120', 'fixed estimate $+120°$', '#b3202e'), ('y_tmpl_-120', 'fixed estimate $-120°$', '#2b5fb3'), ('y_rand_+120', 'random plane, $+120°$', '#4f9d3a')]
fig, ax = plt.subplots(figsize=(4.0, 2.6)); bins = np.arange(-180, 181, 15); ctr = (bins[:-1] + bins[1:]) / 2; out = {}
for key, lab, col in SER:
    e = np.abs(wrap(z[key] - tgt)); ax.scatter(phi, e, s=5, c=col, alpha=0.45, lw=0, label=lab)
    med = [np.median(e[(phi >= a) & (phi < b)]) if ((phi >= a) & (phi < b)).any() else np.nan for a, b in zip(bins[:-1], bins[1:])]; ax.plot(ctr, med, c=col, lw=1.4)
    out[key] = dict(median_pos=float(np.median(e[phi > 0])), median_neg=float(np.median(e[phi < 0])), n_pos=int((phi > 0).sum()), n_neg=int((phi < 0).sum()))
ax.set_xlim(-180, 180); ax.set_xticks([-180, -90, 0, 90, 180]); ax.set_xlabel('block angle (°)', fontsize=8); ax.set_ylabel('error of the answer (°)', fontsize=8); ax.set_ylim(0, 82)
ax.tick_params(labelsize=7); ax.legend(fontsize=6.5, frameon=False, loc='upper center', ncol=2)
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig(OUT / 'fig4a_toy.pdf'); plt.savefig(OUT / 'fig4a_toy.png', dpi=160); json.dump(out, open(OUT / 'fig4a_toy.json', 'w'), indent=1); print(out)
