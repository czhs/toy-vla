"""Figure 2: three bars per chosen colour: a policy trained on that colour with no edit (128 scenes), the red policy with no edit, and the red policy with the switch (2,048 scenes). Medians of the
error to the chosen square with 2,000-draw bootstrap 95% intervals."""
from common import OUT, RES
import numpy as np, json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
d = lambda y, t: np.abs((y - t + 180) % 360 - 180); rng = np.random.default_rng(20261103)
Z = np.load(RES / 'colour_switch.npz'); cols = [str(c) for c in Z['cols']]
NAME = {'red': 'red', 'hue120': 'green', 'hue240': 'blue'}; FILL = {'red': '#b3202e', 'hue120': '#4f9d3a', 'hue240': '#2b5fb3'}
def stat(e, B=2000):
    e = np.asarray(e); m = float(np.median(e)); b = [np.median(rng.choice(e, len(e))) for _ in range(B)]; lo, hi = np.percentile(b, [2.5, 97.5]); return m, float(lo), float(hi), len(e)
out = {}; fig, ax = plt.subplots(figsize=(5.2, 2.6)); x = np.arange(3); w = 0.26
for ci, c in enumerate(cols):
    if c == 'red': native = stat(d(Z['Y0'][:128], Z['angle'][:128, ci]))   # the red policy is the policy trained on red
    else: Zn = np.load(RES / f'colour_native_{NAME[c]}.npz'); assert np.allclose(Zn['angle'], Z['angle'][:128], equal_nan=True); native = stat(d(Zn['Y0'], Zn['angle'][:, ci]))
    bars = [('policy trained on that colour, no edit', native, dict(facecolor='white', edgecolor=FILL[c], hatch='////')),
            ('red policy, no edit', stat(d(Z['Y0'], Z['angle'][:, ci])), dict(facecolor='#d9d9d9', edgecolor=FILL[c])),
            ('red policy with the switch', stat(d(Z['Y'][ci], Z['angle'][:, ci])), dict(facecolor=FILL[c], edgecolor=FILL[c]))]
    for k, (lab, (m, lo, hi, n), sty) in enumerate(bars):
        xx = x[ci] + (k - 1) * w; ax.bar(xx, m, w - 0.03, lw=0.8, zorder=3, **sty); ax.errorbar(xx, m, yerr=[[m - lo], [hi - m]], color='#222', lw=0.8, capsize=2, zorder=4)
        ax.text(xx, hi + 1.5, f'{m:.0f}', ha='center', va='bottom', fontsize=6.5, color='#222'); out[f'{NAME[c]}|{lab}'] = dict(median=m, ci95=[lo, hi], n=n)
ax.set_xticks(x); ax.set_xticklabels([f'go to {NAME[c]}' for c in cols], fontsize=8); ax.set_ylabel('median error to the chosen square (°)', fontsize=8); ax.set_ylim(0, 100); ax.tick_params(labelsize=7)
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
ax.legend([Patch(facecolor='white', edgecolor='#555', hatch='////'), Patch(facecolor='#d9d9d9', edgecolor='#555'), Patch(facecolor='#555')], ['policy trained on that colour, no edit', 'red policy, no edit', 'red policy with the switch'], loc='upper left', frameon=False, fontsize=6.5)
plt.tight_layout(); plt.savefig(OUT / 'fig3_switch.pdf'); plt.savefig(OUT / 'fig3_switch.png', dpi=160)
json.dump(out, open(OUT / 'fig3_switch.json', 'w'), indent=1); print({k: round(v['median'], 2) for k, v in out.items()})
