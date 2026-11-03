"""The pointer test as a figure (the draft reports the same data in a table and does not include this figure). (a) one-layer plants by layer and strength, selection question, 200 scenes;
(b) layer-7 plants with the paper's question, 4,000 scenes, Wilson 95% intervals."""
from common import OUT, RES
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
S = np.load(RES / 'pointer_layer_sweep.npz'); B = np.load(RES / 'pointer_test.npz')
acc = lambda Z, k: float((Z[k] == Z['digit']).mean())
layers = [1, 3, 5, 7, 9, 11, 13, 15]; strengths = [1, 2, 4, 8]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.3, 1]))
for s, c in zip(strengths, ['#c0c0c0', '#8c8c8c', '#555555', '#111111']):
    ax[0].plot(layers, [acc(S, f'L{L}_m{s}|selection') for L in layers], '-o', color=c, ms=3.5, lw=1.2, label=f'strength {s}')
ax[0].plot(layers, [max(acc(S, f'rand_L{L}_m{s}|selection') for s in strengths) for L in layers], ':', color='#999999', lw=1, label='random direction (best of the four)')
for k, lab, ls in [('none', 'no edit', '--'), ('box', 'drawn box', '--'), ('all_odd_m2', 'all eight layers, strength 2', '-.')]:
    y = acc(S, f'{k}|selection'); ax[0].axhline(y, color='#aaaaaa', lw=0.8, ls=ls); ax[0].text(15.6, y, lab, fontsize=6.5, va='center', ha='left', color='#555555', bbox=dict(fc='white', ec='none', pad=0.5))
ax[0].set_xticks(layers); ax[0].set_xlabel('layer of the vision-language model edited', fontsize=8); ax[0].set_ylabel('accuracy', fontsize=8); ax[0].set_ylim(0, 1); ax[0].set_xlim(0.5, 22)
ax[0].legend(fontsize=6.5, frameon=False, loc='upper left'); ax[0].set_title('(a) one layer, keys and values, 200 scenes', fontsize=8)
conds = [('none', 'no edit'), ('rand_L7V_m14', 'random, same norm'), ('L7_m8', 'keys and values, 8'), ('L7V_m10', 'values, 10'), ('L7V_m12', 'values, 12'), ('L7V_m14', 'values, 14'), ('L7V_m16', 'values, 16'), ('box', 'drawn box')]
ys = np.array([acc(B, f'{c}|paper') for c, _ in conds]); n = len(B['digit']); z = 1.96; d = 1 + z * z / n; ctr = (ys + z * z / (2 * n)) / d; h = z * np.sqrt(ys * (1 - ys) / n + z * z / (4 * n * n)) / d
x = np.arange(len(conds)); ax[1].bar(x, ys, color=['#c8c8c8', '#c8c8c8', '#808080', '#606060', '#484848', '#111111', '#484848', '#c8c8c8'], width=0.7)
ax[1].errorbar(x, ys, yerr=[ys - (ctr - h), (ctr + h) - ys], fmt='none', ecolor='k', lw=0.8, capsize=2)
ax[1].set_xticks(x); ax[1].set_xticklabels([l for _, l in conds], rotation=40, ha='right', fontsize=6.5); ax[1].set_ylim(0, 1); ax[1].set_ylabel('accuracy', fontsize=8); ax[1].set_title("(b) layer 7, the paper's question, 4,000 scenes", fontsize=8)
for a in ax: a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.tick_params(labelsize=7)
plt.tight_layout(); plt.savefig(OUT / 'fig2_pointer.pdf'); plt.savefig(OUT / 'fig2_pointer.png', dpi=160)
