"""Remove groups of expert layers at once. At every flow step the attention output (o_proj) and / or the MLP output of the chosen layers on the 50
action rows is replaced by zero (skip the sublayer) or by its mean over scenes at that step. Scenes: phi on a 5-degree grid (72), theta_s 30,
noise seed 0. The paper uses the median final error of `intact`, `all_even|zero|attn+mlp` and `all_odd|zero|attn+mlp`.
Output: results/even_layer_skip.json. Runtime about 1 min on a GPU."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, numpy as np
from toyvla import policy_tools
from toyvla.paths import POLICY, RESULTS, SMOKE
S = policy_tools.S; pol, m = policy_tools.load(POLICY['red']); H = policy_tools.Hooks(m); ex = H.ex
phis = np.arange(-180, 180, 5.0)
if SMOKE: phis = phis[::18]
n = len(phis); inp = policy_tools.make_inputs(pol, [S.scene_at(float(p), 30.0) for p in phis]); noise = policy_tools.noise_for(n, 0)
Wr = lambda a: (np.asarray(a) + 180) % 360 - 180
err = lambda o: (float(np.median(np.abs(Wr(o["final"][:, 0] - phis)))), float(np.median(np.abs(Wr(o["xhat"][0, :, 0] - phis)))))
means = {}
def rec(kind, L):
    def f(mod, a, o):
        if H.phase == 'denoise': means[(kind, L, H.step)] = o[:, -50:].float().mean(0, keepdim=True).clone()
    return f
hs = [ex.layers[L].self_attn.o_proj.register_forward_hook(rec('attn', L)) for L in range(16)] + [ex.layers[L].mlp.register_forward_hook(rec('mlp', L)) for L in range(16)]
base = err(H.run(inp, noise))
for h in hs: h.remove()
print(f"intact: final {base[0]:.2f}  step-0 xhat {base[1]:.2f}", flush=True)
ctl = {'layers': set(), 'kinds': (), 'mode': None}
def edit(kind, L):
    def f(mod, a, o):
        if H.phase != 'denoise' or L not in ctl['layers'] or kind not in ctl['kinds']: return None
        o = o.clone()
        if ctl['mode'] == 'zero': o[:, -50:] = 0
        else: o[:, -50:] = means[(kind, L, H.step)].to(o.dtype).expand(o.shape[0], -1, -1)
        return o
    return f
hs = [ex.layers[L].self_attn.o_proj.register_forward_hook(edit('attn', L)) for L in range(16)] + [ex.layers[L].mlp.register_forward_hook(edit('mlp', L)) for L in range(16)]
EVEN, ODD = list(range(0, 16, 2)), list(range(1, 16, 2))
SETS = {'all_even': EVEN, 'even_except_0': EVEN[1:], 'all_odd': ODD, 'even_0_2': [0, 2], 'even_4_14': EVEN[2:]}
res = {}
for name, Ls in SETS.items():
    if SMOKE and name not in ('all_even', 'all_odd'): continue
    for mode in ('zero',) if SMOKE else ('zero', 'mean'):
        for kinds in (('attn', 'mlp'),) if SMOKE else (('attn',), ('mlp',), ('attn', 'mlp')):
            ctl.update(layers=set(Ls), mode=mode, kinds=kinds); o = H.run(inp, noise); r = err(o); res[f'{name}|{mode}|{"+".join(kinds)}'] = r
            print(f"{name:14s} {mode:4s} {'+'.join(kinds):9s} final {r[0]:7.2f}  step-0 xhat {r[1]:7.2f}", flush=True)
json.dump({'n_scenes': int(n), 'intact': base, 'results': res, 'sets': SETS}, open(RESULTS / 'even_layer_skip.json', 'w'), indent=1)
