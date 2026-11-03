"""The directions the expert reads. y = the final answer after the full 10-step flow (mean over the 50 rows of coordinate 0, degrees).
G_K[L] = dy/dK[L][image rows 0..63], G_V[L] = dy/dV[L][image rows]; 72 fit scenes phi = -180..175 step 5, theta_s = 30, shared noise. Per layer the SVD of
the stacked (n*64, 320) gradients gives ordered directions D_K[L], D_V[L] (top 32 saved) and singular values; the 90% cut is applied by the readers.
Also saves the frozen model's k_proj / v_proj / input_layernorm weights to data/vlm_projection_weights.npz for the pull-back into the stream.
Output: results/gradient_directions.json. Runtime about 1 min; backward through the flow needs about 10 GB of GPU memory."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, hashlib, numpy as np, torch
from toyvla import policy_tools
from toyvla.paths import DEVICE, POLICY, GRADIENT_DIRECTIONS, VLM_WEIGHTS, SMOKE
NS = 10
S = policy_tools.S; pol, m = policy_tools.load(POLICY["red"]); H = policy_tools.Hooks(m); vlm = H.vlm
for p in m.parameters(): p.requires_grad_(False)
f = lambda t: t.detach().float().cpu().numpy().astype(np.float64)
Wf = lambda a: (np.asarray(a) + 180) % 360 - 180
fit_phi = np.arange(-180, 180, 5.0)
if SMOKE: fit_phi = fit_phi[::12]
fit_sc = [S.scene_at(float(p), 30.0) for p in fit_phi]; n = len(fit_sc); BS = 12
def flow(pm, kv, noise, grad):
    x = noise.clone(); B = x.shape[0]; t = torch.tensor(1.0, device=DEVICE); dt = torch.tensor(-1.0 / NS, device=DEVICE)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for k in range(NS):
            v = m.denoise_step(pm, kv, x, t.expand(B)); x = x + dt * v; t = t + dt
    return x[:, :, 0].mean(1) * 180
GK = {L: [] for L in range(16)}; GV = {L: [] for L in range(16)}; ys = []
for j in range(0, n, BS):
    inp = policy_tools.make_inputs(pol, fit_sc[j:j+BS]); pm, kv = H.prefix(inp); B = pm.shape[0]
    leaf = {}
    for L in range(16):
        d = {}
        for key in ("key_states", "value_states"):
            t_ = kv[L][key].detach().float().clone().requires_grad_(True); d[key] = t_; leaf[(L, key)] = t_
        kv[L] = d
    y = flow(pm, kv, policy_tools.noise_for(B, 0), grad=True); y.sum().backward(); ys.append(y.detach().cpu().numpy())
    for L in range(16):
        for key, G in (("key_states", GK), ("value_states", GV)):
            g = leaf[(L, key)].grad[:, :64]; G[L].append(g.reshape(B, 64, -1).cpu().numpy().astype(np.float64))
GK = {L: np.concatenate(v) for L, v in GK.items()}; GV = {L: np.concatenate(v) for L, v in GV.items()}
eK = np.array([(GK[L]**2).sum() for L in range(16)]); eV = np.array([(GV[L]**2).sum() for L in range(16)]); tot = eK.sum() + eV.sum()
DK, DV, sK, sV = {}, {}, {}, {}
for L in range(16):
    _, s_, Vt = np.linalg.svd(GK[L].reshape(-1, GK[L].shape[-1]), full_matrices=False); DK[L] = Vt[:32]; sK[L] = s_
    _, s_, Vt = np.linalg.svd(GV[L].reshape(-1, GV[L].shape[-1]), full_matrices=False); DV[L] = Vt[:32]; sV[L] = s_
yy = np.concatenate(ys); err = np.abs(Wf(yy - fit_phi))
hsh = hashlib.sha256()
for L in range(16):
    a = vlm.layers[L]
    for t_ in (a.self_attn.k_proj.weight, a.self_attn.v_proj.weight, a.input_layernorm.weight): hsh.update(t_.detach().float().cpu().numpy().tobytes())
print(f"n_fit {n}  fit-scene error median {np.median(err):.3f} mean {err.mean():.3f} max {err.max():.3f}")
print("gradient energy share K/V per layer: " + "  ".join(f"L{L}:{eK[L]/tot:.3f}/{eV[L]/tot:.3f}" for L in range(16)))
print("vlm kv hash", hsh.hexdigest()[:16])
VLM_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
np.savez(VLM_WEIGHTS, **{f"{nm}{L}": f(getattr(vlm.layers[L].self_attn, nm2).weight) for L in range(16) for nm, nm2 in (("Wk", "k_proj"), ("Wv", "v_proj"))},
         **{f"gam{L}": f(vlm.layers[L].input_layernorm.weight) for L in range(16)})
json.dump(dict(nstep=NS, fit_phi=fit_phi.tolist(), y_fit=yy.tolist(), err_fit_median=float(np.median(err)), err_fit_mean=float(err.mean()),
               grad_share=dict(K=(eK / tot).tolist(), V=(eV / tot).tolist()), grad_energy_total=float(tot),
               sK={L: sK[L][:64].tolist() for L in range(16)}, sV={L: sV[L][:64].tolist() for L in range(16)},
               DK={L: DK[L].tolist() for L in range(16)}, DV={L: DV[L].tolist() for L in range(16)}, vlm_kv_hash=hsh.hexdigest()), open(GRADIENT_DIRECTIONS, "w"))
