"""Keep-only and erase edits in the 320-dim cache space of an odd layer (V: value_states rows; K: key_states rows, already RoPE-rotated)."""
import json, numpy as np, torch
from toyvla import policy_tools
from toyvla.paths import DEVICE, POLICY, GRADIENT_DIRECTIONS
ODD = list(range(1, 16, 2)); KEYN = {"V": "value_states", "K": "key_states"}
NS = 10


def setup(colour="red"):
    S = policy_tools.S; pol, m = policy_tools.load(POLICY[colour]); H = policy_tools.Hooks(m)
    for p in m.parameters(): p.requires_grad_(False)
    return S, pol, m, H


def read_directions():
    """per (L, 'V'/'K'): orthonormal rows D (k, 320) = leading gradient directions holding 90% of that layer/kind's gradient energy."""
    G = json.load(open(GRADIENT_DIRECTIONS)); D, k = {}, {}
    for L in ODD:
        for kind in "VK":
            s = np.array(G["s" + kind][str(L)]); tot = G["grad_share"][kind][L] * G["grad_energy_total"]
            k[(L, kind)] = int(np.searchsorted(np.cumsum(s ** 2) / tot, 0.90) + 1)
            D[(L, kind)] = np.array(G["D" + kind][str(L)][:k[(L, kind)]], np.float64)
    return D, k, G


def wrapd(a): return (np.asarray(a, float) + 180) % 360 - 180


def ang_diff(a, b):
    """signed a - b in degrees, wrapped."""
    return wrapd(np.asarray(a, float) - np.asarray(b, float))


def scenes(seed, n):
    S = policy_tools.S; return [S.make_scene(np.random.default_rng([seed, i]), "full") for i in range(n)]


def readout(m, pm, kv, noise):
    """final angle per scene (degrees) after the NS-step flow: circular mean of the 50 rows of x*180."""
    x = noise.clone(); B = x.shape[0]; t = torch.tensor(1.0, device=DEVICE); dt = torch.tensor(-1.0 / NS, device=DEVICE)
    with torch.no_grad():
        for k in range(NS):
            v = m.denoise_step(pm, kv, x, t.expand(B)); x = x + dt * v; t = t + dt
    r = x[:, :, 0].float().cpu().numpy().astype(np.float64)
    a = np.radians(r * 180); return np.degrees(np.arctan2(np.sin(a).mean(1), np.cos(a).mean(1)))


def prefix(H, pol, scs, imgs=None):
    inp = policy_tools.make_inputs(pol, scs, imgs=imgs); return H.prefix(inp)


def clone_kv(kv): return {L: {k: v.clone() for k, v in d.items()} for L, d in kv.items()}


def rows(kv, L, kind):
    """(B, 64, 320) float view of the image rows of the layer-L cache."""
    t = kv[L][KEYN[kind]]; return t[:, :64].float().reshape(t.shape[0], 64, -1)


def set_rows(kv, L, kind, X):
    t = kv[L][KEYN[kind]]; sh = t.shape; t[:, :64] = X.reshape(sh[0], 64, *sh[2:]).to(t.dtype)
