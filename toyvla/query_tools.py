"""Flow helpers with the hook phase and step set. Expert layout at odd layers: q (B, 50, 15, 64) = RoPE(q_proj(input_layernorm(x)), action
positions 0..49); k, v (B, 113, 5, 64) = the expert's own k_proj / v_proj applied to the cached (post-RoPE) key / value rows of the same layer;
query head h reads key/value head h // 3."""
import numpy as np, torch
from toyvla import stream_tools as V, readouts as R
from toyvla.paths import DEVICE
NS, NI, ODD = 10, 64, V.ODD
log, peak = V.log, V.peak


def step(m, H, pm, kv, x, t, k):
    H.phase, H.step = "denoise", k
    try: return m.denoise_step(pm, kv, x, t.expand(x.shape[0]))
    finally: H.phase = None; H.step = -1


def flow_eval(m, H, pm, kv, noise):
    """fp32 x, no autocast: final answer (numpy) and y_k per step."""
    x = noise.clone(); B = x.shape[0]; t = torch.tensor(1.0, device=DEVICE); dt = torch.tensor(-1.0 / NS, device=DEVICE); ys = []
    with torch.no_grad():
        for k in range(NS):
            v = step(m, H, pm, kv, x, t, k); ys.append(R.y_of((x - t * v)[:, :, 0]).float().cpu().numpy())
            x = x + dt * v; t = t + dt
    return V.y_eval_np(x), np.stack(ys)
