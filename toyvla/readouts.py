"""Flow readouts used by every experiment: the circular mean of an action chunk's first coordinate and per-scene flow noise."""
import math, numpy as np, torch
from toyvla.paths import DEVICE


def y_of(xc0):
    """xc0 (B, 50) model coordinate 0 -> (B,) degrees: circular mean over the 50 rows."""
    r = xc0 * math.pi
    return torch.rad2deg(torch.atan2(torch.sin(r).mean(1), torch.cos(r).mean(1)))


def noise_per_scene(seeds):
    """one (50, 32) draw per scene with the given seeds."""
    return torch.stack([torch.randn(50, 32, device=DEVICE, generator=torch.Generator(device=DEVICE).manual_seed(int(s))) for s in seeds])
