"""Where data lives and which device runs the model. Everything is relative to the repository unless overridden:
    TOYVLA_DATA      checkpoints, feature caches, large intermediate arrays (default <repo>/data; not tracked by git)
    TOYVLA_RESULTS   per-scene outputs of the experiments, read by the figure and number scripts (default <repo>/results)
    TOYVLA_DEVICE    torch device (default cuda when available, else cpu; flow-noise draws differ between the two generators)
    TOYVLA_SMOKE     set to 1 for a very short run of any experiment script"""
import os
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('TOYVLA_DATA', REPO / 'data'))
RESULTS = Path(os.environ.get('TOYVLA_RESULTS', REPO / 'results'))
CKPT = DATA / 'checkpoints'
FEATURE_CACHE = DATA / 'feature_cache'
POLICY = {'red': CKPT / 'toy_policy_red.pt', 'green': CKPT / 'toy_policy_green.pt', 'blue': CKPT / 'toy_policy_blue.pt'}
GRADIENT_DIRECTIONS = RESULTS / 'gradient_directions.json'
VLM_WEIGHTS = DATA / 'vlm_projection_weights.npz'     # written by experiments/gradient_directions.py
BLOCK_CALIBRATION = RESULTS / 'block_calibration.json'
COLOUR_CALIBRATION = RESULTS / 'colour_calibration.npz'
ESTIMATE_FIT = DATA / 'estimate_fit.npz'              # written by experiments/query_replacement.py
DEVICE = os.environ.get('TOYVLA_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
SMOKE = bool(os.environ.get('TOYVLA_SMOKE'))           # a few scenes and conditions, to check that a script runs from a clean clone; its numbers mean nothing


def peak_gpu_bytes(): return torch.cuda.max_memory_allocated() if DEVICE.startswith('cuda') else 0
def reset_peak_gpu():
    if DEVICE.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
def empty_gpu_cache():
    if DEVICE.startswith('cuda'): torch.cuda.empty_cache()
