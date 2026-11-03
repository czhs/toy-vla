"""Rebuild a full checkpoint from base weights + a delta file (only the tensors that differ).
  make:  python arm/apply_delta.py make  <ckpt_dir> <base_dir> <delta.safetensors>
  apply: python arm/apply_delta.py apply <out_dir (holds config/processor files)> <base_dir> <delta.safetensors>
The arm policy: base_dir = the snapshot of lerobot/smolvla_base, delta = delta.safetensors in the unpacked arm_policy.tar. A folded policy: base_dir = the arm policy,
delta = fold_q_proj_{plus60,minus60}.safetensors (the six q_proj tensors)."""
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

mode, a, base, delta = sys.argv[1:5]
fb = safe_open(str(Path(base) / "model.safetensors"), "pt")
if mode == "make":
    fc = safe_open(str(Path(a) / "model.safetensors"), "pt")
    d = {}
    for k in fc.keys():
        t = fc.get_tensor(k)
        if k not in fb.keys():
            d[k] = t
            continue
        u = fb.get_tensor(k)
        if t.shape != u.shape or t.dtype != u.dtype or not torch.equal(t, u):
            d[k] = t
    save_file(d, delta)
    print("delta tensors", len(d))
else:
    fd = safe_open(delta, "pt")
    full = {k: fb.get_tensor(k) for k in fb.keys()}
    for k in fd.keys():
        full[k] = fd.get_tensor(k)
    save_file(full, str(Path(a) / "model.safetensors"), metadata={"format": "pt"})
    print("wrote", Path(a) / "model.safetensors", len(full), "tensors,", len(list(fd.keys())), "from delta")
