"""Reduce closed-loop evaluation outputs (eval_closed_loop.py JSON, plain condition) to the file the paper's numbers and Figure 3b are computed from.
    python arm/reduce_eval_outputs.py [--eval-dir $TOYVLA_DATA/arm_eval] [--out results/arm/closed_loop.npz]
Input names in eval-dir (written by run_closed_loop.sh) and the keys they become: KEY_seed, KEY_success, KEY_cube_xy per episode, in seed order. big_* are the
3,000 layouts of Figure 3b, dev_* the 200 development layouts (dev_fold is lam 1.25, the chosen scale; lam 0.75 and 1.5 are the other candidates at this
frame set), fresh_* the 200 confirmation layouts. All folds in dev_* and fresh_* have A = +60."""
import argparse, json
from pathlib import Path
import numpy as np
import paths

RUNS = {"big_unedited": [f"big_unedited_{s}.json" for s in "abc"], "big_plus60": [f"big_plus60_{s}.json" for s in "abc"], "big_minus60": [f"big_minus60_{s}.json" for s in "abc"],
        "dev_unedited": ["dev_unedited.json"], "dev_fold": ["dev_fold_lam1.25.json"], "dev_fold_lam0.75": ["dev_fold_lam0.75.json"], "dev_fold_lam1.5": ["dev_fold_lam1.5.json"],
        "fresh_unedited": ["fresh_unedited.json"], "fresh_fold": ["fresh_fold_lam1.25.json"]}


def reduce_files(files):
    seed, ok, cube = [], [], []
    for f in files:
        for e in json.load(open(f))["results"]["plain"]["episodes"]: seed.append(e["seed"]); ok.append(e["success"]); cube.append(e["cube0_xy"])
    return np.array(seed, np.int32), np.array(ok, bool), np.array(cube, np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--eval-dir", default=str(paths.EVAL)); ap.add_argument("--out", default=str(paths.RESULTS / "closed_loop.npz")); a = ap.parse_args()
    out = {}
    for key, names in RUNS.items():
        files = [Path(a.eval_dir) / n for n in names]
        if not all(f.exists() for f in files): print("missing", key); continue
        out[f"{key}_seed"], out[f"{key}_success"], out[f"{key}_cube_xy"] = reduce_files(files); print(key, len(out[f"{key}_seed"]), "episodes, success", round(float(out[f"{key}_success"].mean()), 3))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(a.out, **out); print("wrote", a.out)
