"""Build the frozen SigLIP+connector feature cache of a LeRobot dataset for train_policy.py.

The features are connector(SigLIP(image)) of lerobot/smolvla_base (frozen in expert-only training), computed under bf16
autocast. One cache per dataset + image key; valid for every expert-only run that starts from smolvla_base.

    python arm/build_feature_cache.py --dataset $TOYVLA_DATA/so101_redcube_angled_shadow --image-key observation.images.overhead --workers 14
    python arm/build_feature_cache.py --dataset ... --blocks 0:10                       # part (resumable, 4096-frame blocks)
    python arm/build_feature_cache.py --dataset ... --status

Output: $TOYVLA_DATA/feature_cache/<dataset>__overhead/{features.u16 [N,64,960] bf16 bits, meta.json, done.npy}
(13.6 GB for 110,479 frames). Time: about 7 min on an RTX 4090, about 30 min on an RTX 3060."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import fast_core as FC  # noqa: E402

BLOCK = 4096


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--repo-id", default="local/so101_redcube_angled_shadow")
    ap.add_argument("--image-key", default="observation.images.overhead")
    ap.add_argument("--out", default="", help="cache dir (default $TOYVLA_DATA/feature_cache/<dataset>__overhead)")
    ap.add_argument("--blocks", default="", help="a:b = only blocks a..b-1 of 4096 frames (for short jobs)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2, help="decode workers (~1.1 GB host RAM each); 2 keeps a small GPU GPU-bound")
    ap.add_argument("--base-dir", default="", help="overhead-only smolvla_base config dir (default: a private copy)")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    root = Path(args.dataset).expanduser()
    out = Path(args.out).expanduser() if args.out else FC.cache_dir_for(root, args.image_key)
    info = json.loads((root / "meta/info.json").read_text())
    N = info["total_frames"]
    nb = -(-N // BLOCK)
    if args.status:
        d = np.load(out / "done.npy") if (out / "done.npy").exists() else np.zeros(nb, bool)
        print(f"{out}: {int(d.sum())}/{nb} blocks done")
        return 0

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies import make_policy
    cam = args.image_key.split(".")[-1]
    base = FC.make_base_config(Path(args.base_dir).expanduser() if args.base_dir else out.parent / f"_smolvla_base_{cam}",
                               FC.dataset_cameras(root, [args.image_key]))
    cfg = PreTrainedConfig.from_pretrained(base)
    cfg.pretrained_path = str(base)
    cfg.device = "cuda"
    meta = LeRobotDatasetMetadata(args.repo_id, root=root)
    ds = LeRobotDataset(args.repo_id, root=root, delta_timestamps={args.image_key: [0.0]})
    from lerobot.datasets import dataset_reader as _dr
    _orig = _dr.DatasetReader._get_query_timestamps

    def _qts(self, *a, **k):
        return {kk: v for kk, v in _orig(self, *a, **k).items() if kk == args.image_key}
    _dr.DatasetReader._get_query_timestamps = _qts
    policy = make_policy(cfg=cfg, ds_meta=meta).to("cuda").eval()

    out.mkdir(parents=True, exist_ok=True)
    fp = FC.dataset_fingerprint(root)
    mpath = out / "meta.json"
    if mpath.exists():
        m0 = json.loads(mpath.read_text())
        assert m0["dataset_fingerprint"] == fp, "existing cache belongs to another dataset version; delete it first"
    else:
        mpath.write_text(json.dumps(dict(dataset=str(root), image_key=args.image_key, n_frames=N, tokens=FC.FEAT_TOKENS,
                                         dim=FC.FEAT_DIM, dtype="bfloat16 (stored as uint16 bits)", block=BLOCK,
                                         model="lerobot/smolvla_base vision_model+connector (frozen)",
                                         what="vlm_with_expert.embed_image(prepare_images(batch)[0]) under bf16 autocast",
                                         dataset_fingerprint=fp, created=time.strftime("%Y-%m-%dT%H:%M:%S")), indent=1))
    fpath = out / "features.u16"
    if not fpath.exists():
        np.memmap(fpath, dtype=np.uint16, mode="w+", shape=(N, FC.FEAT_TOKENS, FC.FEAT_DIM)).flush()
    fh = open(fpath, "r+b")            # plain writes (not a writable memmap): keeps host RSS low
    row_bytes = FC.FEAT_TOKENS * FC.FEAT_DIM * 2
    dpath = out / "done.npy"
    done = np.load(dpath) if dpath.exists() else np.zeros(nb, bool)
    a, b = (int(x) for x in args.blocks.split(":")) if args.blocks else (0, nb)
    todo = [k for k in range(a, min(b, nb)) if not done[k]]
    from torch.utils.data import DataLoader, Subset
    t0 = time.time()
    for k in todo:
        lo, hi = k * BLOCK, min(N, (k + 1) * BLOCK)
        dl = DataLoader(Subset(ds, range(lo, hi)), batch_size=args.batch, shuffle=False, num_workers=args.workers,
                        pin_memory=True)
        pos = lo
        for bt in dl:
            img = bt[args.image_key].cuda(non_blocking=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                f = FC.image_features(policy, {args.image_key: img})[0]
            assert f.dtype == torch.bfloat16 and f.shape[1:] == (FC.FEAT_TOKENS, FC.FEAT_DIM)
            fh.seek(pos * row_bytes)
            fh.write(f.cpu().view(torch.int16).numpy().view(np.uint16).tobytes())
            pos += len(f)
        assert pos == hi
        fh.flush()
        import os
        os.fsync(fh.fileno())
        done[k] = True
        np.save(dpath, done)
        el = time.time() - t0
        print(f"block {k} ({lo}-{hi}) done; {int(done.sum())}/{nb}; {el:.0f}s, {(hi - todo[0] * BLOCK) / el:.0f} frames/s",
              flush=True)
    print(f"CACHE {out}: {int(done.sum())}/{nb} blocks", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
