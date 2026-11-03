"""Merge LeRobot dataset shards written by make_dataset.py into one dataset.

    python arm/merge_shards.py --out $TOYVLA_DATA/so101_redcube_angled_shadow $TOYVLA_DATA/so101_shards/shard_00 ...
Also concatenates the per-shard sidecars (meta/so101_episodes.jsonl, episode indices offset)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-id", default="local/so101_redcube_angled_shadow")
    ap.add_argument("shards", nargs="+")
    a = ap.parse_args()
    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    roots = [Path(s).expanduser() for s in a.shards]
    out = Path(a.out).expanduser()
    aggregate_datasets(repo_ids=[f"local/{r.name}" for r in roots], aggr_repo_id=a.repo_id, roots=roots, aggr_root=out)
    side, gens, off = [], [], 0
    for r in roots:
        rows = [json.loads(l) for l in (r / "meta/so101_episodes.jsonl").read_text().splitlines() if l.strip()]
        for x in rows:
            x["shard"] = r.name
            x["episode_index"] += off
        off += len(rows)
        side += rows
        gens.append(json.loads((r / "meta/so101_generation.json").read_text()))
    (out / "meta/so101_episodes.jsonl").write_text("\n".join(json.dumps(x) for x in side) + "\n")
    (out / "meta/so101_generation.json").write_text(json.dumps(dict(shards=gens, total_kept=off), indent=1))
    ds = LeRobotDataset(a.repo_id, root=out)
    assert ds.num_episodes == off, (ds.num_episodes, off)
    print(f"MERGED {ds.num_episodes} episodes, {ds.num_frames} frames -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
