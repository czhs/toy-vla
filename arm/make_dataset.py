"""Record successful scripted-expert episodes as a local LeRobot (v3) dataset.

Keys and units follow so101_follower: observation.state / action = 6 joints (body joints in degrees, gripper 0-100 %
of travel); action = the expert's joint TARGET sent at that step (the next commanded joint positions); images
observation.images.overhead (512x512) and observation.images.wrist (320x240), video; recorded at 10 Hz (every 5th
50 Hz sim step); one task string.

The dataset of the paper re-runs the layout seeds in assets/dataset_episodes.jsonl in 10 shards (see the README):
    python arm/make_dataset.py --camera arm/assets/camera.json --lighting shadow --seed-list arm/assets/dataset_episodes.jsonl \
        --shard K --nshards 10 --root $TOYVLA_DATA/so101_shards/shard_0K --repo-id local/shard_0K

Those seeds came from attempt k = layout seed 1000000 + 10000 * shard + k (4 shards, disjoint from the eval seed lists, which
are < 1,000,000). Failed expert episodes are dropped (and listed). A sidecar <root>/meta/so101_episodes.jsonl records, per kept
episode, the layout seed, cube and disc xy, and the expert result."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import scene as S
import expert as E
from so101_nexus.config import SO101_JOINT_NAMES
from so101_nexus.lerobot_dataset import sim_qpos_to_dataset_row


def features(spec) -> dict:
    names = [f"{j}.pos" for j in SO101_JOINT_NAMES]
    hwc = ["height", "width", "channels"]
    return {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": names},
        "action": {"dtype": "float32", "shape": (6,), "names": names},
        "observation.images.overhead": {"dtype": "video", "shape": (spec.ov_h, spec.ov_w, 3), "names": hwc},
        "observation.images.wrist": {"dtype": "video", "shape": (spec.wr_h, spec.wr_w, 3), "names": hwc},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="plain")
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--seed0", type=int, default=1_000_000)
    ap.add_argument("--attempt0", type=int, default=0)
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo-id", default="local/so101_redcube_angled_shadow")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--camera", default="", help="camera JSON (assets/camera.json) for observation.images.overhead")
    ap.add_argument("--lighting", default="", help="lighting preset override (e.g. shadow)")
    ap.add_argument("--seed-list", default="", help="meta/so101_episodes.jsonl of a dataset: re-run exactly its layout seeds")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    import dataclasses
    spec = S.conditions()[args.cond]
    if args.camera:
        spec = dataclasses.replace(spec, overhead_cam=S.load_camera(args.camera))
    if args.lighting:
        spec = dataclasses.replace(spec, lighting=args.lighting)
    seed_list = None
    if args.seed_list:
        rows = [json.loads(l) for l in Path(args.seed_list).expanduser().read_text().splitlines() if l.strip()]
        allseeds = [r["layout_seed"] for r in sorted(rows, key=lambda r: r["episode_index"])]
        per = -(-len(allseeds) // args.nshards)
        seed_list = allseeds[args.shard * per:(args.shard + 1) * per]
        args.episodes = len(seed_list)
    env = S.make_env(spec)
    assert abs(env.control_dt - 1.0 / S.SIM_HZ) < 1e-9, env.control_dt
    gl = S.grip_limits(env)
    root = Path(args.root).expanduser()
    if root.exists():
        if not args.overwrite:
            print(f"{root} exists; pass --overwrite", file=sys.stderr)
            return 2
        shutil.rmtree(root)
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=S.FPS, features=features(spec),
                               root=root, robot_type="so101_follower", use_videos=True, image_writer_threads=4)
    side = []
    fails = []
    kept, attempt, t0 = 0, 0, time.time()
    while kept < args.episodes and attempt < 2 * args.episodes:
        if seed_list is not None:
            if attempt >= len(seed_list):
                break
            seed = seed_list[attempt]
        else:
            seed = args.seed0 + args.attempt0 + attempt
        attempt += 1

        steps = []   # (obs or None, action) per 50 Hz sim step; obs kept only on recorded steps

        def on_frame(obs, action, info, _st=steps):
            keep = len(_st) % S.RECORD_EVERY == 0
            _st.append((obs if keep else None, np.asarray(action, np.float64).copy()))

        r = E.run_expert_episode(env, seed, on_frame=on_frame)
        if r["success"]:
            T = len(steps)
            # frame k = sim step 5k (10 Hz): state and images at step 5k; action = the expert's joint target at
            # step min(5k + 4, T - 1), i.e. the commanded joint position at the end of the 0.1 s window
            for k in range((T + S.RECORD_EVERY - 1) // S.RECORD_EVERY):
                obs, _a = steps[k * S.RECORD_EVERY]
                act = steps[min(k * S.RECORD_EVERY + S.RECORD_EVERY - 1, T - 1)][1]
                ds.add_frame({
                    "observation.state": sim_qpos_to_dataset_row(obs["state"].astype(np.float64), gripper_limits_rad=gl).astype(np.float32),
                    "action": sim_qpos_to_dataset_row(act, gripper_limits_rad=gl).astype(np.float32),
                    "observation.images.overhead": obs["overhead_camera"],
                    "observation.images.wrist": obs["wrist_camera"],
                    "task": S.TASK,
                })
            ds.save_episode(parallel_encoding=False)
            side.append(dict(episode_index=kept, layout_seed=seed, cube_xy=r["cube_xy"],
                             start_qpos=env.extra_info.get("start_qpos"), frames=(T + S.RECORD_EVERY - 1) // S.RECORD_EVERY,
                             **{k: r[k] for k in ("steps", "final_dist", "first_success")}))
            kept += 1
        else:
            fails.append(r)
        if attempt % 25 == 0:
            print(f"attempt {attempt} kept {kept} fails {len(fails)} ({time.time() - t0:.0f}s)", flush=True)
    ds.finalize()
    (root / "meta" / "so101_episodes.jsonl").write_text("\n".join(json.dumps(x) for x in side) + "\n")
    (root / "meta" / "so101_generation.json").write_text(json.dumps(dict(
        cond=args.cond, spec=spec.to_dict(), seed0=args.seed0, attempt0=args.attempt0, seed_list=args.seed_list,
        shard=args.shard, nshards=args.nshards, attempts=attempt, kept=kept, failed=fails,
        task=S.TASK, fps=S.FPS, sim_hz=S.SIM_HZ, record_every=S.RECORD_EVERY,
        action="expert joint target at the end of each 0.1 s window (sim step 5k+4): the next commanded joint positions",
        state="measured joint positions at sim step 5k", units="body joints deg, gripper 0-100 % (so101_follower)",
        seconds=time.time() - t0), indent=1))
    print(f"DONE kept {kept}/{attempt} frames {ds.num_frames} episodes {ds.num_episodes} -> {root} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
