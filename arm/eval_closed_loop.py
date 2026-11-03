"""Closed-loop evaluation harness for the SO-101 red-cube task (single overhead camera, 10 Hz policy).

    python arm/eval_closed_loop.py --policy smolvla --ckpt <checkpoint dir> --set fresh200 --conds plain \
        --camera arm/assets/camera.json --lighting shadow --out eval.json [--n-action-steps 10] [--hook my_hooks.py:install]
    python arm/eval_closed_loop.py --policy replay --dataset $TOYVLA_DATA/so101_redcube_angled_shadow --set replay30 --out replay.json
    python arm/eval_closed_loop.py --policy random --conds plain --set quick --out random.json

Protocol. The policy is queried at 10 Hz (every 5th 50 Hz sim step) with the overhead image (512 x 512) and
the joint state in dataset units; its action (absolute joint targets, dataset units) is executed over the next
5 sim steps by linear interpolation from the previous target (default ``--exec interp``; ``--exec hold`` holds
it). Replaying the dataset's own actions succeeds 100/100 with interp and 98/100 with hold. Episodes stop at the first
success or after ``--max-sec`` seconds (12 s). Success is the ``success`` flag of the so101-nexus pick-and-place
environment with the tray as target: cube released, static, resting, centre within goal_thresh = 4 cm (xy) of the tray centre.
Layouts: every condition of a set uses the SAME seed list (assets/eval_layouts.json), and a seed fixes the
cube position and arm start pose in every condition (scene.py).

Per episode: success, time to success, final cube-tray distance, phase reached (reach / grasp / lift /
place), the TCP position where the gripper first closed (what the arm went for) and its distance to the
cube and to each distractor. Per condition: success rate with a Wilson 95 % interval, median time to
success, mean final distance, phase-failure counts.

Hooks. ``--hook file.py:fn`` calls ``fn(wrapper, cond_name, spec)`` before each condition; it may register
torch hooks on ``wrapper.policy`` (the lerobot SmolVLAPolicy; ``wrapper.policy.model`` is VLAFlowMatching,
``.vlm_with_expert`` holds SigLIP (``.vlm.model.vision_model``), the VLM text layers and the expert) and may
append frame edits to ``wrapper.frame_hooks`` (callables frame -> frame on the raw observation dict, before
preprocessing). It may return a cleanup callable, run after the condition. From Python, use
``run_condition(wrapper, spec, seeds, ...)`` directly."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np

import paths
import scene as S
from so101_nexus.lerobot_dataset import dataset_row_to_sim_qpos, sim_qpos_to_dataset_row

HERE = Path(__file__).resolve().parent
ROBOT_TYPE = "so101_follower"


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# --------------------------------------------------------------------------- policies
# Interface: reset(seed, env); act(frame) -> 6-vector action in DATASET units (deg, gripper 0-100).
# frame = {"observation.state": float32[6], "observation.images.overhead": uint8[H,W,3],
#          "observation.images.wrist": uint8[h,w,3]}


class RandomPolicy:
    name = "random"

    def __init__(self, seed=0, step_deg=8.0):
        self.rng = np.random.default_rng(seed)
        self.step = step_deg
        self.frame_hooks = []

    def reset(self, seed, env):
        self.rng = np.random.default_rng([seed, 99])

    def act(self, frame):
        a = frame["observation.state"].astype(np.float64).copy()
        a[:5] += self.rng.normal(0, self.step, 5)
        a[5] = self.rng.uniform(0, 100)
        return a


class ReplayPolicy:
    """Replays the recorded 10 Hz actions of the dataset episode with the same layout seed (tests the data
    path, units, 10 Hz subsampling and action-hold execution end to end)."""
    name = "replay"

    def __init__(self, root):
        import pandas as pd

        root = Path(root).expanduser()
        side = [json.loads(l) for l in (root / "meta/so101_episodes.jsonl").read_text().splitlines() if l.strip()]
        self.ep_of_seed = {x["layout_seed"]: x["episode_index"] for x in side}
        df = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet"))])
        self.actions = {int(e): np.stack(g.sort_values("frame_index")["action"].to_numpy())
                        for e, g in df.groupby("episode_index")}
        self.frame_hooks = []

    def seeds(self, n):
        return sorted(self.ep_of_seed)[:n]

    def reset(self, seed, env):
        self.acts = self.actions[self.ep_of_seed[seed]]
        self.k = 0

    def act(self, frame):
        a = self.acts[min(self.k, len(self.acts) - 1)]
        self.k += 1
        return a.astype(np.float64)


class SmolVLAWrapper:
    """lerobot SmolVLAPolicy + its pre/post processors. ``self.policy`` is exposed for hooks."""
    name = "smolvla"

    def __init__(self, ckpt, device="cuda", n_action_steps=None, num_steps=None):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.factory import get_policy_class

        self.torch = torch
        cfg = PreTrainedConfig.from_pretrained(ckpt)
        cfg.pretrained_path = ckpt
        if n_action_steps is not None:
            cfg.n_action_steps = n_action_steps
        if num_steps is not None:
            cfg.num_steps = num_steps
        self.cfg = cfg
        self.policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg)
        self.policy.to(device).eval()
        self.pre, self.post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=ckpt,
                                                       preprocessor_overrides={"device_processor": {"device": device}})
        self.device = torch.device(device)
        self.image_keys = [k for k in cfg.input_features if k.startswith("observation.images.")]
        self.frame_hooks = []
        self.bf16 = True        # as in training (bf16 autocast)

    def reset(self, seed, env):
        self.policy.reset()
        self.pre.reset()
        self.post.reset()
        # per-episode flow-matching noise (reproducible): seed torch's generator per episode
        self.torch.manual_seed(int(seed) % (2 ** 31))

    def act(self, frame):
        from lerobot.policies.utils import prepare_observation_for_inference

        obs = {k: v for k, v in frame.items() if k == "observation.state" or k in self.image_keys}
        with self.torch.inference_mode(), self.torch.autocast("cuda", dtype=self.torch.bfloat16, enabled=self.bf16):
            batch = prepare_observation_for_inference(obs, self.device, S.TASK, ROBOT_TYPE)
            batch = self.pre(batch)
            a = self.policy.select_action(batch)
            a = self.post(a)
        return a.squeeze(0).float().cpu().numpy().astype(np.float64)


# --------------------------------------------------------------------------- rollout


def make_frame(obs, gl):
    return {
        "observation.state": sim_qpos_to_dataset_row(obs["state"].astype(np.float64), gripper_limits_rad=gl).astype(np.float32),
        "observation.images.overhead": obs["overhead_camera"],
        "observation.images.wrist": obs["wrist_camera"],
    }


def run_episode(env, pol, seed, max_sec=12.0, exec_mode="interp", stop_on_success=True, save_frames=False):
    gl = S.grip_limits(env)
    low, high = env.action_space.low, env.action_space.high
    obs, info = env.reset(seed=seed)
    pol.reset(seed, env)
    u = env
    cube0 = u._get_object_pose()[:3].copy()
    tray = u._get_target_pos()[:3].copy()
    max_steps = int(max_sec * S.SIM_HZ)
    prev_q = obs["state"].astype(np.float64).copy()
    t_success, grasped, lifted, min_tcp_cube = -1.0, False, False, 1e9
    first_close = None
    low_xy, reach_xy_min = None, 1e9          # reach accuracy: TCP xy when it first goes below LOW_Z, and the
    frames, traj = [], []                     # min horizontal TCP-cube0 distance while low (z < 6 cm) before a grasp
    t = 0
    while t < max_steps:
        fr = make_frame(obs, gl)
        for h in pol.frame_hooks:
            fr = h(fr)
        a = pol.act(fr)
        if save_frames:
            frames.append(fr["observation.images.overhead"])
        q = dataset_row_to_sim_qpos(np.asarray(a, np.float64), gripper_limits_rad=gl)
        q = np.clip(q, low, high)
        if first_close is None and a[5] < 30.0 and t > 0:
            first_close = u._get_tcp_pose()[:3].copy()
        done = False
        for i in range(S.RECORD_EVERY):
            qi = q if exec_mode == "hold" else prev_q + (q - prev_q) * (i + 1) / S.RECORD_EVERY
            obs, _r, _te, _tr, info = env.step(qi.astype(np.float32))
            t += 1
            tcp = u._get_tcp_pose()[:3]
            cube = u._get_object_pose()[:3]
            min_tcp_cube = min(min_tcp_cube, float(np.linalg.norm(tcp - cube)))
            if low_xy is None and tcp[2] < LOW_Z:
                low_xy = tcp[:2].copy()
            if not grasped and tcp[2] < 0.06:
                reach_xy_min = min(reach_xy_min, float(np.linalg.norm(tcp[:2] - cube0[:2])))
            grasped |= bool(info.get("is_grasped", 0.0))
            lifted |= float(info.get("lift_height", 0.0)) > 0.02
            if info.get("success", False) and t_success < 0:
                t_success = t / S.SIM_HZ
                if stop_on_success:
                    done = True
                    break
        traj.append(u._get_tcp_pose()[:3].round(4).tolist())
        prev_q = q
        if done:
            break
    cube = u._get_object_pose()[:3]
    success = t_success >= 0
    phase = None if success else ("reach" if min_tcp_cube > 0.02 else "grasp" if not grasped else
                                  "lift" if not lifted else "place")
    rec = dict(seed=int(seed), success=success, t_success=t_success, sim_sec=t / S.SIM_HZ,
               final_dist=float(np.linalg.norm(cube[:2] - tray[:2])), cube0_xy=cube0[:2].round(4).tolist(),
               min_tcp_cube=min_tcp_cube, grasped=grasped, lifted=lifted, fail_phase=phase,
               start_qpos=env.extra_info.get("start_qpos"))
    if first_close is not None:
        fc = first_close[:2]
        rec["first_close_xy"] = fc.round(4).tolist()
        rec["first_close_to_cube"] = float(np.linalg.norm(fc - cube0[:2]))
        ds = [float(np.linalg.norm(fc - np.array(d["xy"]))) for d in env.extra_info.get("distractors", []) if d.get("placed")]
        if ds:
            rec["first_close_to_distractors"] = ds
    rec["reach_xy_min"] = reach_xy_min if reach_xy_min < 1e8 else None
    if low_xy is not None:
        e = low_xy - cube0[:2]
        rec["low_xy"] = low_xy.round(4).tolist()
        rec["reach_xy_low"] = float(np.linalg.norm(e))
        objs = {f"distr{i}": np.array(d["xy"]) for i, d in enumerate(env.extra_info.get("distractors", [])) if d.get("placed")}
        objs.update(landmarks(env.spec, low_xy))
        near = {}
        for k, o in objs.items():
            v = o - cube0[:2]
            near[k] = dict(dist_low=float(np.linalg.norm(low_xy - o)), dist_cube=float(np.linalg.norm(v)),
                           cos_err=float(e @ v / (np.linalg.norm(e) * np.linalg.norm(v) + 1e-9)))
        if near:
            rec["objects"] = near
    if env.extra_info.get("distractors"):
        rec["distractors"] = env.extra_info["distractors"]
    rec["tcp_traj_10hz"] = traj
    return rec, frames


LOW_Z = 0.04    # TCP height (m) that marks the start of the grasp descent (expert: hover 0.093, grasp 0.0125)


def landmarks(spec, low_xy):
    """World xy of the toy structured-table elements present in spec.table (yellow / green centres, the tape point
    nearest to low_xy)."""
    t = spec.table
    if not t.startswith("struct"):
        return {}
    parts = S.TOY_PARTS if t == "struct" else tuple(t.split(":", 1)[1].split("+"))
    T, out = S.TOY, {}
    bu, bv = T["base_uv"]
    if "yellow" in parts:
        x0, y0, x1, y1 = T["yellow"]
        out["yellow"] = np.array(S.pixel_to_world(spec, bu + (x0 + x1) / 2, bv - (y0 + y1) / 2))
    if "green" in parts:
        gx, gy = T["R"] * np.cos(np.radians(-135)), T["R"] * np.sin(np.radians(-135))
        out["green"] = np.array(S.pixel_to_world(spec, bu + gx, bv - gy))
    if "tape" in parts:
        xw, _ = S.pixel_to_world(spec, 0.0, bv - T["tape_y"])
        out["tape"] = np.array([float(xw), float(low_xy[1])])
    return out


def summarise(recs):
    n = len(recs)
    k = sum(r["success"] for r in recs)
    ts = [r["t_success"] for r in recs if r["success"]]
    lo, hi = wilson(k, n)
    phases = {p: sum(r["fail_phase"] == p for r in recs) for p in ("reach", "grasp", "lift", "place")}
    return dict(n=n, success=k, rate=k / n if n else float("nan"), wilson95=[lo, hi],
                t_success_median=float(np.median(ts)) if ts else None,
                final_dist_mean=float(np.mean([r["final_dist"] for r in recs])),
                **_reach_summary(recs),
                final_dist_median=float(np.median([r["final_dist"] for r in recs])), fail_phase=phases)


def _reach_summary(recs):
    lo = [r["reach_xy_low"] for r in recs if r.get("reach_xy_low") is not None]
    mn = [r["reach_xy_min"] for r in recs if r.get("reach_xy_min") is not None]
    return dict(reach_low_n=len(lo), reach_low_median=float(np.median(lo)) if lo else None,
                reach_low_mean=float(np.mean(lo)) if lo else None,
                reach_low_within_1cm=float(np.mean(np.array(lo) <= 0.01)) if lo else None,
                reach_low_within_2cm=float(np.mean(np.array(lo) <= 0.02)) if lo else None,
                reach_min_median=float(np.median(mn)) if mn else None)


def run_condition(pol, spec, seeds, name="", log=print, gif_dir=None, gif_n=0, **kw):
    env = S.make_env(spec)
    recs, t0 = [], time.time()
    for i, sd in enumerate(seeds):
        rec, frames = run_episode(env, pol, sd, save_frames=bool(gif_dir) and i < gif_n, **kw)
        recs.append(rec)
        if frames:
            from PIL import Image
            Path(gif_dir).mkdir(parents=True, exist_ok=True)
            ims = [Image.fromarray(f[::2, ::2]) for f in frames]
            ims[0].save(Path(gif_dir) / f"{name}_{sd}_{'ok' if rec['success'] else 'fail'}.gif", save_all=True,
                        append_images=ims[1:], duration=100, loop=0)
        log(f"  [{name}] {i + 1}/{len(seeds)} seed {sd} {'OK  ' if rec['success'] else 'FAIL'} "
            f"t={rec['t_success']:.1f}s dist={rec['final_dist']:.3f} phase={rec['fail_phase']}")
    env.close()
    s = summarise(recs)
    s["sec"] = time.time() - t0
    return s, recs


def load_layouts(path=None, camera=None, lighting=None):
    """Frozen condition specs + seed sets. path=None: assets/eval_layouts.json next to this file.
    camera: path of an assets/camera.json-style file; its camera is put into EVERY condition's spec
    (overhead_cam), so all conditions are rendered from that camera (layout seeds unchanged)."""
    L = json.loads(Path(path or HERE / "assets/eval_layouts.json").read_text())
    if camera:
        cam = S.load_camera(camera)
        for c in L["conditions"].values():
            c["overhead_cam"] = cam
        L["camera"] = cam
    if lighting:        # training lighting of the model (e.g. shadow), applied to every condition
        for c in L["conditions"].values():
            c["lighting"] = lighting
        L["lighting"] = lighting
    return L


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["smolvla", "replay", "random"], required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--dataset", default=str(paths.DATASET))
    ap.add_argument("--conds", default="", help="comma list; default: the set's conditions")
    ap.add_argument("--set", default="quick", help="seed set in assets/eval_layouts.json (quick / main / dev200 / fresh200 / big3k_a ...)")
    ap.add_argument("--n", type=int, default=0, help="use only the first n seeds")
    ap.add_argument("--n-action-steps", type=int, default=10, help="actions executed per chunk before re-planning (training evals use 10; 0 = checkpoint config, 50)")
    ap.add_argument("--num-steps", type=int, default=None, help="flow-matching steps (default: checkpoint's)")
    ap.add_argument("--exec", dest="exec_mode", default="interp", choices=["hold", "interp"])
    ap.add_argument("--max-sec", type=float, default=12.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hook", default="")
    ap.add_argument("--gif-dir", default="")
    ap.add_argument("--gif-n", type=int, default=0)
    ap.add_argument("--layouts", default="", help="layouts JSON (default assets/eval_layouts.json)")
    ap.add_argument("--camera", default="", help="camera JSON (e.g. assets/camera.json) applied to every condition")
    ap.add_argument("--lighting", default="", help="lighting preset applied to every condition (the model's training lighting)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    L = load_layouts(args.layouts or None, args.camera or None, args.lighting or None)
    sset = L["sets"][args.set]
    conds = args.conds.split(",") if args.conds else sset["conditions"]
    if args.policy == "smolvla":
        pol = SmolVLAWrapper(args.ckpt, args.device, args.n_action_steps or None, args.num_steps)
    elif args.policy == "replay":
        pol = ReplayPolicy(args.dataset)
    else:
        pol = RandomPolicy()
    seeds = sset["seeds"] if args.policy != "replay" else pol.seeds(len(sset["seeds"]))
    if args.n:
        seeds = seeds[: args.n]
    hook = None
    if args.hook:
        f, fn = args.hook.rsplit(":", 1)
        spec_ = importlib.util.spec_from_file_location("so101_user_hook", f)
        mod = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(mod)
        hook = getattr(mod, fn)
    out = dict(policy=args.policy, ckpt=args.ckpt, camera=L.get("camera"), lighting=L.get("lighting"), set=args.set, seeds=seeds, exec=args.exec_mode,
               n_action_steps=args.n_action_steps, max_sec=args.max_sec, hook=args.hook, results={})
    for c in conds:
        spec = S.SceneSpec.from_dict(L["conditions"][c])
        cleanup = hook(pol, c, spec) if hook else None
        s, recs = run_condition(pol, spec, seeds, name=c, exec_mode=args.exec_mode, max_sec=args.max_sec,
                                gif_dir=args.gif_dir or None, gif_n=args.gif_n)
        if callable(cleanup):
            cleanup()
        out["results"][c] = dict(summary=s, spec=L["conditions"][c], episodes=recs)
        print(f"== {c}: {s['success']}/{s['n']} = {s['rate']:.3f} [{s['wilson95'][0]:.2f}, {s['wilson95'][1]:.2f}] "
              f"t_med={s['t_success_median']} dist_mean={s['final_dist_mean']:.3f} reach_low_med={s['reach_low_median']} "
              f"(n {s['reach_low_n']}, <=1cm {s['reach_low_within_1cm']}, <=2cm {s['reach_low_within_2cm']}) "
              f"fail={s['fail_phase']} ({s['sec']:.0f}s)", flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1)
    if args.policy == "smolvla":
        import torch
        print(f"GPU peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, reserved "
              f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
