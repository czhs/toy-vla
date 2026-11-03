"""Scripted expert on the toy-matched scene, and its success rate over random layouts.

The expert is the repo's 8-state DLS-IK waypoint FSM (``collect_sim.Expert``: hover, descend, close,
lift, carry, lower, open, retreat), unchanged except for one fix needed by the random arm start pose:
the stage timeout (60 control steps, after which the repo's FSM advances even if the waypoint was not
reached) is raised for moving stages to  60 + distance / MAX_STEP_M  steps, so a far start does not skip
the hover waypoint.

    python expert.py --cond plain --n 200 --seed0 100000 --out expert_plain.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import scene as S
import collect_sim as cs  # the repo's expert (scene puts the repo on sys.path)

MAX_STEPS = 900


class Expert(cs.Expert):
    """Repo expert with distance-scaled stage timeouts."""

    def act(self):
        if self.stage < len(self.plan) and self.stage_steps == 0:
            _n, target, _g, _h = self.plan[self.stage]
            dist = 0.0 if target is None else float(np.linalg.norm(target - self.tcp_goal))
            cs.STAGE_TIMEOUT = 60 + int(dist / cs.MAX_STEP_M)
        return super().act()


def run_expert_episode(env, seed: int, max_steps: int = MAX_STEPS, on_frame=None):
    """One expert episode at the sim rate (50 Hz). ``on_frame(obs, action_rad, info)`` is called before
    each env.step. Waypoint noise uses rng([seed, 1]), so an episode is a pure function of its seed."""
    rng = np.random.default_rng([seed, 1])
    obs, info = env.reset(seed=seed)
    expert = Expert(env, rng)
    expert.build_plan()
    grasped_ever, first_success, t = False, -1, 0
    for t in range(max_steps):
        action = expert.act()
        if on_frame is not None:
            on_frame(obs, action, info)
        obs, _r, _te, _tr, info = env.step(action)
        grasped_ever |= bool(info.get("is_grasped", 0.0))
        if info.get("success", False) and first_success < 0:
            first_success = t + 1
        if expert.done():
            break
    return dict(seed=int(seed), success=bool(info.get("success", False)), steps=t + 1,
                final_dist=float(info["obj_to_target_dist"]), grasped_ever=grasped_ever,
                first_success=first_success, stage=expert.stage_name, cube_xy=env.extra_info.get("cube_xy"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="plain")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=100000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    env = S.make_env(S.conditions()[args.cond])
    res, t0 = [], time.time()
    for i in range(args.n):
        r = run_expert_episode(env, args.seed0 + i)
        res.append(r)
        if not r["success"]:
            print(f"FAIL seed {r['seed']} stage {r['stage']} dist {r['final_dist']:.3f} grasped {r['grasped_ever']} "
                  f"cube {r['cube_xy']} steps {r['steps']}", flush=True)
        if (i + 1) % 20 == 0:
            print(f"{i + 1}: {sum(x['success'] for x in res)}/{i + 1} ({time.time() - t0:.0f}s)", flush=True)
    k = sum(x["success"] for x in res)
    k25 = sum(x["success"] and x["final_dist"] <= 0.025 for x in res)
    summ = dict(cond=args.cond, n=args.n, seed0=args.seed0, success=k, rate=k / args.n, success_within_2p5cm=k25,
                mean_steps=float(np.mean([x["steps"] for x in res])), max_steps=int(max(x["steps"] for x in res)),
                sec=time.time() - t0, episodes=res)
    print(f"EXPERT {args.cond}: {k}/{args.n} = {k / args.n:.3f}; within 2.5 cm {k25}; mean steps "
          f"{summ['mean_steps']:.0f} (max {summ['max_steps']}); {summ['sec']:.0f}s")
    if args.out:
        json.dump(summ, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
