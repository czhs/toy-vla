"""Fast expert-only SmolVLA fine-tune on the SO-101 dataset.

Warm start lerobot/smolvla_base, SigLIP / connector / VLM text frozen, action expert + state/action projections
trained, AdamW (0.9, 0.95) eps 1e-8 wd 1e-10, clip 10, bf16 autocast, batch 32, lr 1e-4 with 1k warmup, constant,
cosine cooldown to 0.025x; checkpoint + closed-loop eval of every --eval_conds condition every --ckpt-every steps.

  * image features come from the per-dataset cache built by build_feature_cache.py (no video decoding, no SigLIP);
  * the frozen [image, lang] prefix runs without grad and without language padding; only the state token and the
    action expert build a graph (fast_core.split_forward);
  * state / action chunks are gathered on the GPU from the parquet tables (no DataLoader workers);
  * fused AdamW; torch.compile of the prefix pass and the grad pass (--no-compile to disable);
  * the in-training closed-loop eval renders only the policy's frames and skips the unused wrist; batching over a pool
    of envs (--eval-pool 10) is optional and not outcome-equivalent.

The policy of the paper (see the README):
    python arm/train_policy.py --run arm_policy --dataset $TOYVLA_DATA/so101_redcube_angled_shadow --camera arm/assets/camera.json \
        --lighting shadow --const-steps 40000 --cooldown 10000 --fixed-schedule --eval_conds plain,struct,struct_yellow,d_several
Smoke:  python arm/train_policy.py --run smoke --smoke 300 --no-eval"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
import paths  # noqa: E402
RUNS = paths.RUNS


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class Prefetch:
    """Background thread: index batches -> pinned bf16 feature tensors per camera (memmap / RAM gather)."""

    def __init__(self, caches, index_batches, depth=6):
        self.q = queue.Queue(depth)
        self.caches, self.it = caches, index_batches
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        for idx in self.it:
            self.q.put((idx, [c.gather(idx).pin_memory() for c in self.caches]))

    def __next__(self):
        return self.q.get()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", default=str(paths.DATASET))
    ap.add_argument("--repo-id", default="local/so101_redcube_angled_shadow")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--const-steps", type=int, default=60000)
    ap.add_argument("--fixed-schedule", action="store_true", help="cooldown exactly at --const-steps (no success-based rules)")
    ap.add_argument("--const-max", type=int, default=90000)
    ap.add_argument("--cooldown", type=int, default=10000)
    ap.add_argument("--final-lr-frac", type=float, default=0.025)
    ap.add_argument("--ckpt-every", type=int, default=10000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--workers", type=int, default=10, help="(unused; kept for CLI compatibility)")
    ap.add_argument("--eval-n", type=int, default=30)
    ap.add_argument("--eval-n-action-steps", type=int, default=10)
    ap.add_argument("--eval_conds", "--eval-conds", dest="eval_conds", default="plain,struct,struct_yellow,d_several",
                    help="conditions evaluated at every checkpoint (first one drives the schedule rules)")
    ap.add_argument("--camera", default="", help="camera JSON (assets/camera.json) for the in-training evals; must match the dataset")
    ap.add_argument("--layouts", default="", help="layouts JSON for the eval conditions (default assets/eval_layouts.json)")
    ap.add_argument("--base-dir", default="", help="base config dir under runs/ (default _smolvla_base_fast_<cameras>)")
    ap.add_argument("--cameras", default="overhead", help="policy cameras in prefix order, e.g. overhead or overhead,wrist")
    ap.add_argument("--lighting", "--eval-lighting", dest="lighting", default="",
                    help="lighting preset for the in-training evals (the dataset's lighting, e.g. shadow)")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--smoke", type=int, default=0, help="run this many steps only, report speed / memory")
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--wandb", action="store_true", help="(unused; kept for CLI compatibility)")
    # fast-path options
    ap.add_argument("--feat-cache", default="", help="comma list of cache dirs, one per camera (default $TOYVLA_DATA/feature_cache/<dataset>__<cam>)")
    ap.add_argument("--feat-in-ram", action="store_true", help="load the whole feature table into RAM (13.6 GB for 110k frames)")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--no-fused", action="store_true")
    ap.add_argument("--eval-pool", type=int, default=1,
                    help="envs per batched eval. 1 (default) = bit-identical to eval_closed_loop; 10 is ~4.5x faster but on 100 "
                         "seeds scored 64/100 vs 75/100 (paired p=0.03), so it is optional")
    ap.add_argument("--eval-fast-chunk", action="store_true", help="fast_core.FastChunker for eval chunks")
    args = ap.parse_args()

    import fast_core as FC
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets import EpisodeAwareSampler
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies import make_policy, make_pre_post_processors

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run = RUNS / args.run
    run.mkdir(parents=True, exist_ok=True)
    logf = open(run / "train_log.jsonl", "a")
    root = Path(args.dataset).expanduser()
    cam_keys = [f"observation.images.{c}" for c in args.cameras.split(",") if c]
    base = FC.make_base_config(RUNS / (args.base_dir or "_smolvla_base_fast_" + "_".join(k.split(".")[-1] for k in cam_keys)),
                               FC.dataset_cameras(root, cam_keys))

    cfg = PreTrainedConfig.from_pretrained(base)
    cfg.pretrained_path = str(base)
    cfg.device = "cuda"
    meta = LeRobotDatasetMetadata(args.repo_id, root=root)
    delta = resolve_delta_timestamps(cfg, meta)
    ds = LeRobotDataset(args.repo_id, root=root, delta_timestamps=delta)
    assert list(cfg.image_features) == cam_keys, (list(cfg.image_features), cam_keys)
    cdirs = [Path(x).expanduser() for x in args.feat_cache.split(",")] if args.feat_cache else \
        [FC.cache_dir_for(root, k) for k in cam_keys]
    caches = []
    for k, cd in zip(cam_keys, cdirs):
        c = FC.FeatureCache(cd)
        c.check(root)
        assert c.meta["image_key"] == k, (c.meta["image_key"], k)
        caches.append(c.load_to_ram() if args.feat_in_ram else c)

    policy = make_policy(cfg=cfg, ds_meta=meta)
    policy.to("cuda")
    pre, post = make_pre_post_processors(policy_cfg=cfg, dataset_stats=meta.stats)
    params = [p for p in policy.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    n_all = sum(p.numel() for p in policy.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10,
                            fused=not args.no_fused)
    tables = FC.TableBatcher(ds, pre, chunk=cfg.chunk_size)
    fast_loss = FC.FastLoss(policy, compile=not args.no_compile)

    sched = dict(const_until=args.const_steps, decay_start=None, decay_len=args.cooldown, decisions=[])
    step = 0
    resume = run / "resume.pt"
    if resume.exists() and not args.smoke:
        st = torch.load(resume, map_location="cpu", weights_only=False)
        policy.load_state_dict(st["policy"], strict=False)
        opt.load_state_dict(st["opt"])
        step, sched = st["step"], st["sched"]
        torch.set_rng_state(st["rng"])
        print(f"resumed at step {step}", flush=True)

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        if sched["decay_start"] is None:
            return args.lr
        f = min(1.0, (s - sched["decay_start"]) / sched["decay_len"])
        return args.lr * (args.final_lr_frac + (1 - args.final_lr_frac) * 0.5 * (1 + math.cos(math.pi * f)))

    def total_steps():
        return (sched["decay_start"] + sched["decay_len"]) if sched["decay_start"] is not None else None

    sampler = EpisodeAwareSampler(ds.meta.episodes["dataset_from_index"], ds.meta.episodes["dataset_to_index"],
                                  episode_indices_to_use=ds.episodes, drop_n_last_frames=0, shuffle=True,
                                  seed=args.seed + step, absolute_to_relative_idx=ds.absolute_to_relative_idx)

    def index_batches():            # = DataLoader(batch_size, sampler, drop_last=True) index grouping, epoch after epoch
        while True:
            buf = []
            for i in sampler:
                buf.append(int(i))
                if len(buf) == args.batch:
                    yield np.asarray(buf)
                    buf = []

    torch.empty((), dtype=torch.int64).random_()   # a DataLoader iterator would draw its base seed here (CPU RNG); kept so the noise stream is the one the policy was trained with
    feed = Prefetch(caches, index_batches())

    info = dict(run=args.run, dataset=str(root), episodes=ds.num_episodes, frames=ds.num_frames, fps=ds.fps,
                n_trainable=n_train, n_total=n_all, batch=args.batch, lr=args.lr, schedule=vars(args),
                input_features=list(cfg.input_features), chunk_size=cfg.chunk_size, started=now(),
                feat_cache=[str(c.path) for c in caches])
    (run / "run_info.json").write_text(json.dumps(info, indent=1, default=str))
    print(json.dumps(info, default=str), flush=True)

    def save_ckpt(s, tag=None):
        d = run / "checkpoints" / (tag or f"{s:06d}") / "pretrained_model"
        d.mkdir(parents=True, exist_ok=True)
        policy.save_pretrained(d)
        pre.save_pretrained(d)
        post.save_pretrained(d)
        tmp = run / "resume.pt.tmp"
        torch.save(dict(policy={n: p.detach().cpu() for n, p in policy.named_parameters() if p.requires_grad},
                        opt=opt.state_dict(), step=s, sched=sched, rng=torch.get_rng_state()), tmp)
        os.replace(tmp, resume)
        return d

    evals = []
    if (run / "eval_curve.jsonl").exists():
        evals = [json.loads(l) for l in (run / "eval_curve.jsonl").read_text().splitlines() if l.strip()]
    wrapper = SimpleNamespace(policy=policy, pre=pre, post=post, device=torch.device("cuda"),
                              image_keys=list(cfg.image_features))

    def closed_loop_eval(s):
        """Batched closed-loop eval of every --eval_conds condition on the same 'quick' seeds."""
        import eval_closed_loop as E
        import scene as S

        L = E.load_layouts(args.layouts or None, args.camera or None)
        if args.lighting:
            for cc in L["conditions"].values():
                cc["lighting"] = args.lighting
        seeds = L["sets"]["quick"]["seeds"][: args.eval_n]
        policy.eval()
        old = policy.config.n_action_steps
        policy.config.n_action_steps = args.eval_n_action_steps
        bpol = FC.BatchedPolicy(wrapper, fast_chunk=args.eval_fast_chunk)
        conds = [c for c in args.eval_conds.split(",") if c]
        per, allrecs, t0 = {}, {}, time.time()
        for c in conds:
            summ, recs = FC.run_condition_batched(bpol, S.SceneSpec.from_dict(L["conditions"][c]), seeds,
                                                  pool=args.eval_pool)
            per[c] = {k: summ[k] for k in ("rate", "success", "n", "wilson95", "t_success_median", "final_dist_mean",
                                           "reach_low_median", "reach_low_within_1cm", "reach_low_within_2cm",
                                           "fail_phase")}
            allrecs[c] = recs
        policy.config.n_action_steps = old
        policy.reset()
        policy.train()
        p0 = per[conds[0]]
        row = dict(step=s, cond=conds[0], rate=p0["rate"], success=p0["success"], n=p0["n"], wilson95=p0["wilson95"],
                   t_success_median=p0["t_success_median"], final_dist_mean=p0["final_dist_mean"],
                   fail_phase=p0["fail_phase"], per_cond=per, sec=time.time() - t0, ts=now(),
                   eval="batched pool %d" % args.eval_pool, eval_lighting=args.lighting or None)
        with open(run / "eval_curve.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        (run / "eval_episodes").mkdir(exist_ok=True)
        json.dump(allrecs, open(run / "eval_episodes" / f"{s:06d}.json", "w"))
        print("EVAL", json.dumps(row), flush=True)
        return row

    def decide(s):
        """Schedule rules after the eval at checkpoint step s (constant phase only)."""
        if sched["decay_start"] is not None or not evals:
            return
        if args.fixed_schedule:
            if s >= sched["const_until"]:
                sched["decay_start"] = s
                sched["decisions"].append(dict(step=s, action="cooldown (fixed schedule)"))
                (run / "schedule.json").write_text(json.dumps(sched, indent=1))
            return
        r = [e["rate"] for e in evals]
        if len(r) >= 2 and r[-1] >= 0.9 and r[-2] >= 0.9 and abs(r[-1] - r[-2]) <= 0.10:
            sched["decay_start"] = s
            sched["decisions"].append(dict(step=s, action="early cooldown", rates=r[-2:]))
        elif s >= sched["const_until"]:
            rising = (len(r) >= 2 and r[-1] - r[-2] >= 0.10) or (len(r) >= 3 and r[-1] - r[-3] >= 0.15)
            if rising and s + args.ckpt_every <= args.const_max:
                sched["const_until"] = s + args.ckpt_every
                sched["decisions"].append(dict(step=s, action=f"extend constant to {sched['const_until']}", rates=r[-3:]))
            else:
                sched["decay_start"] = s
                sched["decisions"].append(dict(step=s, action="cooldown (scheduled)", rates=r[-3:]))
        (run / "schedule.json").write_text(json.dumps(sched, indent=1))
        if sched["decisions"]:
            print("SCHEDULE", json.dumps(sched["decisions"][-1]), flush=True)

    policy.train()
    torch.cuda.reset_peak_memory_stats()
    t_log, lsum, nl, t_data = time.time(), 0.0, 0, 0.0
    t_start = time.time()
    while True:
        ctrl = run / "control.json"
        if ctrl.exists() and sched["decay_start"] is None:
            c = json.loads(ctrl.read_text())
            if c.get("cooldown_at") is not None and step >= int(c["cooldown_at"]):
                sched["decay_start"] = step
                sched["decisions"].append(dict(step=step, action="manual cooldown (control.json)"))
        if args.fixed_schedule and sched["decay_start"] is None and step >= sched["const_until"] and \
                (args.no_eval or step % args.ckpt_every != 0 or step == 0):
            sched["decay_start"] = step
            sched["decisions"].append(dict(step=step, action="cooldown (fixed schedule)"))
        tot = total_steps()
        if (tot is not None and step >= tot) or (args.smoke and step >= args.smoke):
            break
        td = time.time()
        idx, feat = next(feed)
        feat = [f.cuda(non_blocking=True) for f in feat]
        batch = tables.batch(idx)
        t_data += time.time() - td
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = fast_loss(batch, feat)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        lsum += float(loss)
        nl += 1
        if step % args.log_every == 0 or (args.smoke and step == args.smoke):
            dt = time.time() - t_log
            row = dict(step=step, loss=lsum / nl, grad_norm=float(gn), lr=lr, it_s=nl / dt, data_frac=t_data / dt,
                       peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
                       reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30, ts=now())
            logf.write(json.dumps(row) + "\n")
            logf.flush()
            print(json.dumps(row), flush=True)
            t_log, lsum, nl, t_data = time.time(), 0.0, 0, 0.0
        if not args.smoke and step % args.ckpt_every == 0:
            save_ckpt(step)
            if not args.no_eval and sched["decay_start"] is None:
                evals.append(closed_loop_eval(step))
                decide(step)
            (run / "schedule.json").write_text(json.dumps(sched, indent=1))
    if args.smoke:
        el = time.time() - t_start
        res = dict(steps=step, sec=el, it_s_overall=step / el, peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
                   reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30, batch=args.batch)
        if not args.no_eval:
            t0 = time.time()
            row = closed_loop_eval(step)
            res["eval_sec"] = time.time() - t0
            res["eval_rate"] = row["rate"]
        (run / "smoke.json").write_text(json.dumps(res, indent=1))
        print("SMOKE", json.dumps(res), flush=True)
        return 0
    d = save_ckpt(step, tag="final")
    if not args.no_eval:
        evals.append(closed_loop_eval(step))
    info["finished"] = now()
    info["final_step"] = step
    info["final_ckpt"] = str(d)
    (run / "run_info.json").write_text(json.dumps(info, indent=1, default=str))
    print("DONE", step, d, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
