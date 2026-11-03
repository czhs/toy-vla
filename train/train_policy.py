"""Fine-tune the toy policy's action expert (the vision-language model stays frozen).

    python train/build_feature_cache.py [--block_colour red|green|blue]     # once per colour; about 25 min on a 24 GB GPU, 61.4 GB on disk
    python train/train_policy.py --run toy_policy_red --mask_pad --max_steps 134400 --const_steps 112000 --cooldown 22400 [--block_colour red|green|blue]   # about 3 h on a 24 GB GPU

Recipe: warm start from lerobot/smolvla_base; AdamW (0.9, 0.95), weight decay 1e-10, gradient clip 10, batch 32, lr 1e-4 with 1k warmup; constant to --const_steps, then a --cooldown cosine to 0.02x; bf16 autocast;
flow MSE on coordinate 0 with weight 16, denominator B*50*47, padding coordinates weight 0 with --mask_pad. Training scenes: a pool of 500k fixed by seed. The target colour only changes the block's RGB; the prompt stays
"Move the arm to the red block." Output: <out_root>/<run>/{metadata.json, training.jsonl, evaluations.jsonl, milestones, resume_latest.pt, final.pt}.

Speedups, each with a switch: image features from the cache (--no_featcache: render and run SigLIP in the step); split prefix pass under no_grad without the language pads (--no_split); fused AdamW (--no_fused);
torch.compile (--no_compile); the next step's prefix on a side stream (--no_overlap). --eval_mode cached is sample_actions on cached panel features; legacy recomputes the features; fast / fast_compiled use a split chunker."""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse, json, math, sys, time, hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from toyvla import scene as S
from toyvla.paths import DATA

ap = argparse.ArgumentParser()
ap.add_argument('--block_colour', choices=['red', 'green', 'blue'], default='red', help='RGB of the target block; red (220, 30, 30), green (30, 220, 30), blue (30, 30, 220)')
ap.add_argument('--run', required=True)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--batch', type=int, default=32)
ap.add_argument('--workers', type=int, default=8)          # legacy DataLoader workers: also defines the sample order
ap.add_argument('--min_steps', type=int, default=40_000)
ap.add_argument('--max_steps', type=int, default=250_000)
ap.add_argument('--const_steps', type=int, default=0)
ap.add_argument('--cooldown', type=int, default=0)
ap.add_argument('--eval_every', type=int, default=2000)
ap.add_argument('--xeval_every', type=int, default=10_000)
ap.add_argument('--window', type=int, default=10)
ap.add_argument('--tol', type=float, default=0.03)
ap.add_argument('--n_panel', type=int, default=256)
ap.add_argument('--n_xpanel', type=int, default=128)
ap.add_argument('--resume', action='store_true')
ap.add_argument('--smoke', action='store_true')
# fast-path options
ap.add_argument('--featcache', default='', help='feature cache dir (default data/feature_cache/<key>)')
ap.add_argument('--no_featcache', action='store_true')
ap.add_argument('--no_split', action='store_true')
ap.add_argument('--no_fused', action='store_true')
ap.add_argument('--no_compile', action='store_true')
ap.add_argument('--no_overlap', action='store_true')
ap.add_argument('--eval_mode', choices=['cached', 'legacy', 'fast', 'fast_compiled'], default='cached',
                help='cached: sample_actions on cached panel features (default); fast: eager split chunker; fast_compiled: compiled chunker, systematically shifted')
ap.add_argument('--seed', type=int, default=20260930, help='training seed')
ap.add_argument('--mask_pad', action='store_true', help='zero loss weight on the 32-NJ padding action coordinates (no gradient from them); DEN unchanged so the joint coordinate keeps the reference loss scale')
ap.add_argument('--stop_after', type=int, default=0, help='benchmark: exit after this step (no final eval / final.pt)')
ap.add_argument('--render_procs', type=int, default=16, help='processes for rendering the eval panels at start')
ap.add_argument('--out_root', default='', help='run dir = <out_root>/<run> (default data/runs/<run>)')
args = ap.parse_args()
if args.smoke:
    args.min_steps, args.max_steps, args.eval_every, args.xeval_every, args.window = 60, 120, 20, 40, 2
    args.n_panel, args.n_xpanel = 64, 32
assert (not args.smoke) or args.run.startswith('smoke'), '--smoke runs must be named smoke*'
OUT = (Path(args.out_root) if args.out_root else DATA / 'runs') / args.run; OUT.mkdir(parents=True, exist_ok=True)
if not args.resume:
    for fn in ('evaluations.jsonl', 'training.jsonl', 'perf.jsonl'):
        if (OUT / fn).exists(): (OUT / fn).rename(OUT / (fn + time.strftime('.old-%Y%m%dT%H%M%S')))
S.COL['red'] = {'red': (220, 30, 30), 'green': (30, 220, 30), 'blue': (30, 30, 220)}[args.block_colour]
POOL_N = S.POOL; NJ = S.NJ
TRAIN_SPLIT, VAL_SPLIT = 'full', 'full'
BASE = dict(model="lerobot/smolvla_base", model_revision="d9f33c94a60fb382c90dea2164c96845bd955e28",
            backbone="HuggingFaceTB/SmolVLM2-500M-Video-Instruct", backbone_revision="7b375e1b73b11138ff12fe22c8f2822d8fe03467")
TRAIN_SEED, NOISE_SEED = args.seed, 77
MILESTONES = {500, 1000, 2000, 4000, 8000, 16000, 24000, 32000, 48000, 64000, 96000, 128000, 192000}
torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
now = lambda: datetime.now(timezone.utc).isoformat()
USE_CACHE = not args.no_featcache
OVERLAP = USE_CACHE and not args.no_split and not args.no_overlap


def atomic_json(p, o): t = p.with_suffix(p.suffix + '.tmp'); t.write_text(json.dumps(o, indent=2) + '\n'); os.replace(t, p)
def atomic_torch(p, o): t = p.with_suffix(p.suffix + '.tmp'); torch.save(o, t); os.replace(t, p)
def append(p, o):
    with p.open('a') as f: f.write(json.dumps(o) + '\n')


def scene_image(sc, bg=None, cam=None):
    return S.render(sc, bg or 'plain', cam=cam)


def _render_job(a): sc, bg, cam = a; return scene_image(sc, bg, cam)


def make_panel(scs, bg=None, cam=None, pool=None):
    jobs = [(sc, bg, cam) for sc in scs]
    ims = pool.map(_render_job, jobs, chunksize=16) if pool is not None else [_render_job(j) for j in jobs]
    imgs = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).contiguous()
    ang = torch.tensor([[sc['t1s'], sc['t2s'], sc['t1'], sc['t2']] for sc in scs], dtype=torch.float32)
    return dict(imgs=imgs, ang=ang, prompts=[S.text_prompt(sc) for sc in scs], scs=scs)


# ------------------------------------------------------------------ panels (rendered before CUDA init, in parallel; same images)
other_bg = 'struct'
t0 = time.time()
import multiprocessing as mp
with mp.get_context('fork').Pool(max(1, args.render_procs)) as rpool:
    PANELS = {'train': make_panel([S.pool_scene(i, TRAIN_SPLIT) for i in range(args.n_panel)], pool=rpool),
              'val': make_panel(S.panel('val', args.n_panel, split=VAL_SPLIT), pool=rpool),
              'H1': make_panel(S.panel('H1', args.n_panel), pool=rpool), 'H2': make_panel(S.panel('H2', args.n_panel), pool=rpool),
              'HC': make_panel(S.panel('HC', args.n_panel), pool=rpool), 'HCrev': make_panel(S.panel('HCrev', args.n_panel), pool=rpool)}
    CAMS = {'cam_u+8': (8, 0, 0, 1), 'cam_v+8': (0, 8, 0, 1), 'cam_u-8': (-8, 0, 0, 1), 'cam_roll2': (0, 0, 2, 1), 'cam_zoom1.03': (0, 0, 0, 1.03)}
    cam_scs = S.panel('cam', args.n_xpanel, split=VAL_SPLIT)
    XPANELS = {'cam_ref': make_panel(cam_scs, pool=rpool), f'xbg_{other_bg}': make_panel(S.panel('val', args.n_xpanel, split=VAL_SPLIT), bg=other_bg, pool=rpool)}
    XPANELS.update({k: make_panel(cam_scs, cam=c, pool=rpool) for k, c in CAMS.items()})
t_panels = time.time() - t0
print(f'panels rendered in {t_panels:.0f}s', flush=True)

# ------------------------------------------------------------------ model (exactly as the plain lerobot step)
from huggingface_hub import snapshot_download
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
import fast_core as TC
path = snapshot_download(BASE['model'], revision=BASE['model_revision'])
config = PreTrainedConfig.from_pretrained(path); config.device = 'cpu'; config.load_vlm_weights = False
config.vlm_model_name = snapshot_download(BASE['backbone'], revision=BASE['backbone_revision'],
                                          allow_patterns=['*.json', '*.txt', '*.model', '*.jinja'])
policy = SmolVLAPolicy.from_pretrained(path, config=config, strict=False).float().to('cuda')
model = policy.model; vlmm = model.vlm_with_expert.get_vlm_model(); vwe = model.vlm_with_expert
n_default = sum(p.numel() for p in model.parameters() if p.requires_grad)
assert n_default == 99_880_992, n_default
c = model.config
assert c.attention_mode == 'cross_attn' and not c.add_image_special_tokens and c.prefix_length <= 0 and c.chunk_size == 50
assert c.num_steps == 10, c.num_steps
names = {n for n, p in model.named_parameters() if p.requires_grad}
n_train = sum(p.numel() for n, p in model.named_parameters() if n in names)
vis_attn = getattr(vlmm.vision_model.config, '_attn_implementation', None)
print(f'trainable {n_train:,}; vision attn impl {vis_attn}', flush=True)
def set_train(): model.train()


tok = policy.language_tokenizer
def tokenize(texts):
    t = tok([x + '\n' for x in texts], padding='max_length', max_length=48, return_tensors='pt')
    return t['input_ids'], t['attention_mask'].bool()
PROMPT_TEXT = S.PROMPT
FIX_IDS, FIX_MK = tokenize([S.PROMPT])
PROMPT_TOKENS = int(FIX_MK.sum())
print(f'prompt {PROMPT_TEXT!r}: {PROMPT_TOKENS} language tokens {FIX_IDS[0, :PROMPT_TOKENS].tolist()}; prefix = 64 image + {PROMPT_TOKENS} language tokens', flush=True)
PE = TC.PrefixEmb(model, FIX_IDS, FIX_MK)
IDS_G, MK_G = FIX_IDS.to('cuda'), FIX_MK.to('cuda')


class _FeatInject:
    """Within the block, vwe.embed_image(img) returns the given features (the legacy forward / sample_actions then run
    unchanged on them). Exact when the features are what embed_image would compute for that batch."""
    def __init__(self, feat): self.feat = feat
    def __enter__(self): vwe.embed_image = lambda img: self.feat
    def __exit__(self, *a): del vwe.embed_image


def state_actions(ang):
    B = ang.shape[0]
    st = torch.zeros(B, 32, device='cuda'); act = torch.zeros(B, 50, 32, device='cuda')
    st[:, :NJ] = ang[:, :NJ] / 180
    act[:, :, :NJ] = (ang[:, 2:2 + NJ] / 180)[:, None]
    return st, act


def to_img(img_u8): return img_u8.to('cuda', non_blocking=True).float().div_(127.5).sub_(1.0)


# ------------------------------------------------------------------ eval
t0 = time.time()
if args.eval_mode != 'legacy':                       # panel features: the same bs-64 chunks the legacy eval feeds SigLIP
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for P in list(PANELS.values()) + list(XPANELS.values()):
            P['feat'] = torch.cat([TC.image_features(vwe, to_img(P['imgs'][j:j + 64])) for j in range(0, len(P['scs']), 64)])
t_pfeat = time.time() - t0
CHUNKER = TC.FastChunker(model, PE, compile=args.eval_mode == 'fast_compiled') if args.eval_mode.startswith('fast') else None
EVAL_T = []


@torch.no_grad()
def evaluate(name, P, step):
    model.eval(); preds = []
    for j in range(0, len(P['scs']), 64):
        ang = P['ang'][j:j + 64].to('cuda', non_blocking=True); B = ang.shape[0]
        st, _ = state_actions(ang)
        noise = torch.randn((B, 50, 32), device='cuda', generator=torch.Generator('cuda').manual_seed(NOISE_SEED + j))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if args.eval_mode == 'legacy':
                out = model.sample_actions([to_img(P['imgs'][j:j + 64])], [torch.ones(B, dtype=torch.bool, device='cuda')],
                                           IDS_G.expand(B, -1), MK_G.expand(B, -1), st, noise=noise)
            elif args.eval_mode == 'cached':
                with _FeatInject(P['feat'][j:j + 64]):
                    out = model.sample_actions([torch.empty(B, 0, device='cuda')], [torch.ones(B, dtype=torch.bool, device='cuda')],
                                               IDS_G.expand(B, -1), MK_G.expand(B, -1), st, noise=noise)
            else:
                out = CHUNKER(P['feat'][j:j + 64], st, noise)
        preds.append(out[:, :, :2].float().cpu().numpy())
    pr = np.concatenate(preds) * 180
    tg = P['ang'][:, 2:4].numpy()
    e = S.angle_err(pr, tg[:, None]); ee, inside = S.ee_err(pr, P['scs'])
    r = dict(panel=name, step=step, n=len(pr), ang_mae=float(e.mean()), t1_mae=float(e.mean()), unused_coord1_mae=float(np.abs(pr[..., 1]).mean()),
             ee_mae=float(ee.mean()), ee_med=float(np.median(ee)), ee_p95=float(np.percentile(ee, 95)), success=float(inside.mean()),
             outside_canonical=float(S.outside_canonical(pr).mean()), pos_spread=float(np.abs(S.wrap(pr[..., 0] - pr[:, :1, 0])).mean()), ts=now())
    append(OUT / 'evaluations.jsonl', r); set_train(); return r


def eval_all(step, x=False):
    torch.cuda.synchronize(); te = time.time()
    res = {k: evaluate(k, P, step) for k, P in PANELS.items()}
    if x: res.update({k: evaluate(k, P, step) for k, P in XPANELS.items()})
    held = float(np.mean([res[k]['ee_mae'] for k in ('H1', 'H2', 'HC')])); held_med = float(np.mean([res[k]['ee_med'] for k in ('H1', 'H2', 'HC')]))
    dt = time.time() - te; EVAL_T.append(dt)
    append(OUT / 'perf.jsonl', dict(kind='eval', step=step, x=bool(x), sec=dt, eval_mode=args.eval_mode, ts=now()))
    print(f'EVAL {step} ({dt:.1f}s): ' + ' | '.join(f"{k} ee {v['ee_mae']:.1f}px ang {v['ang_mae']:.2f} succ {v['success']:.2f}" for k, v in res.items()) + f' | held {held:.1f} (med {held_med:.1f})', flush=True)
    return res['val']['ee_med'], held_med


# ------------------------------------------------------------------ optimisation
params = [p for n, p in model.named_parameters() if n in names]
groups = [dict(params=params, lr_mult=1.0)]
opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10, fused=None if args.no_fused else True)
wts = torch.ones((1, 1, 32), device='cuda'); wts[:, :, :NJ] = 16; DEN = (32 - NJ) + 16 * NJ
if args.mask_pad: wts[:, :, NJ:] = 0; print('mask_pad: padding coordinates carry no loss; DEN kept at', DEN, flush=True)
LOSS_STR = f'flow MSE, coords 0..{NJ-1} weight 16, denom B*50*{DEN}' + (', padding coords weight 0 (mask_pad)' if args.mask_pad else '')
state = lambda dtype=torch.float32: {n: p.detach().to('cpu', dtype) for n, p in model.named_parameters() if n in names}
sched = dict(decay_start=None, decay_len=None, stop_at=None)
hist = []
start = 0
SL = None if args.no_split else TC.SplitLoss(model, PE, compile=not args.no_compile)
CACHE = None
if USE_CACHE:
    cdir = Path(args.featcache) if args.featcache else TC.FEAT_ROOT / TC.cache_key(args.block_colour)
    CACHE = TC.FeatCache(cdir)
    CACHE.check(dict(block_colour=args.block_colour, split=TRAIN_SPLIT, pool=S.POOL, pool_seed=S.POOL_SEED, n=POOL_N, scene_sha256=hashlib.sha256(Path(S.__file__).read_bytes()).hexdigest()))
    ANG_G = torch.from_numpy(CACHE.angles).to('cuda')
fast_info = dict(featcache=str(CACHE.path) if CACHE else None, split_step=not args.no_split,
                 fused_adamw=not args.no_fused, compile=not args.no_compile, overlap=OVERLAP, eval_mode=args.eval_mode,
                 core_sha256=TC.sha256_file(TC.__file__), featcache_meta=CACHE.meta if CACHE else None)
if args.resume:
    ck = torch.load(OUT / 'resume_latest.pt', map_location='cpu', weights_only=False)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in names: p.copy_(ck['model'][n].to(p.device))
    opt.load_state_dict(ck['optimizer']); start, sched, hist = ck['step'], ck['sched'], ck['hist']
    for g in opt.param_groups: g['fused'] = not args.no_fused; g['foreach'] = None if not args.no_fused else g.get('foreach')
    if not args.no_fused:
        for st_ in opt.state.values():
            if 'step' in st_: st_['step'] = st_['step'].to('cuda', torch.float32)
    def _keep(l):
        try: return json.loads(l)['step'] <= start
        except Exception: return False
    for fn in ('evaluations.jsonl', 'training.jsonl'):
        p_ = OUT / fn
        if p_.exists(): p_.write_text(''.join(l + '\n' for l in p_.read_text().splitlines() if l.strip() and _keep(l)))
    md = json.loads((OUT / 'metadata.json').read_text())
    md.setdefault('resumes', []).append(dict(at=now(), step=start, prompt=PROMPT_TEXT, prompt_tokens=PROMPT_TOKENS, train_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                            scene_sha256=hashlib.sha256(Path(S.__file__).read_bytes()).hexdigest(), **fast_info))
    atomic_json(OUT / 'metadata.json', md)
    print('RESUME from', start, sched, flush=True)
else:
    torch.manual_seed(TRAIN_SEED); torch.cuda.manual_seed_all(TRAIN_SEED)
    atomic_json(OUT / 'metadata.json', dict(created=now(), args=vars(args), **BASE, trainable_params=n_train, vision_trainable=False, block_colour=args.block_colour, n_joints=NJ, train_split=TRAIN_SPLIT,
        vision_attn_impl=vis_attn, prompt=PROMPT_TEXT, prompt_tokens=PROMPT_TOKENS, prompt_ids=FIX_IDS[0, :PROMPT_TOKENS].tolist(),
        scene_sha256=hashlib.sha256(Path(S.__file__).read_bytes()).hexdigest(), scene_module=S.__name__, train_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        pool=S.POOL, pool_seed=S.POOL_SEED, train_seed=TRAIN_SEED, noise_seed=NOISE_SEED, loss=LOSS_STR,
        optimizer='AdamW b=(0.9,0.95) wd=1e-10 clip 10' + (' (fused)' if not args.no_fused else ''), schedule='warmup 1k, constant, plateau (val ee_med and mean held-out ee_med) -> cosine cooldown to 0.02x',
        fast=fast_info, startup=dict(panels_render_s=t_panels, panel_features_s=t_pfeat)))
    v, h = eval_all(0, x=True); hist.append((0, v, h))


def lr_at(step):
    if sched['decay_start'] is None: return args.lr * min(1.0, step / 1000)
    f = min(1.0, (step - sched['decay_start']) / sched['decay_len'])
    return args.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * f)))


def dlen(step): return 20 if args.smoke else max(10_000, int(0.2 * step))


def plateaued():
    W = args.window
    if len(hist) <= W + 1 or hist[-1][0] < args.min_steps: return False
    old, new = hist[:-W], hist[-W:]
    vo, vn = min(x[1] for x in old), min(x[1] for x in new); ho, hn = min(x[2] for x in old), min(x[2] for x in new)
    return vn > (1 - args.tol) * vo and hn > (1 - args.tol) * ho


def save_resume(step):
    atomic_torch(OUT / 'resume_latest.pt', dict(step=step, model=state(), optimizer=opt.state_dict(), sched=sched, hist=hist))


# ------------------------------------------------------------------ data
class Stream(torch.utils.data.IterableDataset):                      # = train_ik.Stream (used with --no_featcache)
    def __init__(self, seed): self.seed = seed
    def __iter__(self):
        wi = torch.utils.data.get_worker_info(); rng = np.random.default_rng([self.seed, 0 if wi is None else wi.id])
        while True:
            i = int(rng.integers(S.POOL)); sc = S.pool_scene(i, TRAIN_SPLIT)
            yield (torch.from_numpy(np.ascontiguousarray(scene_image(sc))).permute(2, 0, 1),
                   torch.tensor([sc['t1s'], sc['t2s'], sc['t1'], sc['t2']], dtype=torch.float32), '', i)


if USE_CACHE:
    torch.empty((), dtype=torch.int64).random_()                     # the legacy DataLoader iterator draws its base seed here
    feed = TC.Prefetch(CACHE, TC.LegacyIndexStream(TRAIN_SEED + start, args.workers, args.batch, POOL_N))
    def next_batch():                                                 # -> (feat bf16 cuda, ang fp32 cuda)
        idx, f = next(feed)
        return f.to('cuda', non_blocking=True), ANG_G[torch.from_numpy(idx).to('cuda', non_blocking=True)]
else:
    loader = torch.utils.data.DataLoader(Stream(TRAIN_SEED + start), batch_size=args.batch, num_workers=args.workers, pin_memory=True,
                                         prefetch_factor=4, persistent_workers=True)
    it = iter(loader)
    def next_batch():
        img_u8, ang, _, _ = next(it)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            f = TC.image_features(vwe, to_img(img_u8))
        return f, ang.to('cuda', non_blocking=True)

side = torch.cuda.Stream() if OVERLAP else None


def launch_prefix():
    """fetch the next batch and (overlap) compute its prefix K/V on the side stream."""
    tw = time.time(); feat, ang = next_batch(); wait = time.time() - tw
    if not OVERLAP: return feat, ang, None, None, wait
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.autocast('cuda', dtype=torch.bfloat16):
        kv = SL.kv(feat)
        ev = torch.cuda.Event(); ev.record(side)
    main = torch.cuda.current_stream()
    for k, v in kv: k.record_stream(main); v.record_stream(main)
    feat.record_stream(main); ang.record_stream(main)
    return feat, ang, kv, ev, wait


set_train(); tl = time.time(); last_eval = (start, True); step = start
LBUF = torch.zeros(100, device='cuda'); nl = 0; wsum = 0.0; ev_s = 0.0
nxt = launch_prefix()
while True:
    step += 1
    lr = lr_at(step)
    for g in opt.param_groups: g['lr'] = lr * g['lr_mult']
    feat, ang, kv, ev, wait = nxt; wsum += wait
    if OVERLAP: nxt = launch_prefix()                                 # prefix of step s+1 runs while step s trains
    st, act = state_actions(ang); B = st.shape[0]
    opt.zero_grad(set_to_none=True)
    noise = model.sample_noise((B, 50, 32), 'cuda'); time_ = model.sample_time(B, 'cuda')   # = VLAFlowMatching.forward's draws
    with torch.autocast('cuda', dtype=torch.bfloat16):
        if args.no_split:
            with _FeatInject(feat):
                losses = model.forward([torch.empty(B, 0, device='cuda')], [torch.ones(B, dtype=torch.bool, device='cuda')],
                                       IDS_G.expand(B, -1), MK_G.expand(B, -1), st, act, noise=noise, time=time_)
        elif OVERLAP:
            torch.cuda.current_stream().wait_event(ev)
            losses = SL.from_kv(kv, st, act, noise, time_)
        else:
            losses = SL(feat, st, act, noise, time_)
        loss = (losses * wts).sum() / (B * 50 * DEN)
    loss.backward(); gn = torch.nn.utils.clip_grad_norm_(params, 10.0)
    opt.step()
    LBUF[nl] = loss.detach(); nl += 1
    if not OVERLAP: nxt = launch_prefix()
    if step % 100 == 0 or step == start + 10:
        ls = LBUF[:nl].tolist()
        assert all(math.isfinite(x) for x in ls), f'non-finite loss before step {step}'
        el = time.time() - tl
        row = dict(step=step, loss=sum(ls) / nl, grad_norm=float(gn), lr=lr, it_s=nl / el, peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30, ts=now())
        append(OUT / 'training.jsonl', row); print('TRAIN ' + json.dumps(row), flush=True)
        append(OUT / 'perf.jsonl', dict(kind='train', step=step, n=nl, sec=el, it_s=nl / el, data_wait_s=wsum, eval_s=ev_s, it_s_ex_eval=nl / max(1e-9, el - ev_s), ts=now()))
        tl = time.time(); nl = 0; wsum = 0.0; ev_s = 0.0
    if step in MILESTONES: atomic_torch(OUT / f'milestone_{step:06d}.pt', state(torch.bfloat16))
    if args.stop_after and step >= args.stop_after:
        atomic_torch(OUT / f'bench_{step:06d}.pt', state()); print('STOP_AFTER', step, flush=True); os._exit(0)
    if step % args.eval_every == 0:
        te = time.time()
        v, h = eval_all(step, x=(step % args.xeval_every == 0)); hist.append((step, v, h)); last_eval = (step, step % args.xeval_every == 0)
        fixed = args.const_steps > 0 and step >= args.const_steps
        if sched['decay_start'] is None and (fixed or (args.const_steps <= 0 and plateaued()) or step + dlen(step) >= args.max_steps):
            L = (args.cooldown or dlen(step)) if fixed else dlen(step)
            sched.update(decay_start=step, decay_len=max(1, min(L, args.max_steps - step)))
            sched['stop_at'] = step + sched['decay_len']
            atomic_torch(OUT / f'milestone_{step:06d}_decaystart.pt', state(torch.bfloat16))
            print('PLATEAU -> cooldown', json.dumps(sched), flush=True)
        if step % (5 * args.eval_every) == 0: save_resume(step)
        atomic_json(OUT / 'status.json', dict(status='running', step=step, sched=sched, last_val_ee=v, last_held_ee=h, updated=now(),
                                              peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30))
        ev_s += time.time() - te
    if sched['stop_at'] is not None and step >= sched['stop_at']:
        break
if last_eval[0] != step: v, h = eval_all(step, x=True); hist.append((step, v, h))
elif not last_eval[1]: [evaluate(k, P, step) for k, P in XPANELS.items()]
atomic_torch(OUT / 'final.pt', state()); save_resume(step)
atomic_json(OUT / 'status.json', dict(status='complete', step=step, sched=sched, final_val_ee=v, final_held_ee=h, updated=now()))
(OUT / 'TRAIN.DONE').write_text(now() + '\n'); print('COMPLETE', step, v, h, flush=True)
os._exit(0)
