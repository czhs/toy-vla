"""Build the frozen-image-feature cache for train_policy.py: connector(SigLIP(img)) of every pool scene (scene.pool_scene(i, 'full'), i < POOL) under bf16 autocast, batch 32.

Output (default data/feature_cache/toy_plain_<colour>/):
  features.u16  [POOL, 64, 960] bf16 bits (123 KB per scene; 61.4 GB for the 500k pool)
  angles.npy    [POOL, 4] float32 (t1s, t2s, t1, t2)
  done.npy      per-block completion flags (resumable; blocks of --block scenes)
  meta.json     what the cache is for (checked by train_policy.py)

    python train/build_feature_cache.py [--block_colour red|green|blue]    # full pool, about 25 min on a 24 GB GPU
    python train/build_feature_cache.py --max_blocks 2                     # partial, for testing"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse, hashlib, json, sys, threading, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from toyvla import scene as S
import fast_core as TC

ap = argparse.ArgumentParser()
ap.add_argument('--block_colour', choices=['red', 'green', 'blue'], default='red')
ap.add_argument('--out', default='')
ap.add_argument('--batch', type=int, default=32)          # = the legacy training batch (same SigLIP batch shape)
ap.add_argument('--block', type=int, default=4096)
ap.add_argument('--workers', type=int, default=16)
ap.add_argument('--max_blocks', type=int, default=0)
ap.add_argument('--n', type=int, default=0)               # 0 = S.POOL
args = ap.parse_args()

SPLIT = 'full'
S.COL['red'] = {'red': (220, 30, 30), 'green': (30, 220, 30), 'blue': (30, 30, 220)}[args.block_colour]
KEY = TC.cache_key(args.block_colour)
OUT = Path(args.out) if args.out else TC.FEAT_ROOT / KEY
OUT.mkdir(parents=True, exist_ok=True)
N = args.n or S.POOL
assert args.block % args.batch == 0
NB = (N + args.block - 1) // args.block


def scene_image(sc): return S.render(sc, 'plain', cam=None)


class Blocks(torch.utils.data.Dataset):
    """item j = one batch of `batch` consecutive pool indices of the current block list."""
    def __init__(self, starts): self.starts = starts
    def __len__(self): return len(self.starts)
    def __getitem__(self, j):
        s = self.starts[j]; idx = range(s, min(s + args.batch, N)); scs = [S.pool_scene(i, SPLIT) for i in idx]
        img = torch.from_numpy(np.stack([scene_image(sc) for sc in scs])).permute(0, 3, 1, 2).contiguous()
        ang = torch.tensor([[sc['t1s'], sc['t2s'], sc['t1'], sc['t2']] for sc in scs], dtype=torch.float32)
        return s, img, ang


meta = dict(n=N, key=KEY, block_colour=args.block_colour, split=SPLIT, pool=S.POOL, pool_seed=S.POOL_SEED,
            scene_sha256=hashlib.sha256(Path(S.__file__).read_bytes()).hexdigest(),
            batch=args.batch, block=args.block, dtype='bfloat16 stored as uint16', shape=[N, TC.FEAT_TOKENS, TC.FEAT_DIM],
            features='connector(SigLIP(u8/127.5-1)) under bf16 autocast, = SmolVLMWithExpertModel.embed_image',
            torch=torch.__version__, gpu=torch.cuda.get_device_name(0), builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
mp = OUT / 'meta.json'
if mp.exists():
    old = json.loads(mp.read_text())
    diff = {k: (old.get(k), v) for k, v in meta.items() if k not in ('torch', 'gpu', 'builder_sha256', 'created') and old.get(k) != v}
    assert not diff, f'existing cache at {OUT} was built for something else: {diff}'
    meta['created'] = old.get('created')
else:
    meta['created'] = datetime.now(timezone.utc).isoformat()
mp.write_text(json.dumps(meta, indent=2) + '\n')
fpath = OUT / 'features.u16'
feats = np.memmap(fpath, dtype=np.uint16, mode='r+' if fpath.exists() else 'w+', shape=(N, TC.FEAT_TOKENS, TC.FEAT_DIM))
apath = OUT / 'angles_partial.npy'
angles = np.load(apath) if apath.exists() else np.full((N, 4), np.nan, np.float32)
dpath = OUT / 'done.npy'
done = np.load(dpath) if dpath.exists() else np.zeros(NB, bool)
todo = [b for b in range(NB) if not done[b]]
if args.max_blocks: todo = todo[:args.max_blocks]
print(f'{KEY}: {N} scenes, {NB} blocks, {int(done.sum())} done, building {len(todo)} -> {OUT}', flush=True)

# ---- model
from huggingface_hub import snapshot_download
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
BASE = dict(model="lerobot/smolvla_base", model_revision="d9f33c94a60fb382c90dea2164c96845bd955e28",
            backbone="HuggingFaceTB/SmolVLM2-500M-Video-Instruct", backbone_revision="7b375e1b73b11138ff12fe22c8f2822d8fe03467")
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
path = snapshot_download(BASE['model'], revision=BASE['model_revision'])
config = PreTrainedConfig.from_pretrained(path); config.device = 'cpu'; config.load_vlm_weights = False
config.vlm_model_name = snapshot_download(BASE['backbone'], revision=BASE['backbone_revision'],
                                          allow_patterns=['*.json', '*.txt', '*.model', '*.jinja'])
policy = SmolVLAPolicy.from_pretrained(path, config=config, strict=False).float().to('cuda')
model = policy.model; model.train()     # SmolVLA keeps the vision-language model in eval mode
vwe = model.vlm_with_expert

starts = [s for b in todo for s in range(b * args.block, min((b + 1) * args.block, N), args.batch)]
loader = torch.utils.data.DataLoader(Blocks(starts), batch_size=None, shuffle=False, num_workers=args.workers,
                                     pin_memory=True, prefetch_factor=4, persistent_workers=False)
wq = []                                                   # writer thread: D2H done -> memmap write
lock = threading.Lock()


def writer(s, host, ev):
    ev.synchronize(); feats[s:s + host.shape[0]] = host.numpy().view(np.uint16)


t0 = time.time(); n_done = 0; threads = []
for s, img, ang in loader:
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        x = img.to('cuda', non_blocking=True).float().div_(127.5).sub_(1.0)
        f = vwe.embed_image(x)
    assert f.dtype == torch.bfloat16 and f.shape[1:] == (TC.FEAT_TOKENS, TC.FEAT_DIM), (f.dtype, f.shape)
    host = torch.empty(f.shape, dtype=torch.int16, pin_memory=True)
    host.copy_(f.view(torch.int16), non_blocking=True); ev = torch.cuda.Event(); ev.record()
    th = threading.Thread(target=writer, args=(s, host, ev)); th.start(); threads.append(th)
    angles[s:s + ang.shape[0]] = ang.numpy()
    n_done += ang.shape[0]
    if len(threads) > 16: threads.pop(0).join()
    e = s + ang.shape[0]
    if e % args.block == 0 or e == N:                   # block finished
        for th in threads: th.join()
        threads = []; feats.flush(); b = (e - 1) // args.block; done[b] = True
        np.save(apath, angles); np.save(dpath, done)
        el = time.time() - t0
        print(f'block {b} done ({int(done.sum())}/{NB}) | {n_done / el:.0f} scenes/s | elapsed {el / 60:.1f} min', flush=True)
for th in threads: th.join()
feats.flush(); np.save(apath, angles); np.save(dpath, done)
if done.all():
    assert not np.isnan(angles).any()
    np.save(OUT / 'angles.npy', angles)
    print('COMPLETE', OUT, f'{(time.time() - t0) / 60:.1f} min', flush=True)
