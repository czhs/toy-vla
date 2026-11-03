"""The one-square distractor and its read-direction share. Scene with a green or blue square D and the same scene without it R; the expert's k_proj / v_proj inputs at
the 64 image tokens of all eight odd layers are assembled from the two caches: intact (D everywhere), nodist_cache (R everywhere), erase_read_dirs (D - P (D - R)),
keep_read_dirs (R + P (D - R)), keep_random (R + P_rand (D - R)), P the projector on the read directions. 256 scenes, noise 7700 + scene.
Output: results/distractor_one_square.npz. Runtime about 1 min on a GPU (2.3 GB)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, time, numpy as np, torch
from toyvla import query_tools as Q, colour_scenes as CG
from toyvla.paths import DEVICE, RESULTS, GRADIENT_DIRECTIONS, SMOKE
V, R = Q.V, Q.V.R; ODD = CG.ODD; NSL = CG.NSL
N = 2 if SMOKE else 256; BS = 16; FRAC = 0.90
S, pol, m, H, st = V.setup()
AR = (-180.0, 180.0)
PAL = CG.palette(S); DIST = ['hue120', 'hue240']
G = json.load(open(GRADIENT_DIRECTIONS))
dev = DEVICE; T = lambda a: torch.tensor(np.asarray(a, np.float64), dtype=torch.float32, device=dev); err = lambda y, t: np.abs(V.adiff(y, t)); rng_u = np.random.default_rng(61081)


def read_dirs(kind, L):
    s = np.array(G['s' + kind][str(L)]); D = np.array(G['D' + kind][str(L)])
    if D.shape[0] != len(s) and D.shape[1] == len(s): D = D.T
    k = int(np.searchsorted(np.cumsum(s ** 2) / np.sum(s ** 2), FRAC) + 1); return np.linalg.qr(D[:k].T)[0].T
def randsub(k): return np.linalg.qr(rng_u.standard_normal((320, k)))[0].T
P = {}
for L in ODD:
    BK, BV = read_dirs('K', L), read_dirs('V', L); RK, RV = randsub(BK.shape[0]), randsub(BV.shape[0])
    P[L] = dict(k=T(BK.T @ BK), v=T(BV.T @ BV), rk=T(RK.T @ RK), rv=T(RV.T @ RV))
IMG = np.arange(64); TXTS = np.arange(64, NSL)
# condition = which image tokens come from D ('all' | 'none'), text rows from D (bool), projection spec ('keep' | 'erase', 'all', 'read_dirs' | 'rand')
CONDS = {}
def cond(name, src_D_img, src_D_txt, proj=None): CONDS[name] = dict(img=src_D_img, txt=src_D_txt, proj=proj)
cond('intact', 'all', True); cond('nodist_cache', 'none', False)
cond('keep_read_dirs', 'none', False, ('keep', 'all', 'read_dirs')); cond('erase_read_dirs', 'all', True, ('erase', 'all', 'read_dirs')); cond('keep_random', 'none', False, ('keep', 'all', 'rand'))


class Assemble:
    def __init__(self): self.active = False; self.c = None; self.Dc = None; self.Rc = None; self.handles = []
    def tokmask(self, which, B):
        if which == 'all': return torch.ones(B, 64, dtype=torch.bool, device=dev)
        if which == 'none': return torch.zeros(B, 64, dtype=torch.bool, device=dev)
        raise ValueError(which)
    def edit(self, L, x, side):
        c = CONDS[self.c]; Dr, Rr = self.Dc[L][side], self.Rc[L][side]; B = x.shape[0]
        mimg = self.tokmask(c['img'], B); src = torch.zeros(B, NSL, dtype=torch.bool, device=dev); src[:, :64] = mimg; src[:, 64:] = bool(c['txt'])
        out = torch.where(src[..., None], Dr, Rr)
        if c['proj'] is not None:
            kind, which, sub = c['proj']; Pm = P[L][('k' if side == 'k' else 'v') if sub == 'read_dirs' else ('rk' if side == 'k' else 'rv')]
            tm = self.tokmask(which, B)[..., None]; delta = (Dr[:, :64] - Rr[:, :64]) @ Pm
            base = Rr[:, :64] + delta if kind == 'keep' else Dr[:, :64] - delta
            out = out.clone(); out[:, :64] = torch.where(tm, base, out[:, :64])
        x2 = x.clone(); x2[:, :NSL] = out.to(x.dtype); return x2
    def attach(self, m):
        ex = m.vlm_with_expert.lm_expert
        for L in ODD:
            att = ex.layers[L].self_attn
            def kh(mod, args, L=L):
                x = args[0]
                if not self.active or x.dim() != 3 or x.shape[1] < NSL: return None
                return (self.edit(L, x, 'k'),) + tuple(args[1:])
            def vh(mod, args, L=L):
                x = args[0]
                if not self.active or x.dim() != 3 or x.shape[1] < NSL: return None
                return (self.edit(L, x, 'v'),) + tuple(args[1:])
            self.handles += [att.k_proj.register_forward_pre_hook(kh), att.v_proj.register_forward_pre_hook(vh)]
        return self


asm = Assemble().attach(m)
scenes = [CG.compose(S, np.random.default_rng([61070, i]), PAL, k=1, cols=[DIST[i % 2]], lo=AR[0], hi=AR[1]) for i in range(N)]
phi = np.array([sc['phi'] for sc, _, _ in scenes]); dang = np.array([sq[0][1] for _, sq, _ in scenes]); dcol = np.array([sq[0][0] for _, sq, _ in scenes])
cover = np.zeros((N, 64), np.float32)
for i, (sc, sq, objs) in enumerate(scenes):
    full = CG.render_objs(S, sc, 'plain', objs=objs); without = CG.render_objs(S, sc, 'plain', objs=())
    cover[i] = np.any(full != without, -1).reshape(8, 64, 8, 64).mean((1, 3)).reshape(64)
Y = {c: np.zeros(N) for c in CONDS}; Y['nodist_scene'] = np.zeros(N); t0 = time.time()
rows = lambda kv, L: {'k': kv[L]['key_states'][:, :NSL].float().reshape(-1, NSL, 320), 'v': kv[L]['value_states'][:, :NSL].float().reshape(-1, NSL, 320)}
for j in range(0, N, BS):
    sub = scenes[j:j + BS]; B = len(sub); scs = [sc for sc, _, _ in sub]
    imgs = [CG.render_objs(S, sc, 'plain', objs=o) for sc, _, o in sub]; imgs_ref = [CG.render_objs(S, sc, 'plain', objs=()) for sc, _, _ in sub]
    noise = R.noise_per_scene(range(7700 + j, 7700 + j + B))
    E_ref = V.Embs(H, pol, scs, imgs=imgs_ref); kv_ref = E_ref.kv(); asm.active = False; Y['nodist_scene'][j:j + B], _ = Q.flow_eval(m, H, E_ref.pm, kv_ref, noise)
    E = V.Embs(H, pol, scs, imgs=imgs); kv = E.kv()
    asm.Dc = {L: rows(kv, L) for L in ODD}; asm.Rc = {L: rows(kv_ref, L) for L in ODD}
    for c in CONDS:
        asm.c = c; asm.active = True
        try: Y[c][j:j + B], _ = Q.flow_eval(m, H, E.pm, kv, noise)
        finally: asm.active = False
    Q.log(f'{j + B}/{N} {time.time() - t0:.0f}s', Q.peak())
np.savez(RESULTS / 'distractor_one_square.npz', **{f'y_{c}': Y[c] for c in Y}, phi=phi, dang=dang, dcol=dcol, cover=cover)
Q.log('median error to the block', {c: round(float(np.median(err(Y[c], phi))), 2) for c in Y})
