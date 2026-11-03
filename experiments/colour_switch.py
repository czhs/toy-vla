"""The colour switch. Detector at layer-3 keys: sig_j = ||P (k_j - r_j)|| - m (P: projector on the layer-3 key read directions; r_j: the arm-only reference row; m: median over 32 arm-only
calibration scenes), theta = 99th percentile of sig on those tokens, ramp = 2 theta / theta0; rho_j = x_X . k_j - m_x; on_j = clamp((sig_j - theta) / ramp, 0, 1);
h_j = clamp((rho_j / max(sig_j, 1e-3) - ra) / (rb - ra), 0, 1), ra / rb = 90th percentile on the other colours' tokens / 10th on X's own; G = on (1 - h), T = on h.
Edit at all eight odd layers, keys and values: row - G d + (b - 1) T d, d = P_L (row - r_L); b = 1 for keys, 1 + 0.5 sum(G |d|) / sum(T |d|) for values.
Panel: 2,048 scenes with a red block, a green and a blue square (seed [61060, i], noise 7000 + i), and 512 red-only scenes (seed [61061, i], noise 7500 + i).
Output: results/colour_switch.npz. Runtime a few minutes on a GPU."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, time, numpy as np, torch
from toyvla import query_tools as Q, colour_scenes as CG
from toyvla.paths import DEVICE, RESULTS, GRADIENT_DIRECTIONS, COLOUR_CALIBRATION, SMOKE
V, R = Q.V, Q.V.R; ODD = CG.ODD; GL, GLI, NSL = CG.GL, CG.GLI, CG.NSL
N = 2 if SMOKE else 2048; NP = 2 if SMOKE else 512; BS = 16; FRAC = 0.90; dev = DEVICE; NCAL = 2 if SMOKE else 32; COV_T = 0.05
S, pol, m, H, st = V.setup()
AR = (-180.0, 180.0)
PAL = CG.palette(S); COLS = list(PAL)
A = np.load(COLOUR_CALIBRATION); Kref, Vref = A['Kref'], A['Vref']; x3d = {X: A[f'x3_{X}'] for X in COLS}; mKx = {X: float(A[f'mKx_{X}']) for X in COLS}; theta0 = float(A['theta0'])
G = json.load(open(GRADIENT_DIRECTIONS))
T = lambda a: torch.tensor(np.asarray(a, np.float64), dtype=torch.float32, device=dev); err = lambda y, t: np.abs(V.adiff(y, t))


def read_dirs(kind, L):
    s = np.array(G['s' + kind][str(L)]); D = np.array(G['D' + kind][str(L)])
    if D.shape[0] != len(s) and D.shape[1] == len(s): D = D.T
    k = int(np.searchsorted(np.cumsum(s ** 2) / np.sum(s ** 2), FRAC) + 1); return np.linalg.qr(D[:k].T)[0].T
P = {}
for li, L in enumerate(ODD):
    BK, BV = read_dirs('K', L), read_dirs('V', L); P[L] = dict(k=T(BK.T @ BK), v=T(BV.T @ BV), refk=T(Kref[li][:64]), refv=T(Vref[li][:64]), dk=BK.shape[0], dv=BV.shape[0])

# ---------------------------------------------------------------- 1. calibrate the layer-3 detector (32 calibration scenes)
scs_c, arm_c, imgs_c, cov_c = [], [], {X: [] for X in COLS}, {X: [] for X in COLS}
for k in range(NCAL):
    rng = np.random.default_rng([61050, 0, k]); sc = S.make_scene(rng, 'full'); angs = []
    while len(angs) < 3:
        a = float(rng.uniform(*AR))
        if all(abs(CG.wrap(a - b)) >= 40 for b in angs): angs.append(a)
    arm_only = CG.render_objs(S, sc, 'plain', objs=(), red=False); scs_c.append(sc); arm_c.append(arm_only)
    for X, a in zip(COLS, angs):
        im = CG.render_objs(S, sc, 'plain', objs=[CG.obj('square', *S.block_xy(a), PAL[X][1])], red=False); imgs_c[X].append(im)
        cov_c[X].append(np.any(im != arm_only, -1).reshape(8, 64, 8, 64).mean((1, 3)).reshape(64))
ORDER = ['arm'] + COLS; allsc = [sc for sc in scs_c for _ in ORDER]; allim = [im for k in range(NCAL) for im in [arm_c[k]] + [imgs_c[X][k] for X in COLS]]
K3 = []
for j in range(0, len(allsc), BS):
    kv = V.Embs(H, pol, allsc[j:j + BS], imgs=allim[j:j + BS]).kv(); B = len(allsc[j:j + BS])
    K3.append(kv[GL]['key_states'][:, :64].float().reshape(B, 64, 320).cpu().numpy().astype(np.float64))
K3 = np.concatenate(K3).reshape(NCAL, 4, 64, 320); cov = {X: np.stack(cov_c[X]) for X in COLS}
PK3 = P[GL]['k'].cpu().numpy().astype(np.float64); ref3 = Kref[GLI][:64]
nrm = lambda K: np.linalg.norm((K - ref3[None]) @ PK3, axis=-1)                                   # (n, 64) norm of the read-direction part of the key delta
n0 = nrm(K3[:, 0]).ravel(); m_e = float(np.median(n0)); sig0 = n0 - m_e; theta_e = float(np.percentile(sig0, 99))
ramp_e = 2.0 * theta_e / theta0
x3 = {X: CG.rope_rot(x3d[X]) for X in COLS}                                                        # the directions are applied RoPE-rotated per position
rho_of = lambda K, X: np.einsum('njd,jd->nj', K, x3[X]) - mKx[X]
det = {}
for X in COLS:
    rat = {Y: (rho_of(K3[:, 1 + COLS.index(Y)], X) / np.maximum(nrm(K3[:, 1 + COLS.index(Y)]) - m_e, 1e-3))[cov[Y] > COV_T] for Y in COLS}
    own = rat[X]; oth = np.concatenate([rat[Y] for Y in COLS if Y != X]); ra, rb = float(np.percentile(oth, 90)), float(np.percentile(own, 10)); sep = rb > ra
    if not sep: mid = 0.5 * (ra + rb); ra, rb = mid - 0.01, mid + 0.01
    det[X] = dict(ra=ra, rb=rb, separable=bool(sep), own_med=float(np.median(own)), others_med=float(np.median(oth)), auc=float(CG.auc(own, oth)), own_tokens=int(len(own)))
Q.log('detector', json.dumps(det))

# ---------------------------------------------------------------- 2. the detector and the edit
PK3t, ref3t, m_et = P[GL]['k'], T(ref3), m_e; x3t = {X: T(x3[X]) for X in COLS}; mxt = mKx
def flags(kv, X):
    k3 = kv[GL]['key_states']; B = k3.shape[0]; k3 = k3[:, :64].float().reshape(B, 64, 320)
    sig = ((k3 - ref3t[None]) @ PK3t).norm(dim=-1) - m_et; rho = torch.einsum('bjd,jd->bj', k3, x3t[X]) - mxt[X]
    on = ((sig - theta_e) / ramp_e).clamp(0, 1); h = ((rho / sig.clamp_min(1e-3) - det[X]['ra']) / (det[X]['rb'] - det[X]['ra'])).clamp(0, 1)
    return on * (1 - h), on * h


class ColourSwitch:
    def __init__(self): self.active = False; self.adapt = 0.5; self.G = None; self.Tg = None; self.handles = []
    def edit(self, L, x, side):
        xi = x[:, :64].float(); Pm = P[L][side]; ref = P[L]['ref' + side][None]; d = (xi - ref) @ Pm; Gc, Tg = self.G[..., None], self.Tg[..., None]
        nrm_ = d.norm(dim=-1); ratio = (Gc[..., 0] * nrm_).sum(1) / (Tg[..., 0] * nrm_).sum(1).clamp_min(1e-3)
        b = (1.0 + self.adapt * ratio)[:, None, None] if side == 'v' else 1.0
        out = xi - Gc * d + (b - 1.0) * Tg * d
        x2 = x.clone(); x2[:, :64] = out.to(x.dtype); return x2
    def attach(self, m):
        ex = m.vlm_with_expert.lm_expert
        for L in ODD:
            att = ex.layers[L].self_attn
            def kh(mod, args, L=L):
                x = args[0]
                if not self.active or x.dim() != 3 or x.shape[1] < 64: return None
                return (self.edit(L, x, 'k'),) + tuple(args[1:])
            def vh(mod, args, L=L):
                x = args[0]
                if not self.active or x.dim() != 3 or x.shape[1] < 64: return None
                return (self.edit(L, x, 'v'),) + tuple(args[1:])
            self.handles += [att.k_proj.register_forward_pre_hook(kh), att.v_proj.register_forward_pre_hook(vh)]
        return self


sw = ColourSwitch().attach(m)
def run(E, kv, noise, X):
    sw.G, sw.Tg = flags(kv, X); sw.active = True
    try: y, _ = Q.flow_eval(m, H, E.pm, kv, noise)
    finally: sw.active = False
    return y
scenes = [CG.compose(S, np.random.default_rng([61060, i]), PAL, k=2, lo=AR[0], hi=AR[1]) for i in range(N)]
present = np.zeros((N, 3), bool); angle = np.full((N, 3), np.nan)
for i, (sc_, sq_, objs) in enumerate(scenes):
    present[i, 0] = True; angle[i, 0] = sc_['phi']
    for (c, a) in sq_: present[i, COLS.index(c)] = True; angle[i, COLS.index(c)] = a
Y0 = np.zeros(N); Y = np.zeros((3, N)); t0 = time.time()
for j in range(0, N, BS):
    sub = scenes[j:j + BS]; B = len(sub); E = V.Embs(H, pol, [s_ for s_, _, _ in sub], imgs=[CG.render_objs(S, s_, 'plain', objs=o) for s_, _, o in sub]); kv = E.kv()
    noise = R.noise_per_scene(range(7000 + j, 7000 + j + B)); sw.active = False; Y0[j:j + B], _ = Q.flow_eval(m, H, E.pm, kv, noise)
    for ci, X in enumerate(COLS):
        Y[ci, j:j + B] = run(E, kv, noise, X)
    if (j // BS) % 16 == 0: Q.log(f'panel {j + B}/{N} {time.time() - t0:.0f}s', Q.peak())
pl = [CG.compose(S, np.random.default_rng([61061, i]), PAL, k=0) for i in range(NP)]; P0 = np.zeros(NP); PG = np.zeros(NP); ptgt = np.array([s_['phi'] for s_, _, _ in pl])
for j in range(0, NP, BS):
    sub = pl[j:j + BS]; B = len(sub); E = V.Embs(H, pol, [s_ for s_, _, _ in sub], imgs=[CG.render_objs(S, s_, 'plain', objs=()) for s_, _, _ in sub]); kv = E.kv()
    noise = R.noise_per_scene(range(7500 + j, 7500 + j + B)); sw.active = False; P0[j:j + B], _ = Q.flow_eval(m, H, E.pm, kv, noise)
    PG[j:j + B] = run(E, kv, noise, 'red')
np.savez(RESULTS / 'colour_switch.npz', cols=np.array(COLS), Y0=Y0, Y=Y, angle=angle, present=present, P0=P0, PG=PG, ptgt=ptgt)
Q.log('median error to the chosen square', {PAL[X][0]: round(float(np.median(err(Y[ci], angle[:, ci]))), 2) for ci, X in enumerate(COLS)}, 'red-only', round(float(np.median(err(PG, ptgt))), 2), 'intact', round(float(np.median(err(P0, ptgt))), 2))
