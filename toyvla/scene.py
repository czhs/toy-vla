"""The toy task: one link rotating about a fixed pivot and a red block on the circle traced by its tip.

World frame: pivot at the origin, x right, y up; 1 unit = 1 px: pixel (u, v) = (256 + x, 288 - y). The pivot sits half a token below the
image centre, so the wrap line (phi = +-180, the negative x axis, v = 288) runs through the middle of token row 4 instead of along a token
boundary and the wrap cannot be read off token identity.
Link: length R = 176, drawn at the start angle theta_s. Block: 22 x 22 px, centred on the circle at angle phi. Target theta = phi (canonical
[-180, 180)), state coordinate 0 = theta_s / 180, action coordinate 0 = theta / 180 at all 50 chunk positions, coordinates 1..31 unused.
phi, theta_s ~ U[-180, 180), independent. Training uses split 'full' (no exclusions); the trainer also logs the panels H1, H2, HC, HCrev of a
held-out design ('token hole', 'small hole', combination of block and start ranges) and the camera shifts below. Backgrounds: 'plain' (uniform
table) or 'struct' (fixed checkerboard table with tape, a disk and a rectangle; only an evaluation panel of the trainer uses it)."""
import numpy as np
from PIL import Image, ImageDraw

NJ = 1                                             # joints used in state/action coords
IMG, SS = 512, 2
BASE_UV = (256.0, 288.0)                           # pivot pixel
R = 176.0
HALF = 11.0
MIN_SEP_DEG = 20.0
P_DISTRACTOR = 0.5
PROMPT = "Move the arm to the red block."
COL = dict(table=(214, 202, 178), tile=(158, 142, 116), tape=(248, 248, 248), green=(60, 150, 80), yellow=(225, 195, 50),
           red=(220, 30, 30), blue=(30, 70, 220), black=(25, 25, 25), arm=(60, 60, 60), joint=(150, 150, 150), outline=(35, 35, 35), base=(95, 95, 95))
TILE, TILE_OFF = 48.0, (7.0, 13.0)
TAPE_Y, TAPE_W = 100.0, 5.0
GREEN_DISK = ((R * np.cos(np.radians(-135)), R * np.sin(np.radians(-135))), 26.0)
YELLOW_RECT = (120.0, -75.0, 200.0, -35.0)
BLACK_PHI = 135.0                                  # optional fixed black block on the ring
H1_TOKEN, H1_MARGIN = (2, 6), 2.0
H2_C, H2_W = -110.0, 4.0
HC_BLOCK, HC_START = (90.0, 180.0), (-90.0, 0.0)
POOL_SEED, POOL = 20260925, 500_000
SEEDS = dict(val=2101, H1=2102, H2=2103, HC=2104, HCrev=2105, cam=2106, disc=2201)

wrap = lambda a: (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def block_xy(phi): return R * np.cos(np.radians(phi)), R * np.sin(np.radians(phi))
def w2uv(x, y): return BASE_UV[0] + np.asarray(x), BASE_UV[1] - np.asarray(y)
def uv2w(u, v): return np.asarray(u) - BASE_UV[0], BASE_UV[1] - np.asarray(v)


def _touches_token(phi, tok, margin):
    u, v = w2uv(*block_xy(phi)); r, c = tok; m = HALF + margin
    return (u + m > 64 * c) & (u - m < 64 * (c + 1)) & (v + m > 64 * r) & (v - m < 64 * (r + 1))


_g = np.arange(-180, 180, 0.01); _h = _g[_touches_token(_g, H1_TOKEN, H1_MARGIN)]
H1_ARC = (float(_h.min()), float(_h.max()))        # contiguous arc of excluded phi (checked in self-test)


def in_h1(phi): phi = wrap(phi); return (phi >= H1_ARC[0]) & (phi <= H1_ARC[1])
def in_h2(phi): return np.abs(wrap(np.asarray(phi) - H2_C)) <= H2_W
def free(phi): return ~in_h1(phi) & ~in_h2(phi)
def in_hc(phi, ts):
    phi, ts = wrap(phi), wrap(ts)
    return (phi >= HC_BLOCK[0]) & (phi < HC_BLOCK[1]) & (ts >= HC_START[0]) & (ts < HC_START[1])
def in_hcrev(phi, ts):
    phi, ts = wrap(phi), wrap(ts)
    return (phi >= HC_START[0]) & (phi < HC_START[1]) & (ts >= HC_BLOCK[0]) & (ts < HC_BLOCK[1])


def _u(rng, lo=-180.0, hi=180.0): return float(rng.uniform(lo, hi))


def rand_nonred(rng):
    """Random non-red colour: 20% greys (level 20-235, black to white), else HSV with hue in
    [40, 320] deg (at least 40 deg from red), saturation 0.5-1, value 0.45-1. Returns an (r, g, b) int tuple."""
    import colorsys
    if rng.random() < 0.2:
        g = int(rng.integers(20, 236)); return (g, g, g)
    h, sat, val = rng.uniform(40, 320) / 360.0, rng.uniform(0.5, 1.0), rng.uniform(0.45, 1.0)
    return tuple(int(round(255 * c)) for c in colorsys.hsv_to_rgb(h, sat, val))


def scene_at(phi, t1s, phi_q=np.nan, has_d=False, split='custom', qcol=None):
    phi, t1s = float(wrap(phi)), float(wrap(t1s)); px, py = block_xy(phi); p0x, p0y = block_xy(t1s)
    qx, qy = (block_xy(phi_q) if has_d else (np.nan, np.nan))
    return dict(split=split, phi=phi, px=float(px), py=float(py), t1s=t1s, t2s=0.0, p0x=float(p0x), p0y=float(p0y),
                t1=phi, t2=0.0, qx=float(qx), qy=float(qy), phi_q=float(phi_q) if has_d else np.nan, has_d=bool(has_d),
                qcol=(tuple(int(c) for c in qcol) if qcol is not None else None))


def make_scene(rng, split='train'):
    """split 'full': no exclusions (the policies of the paper).
    'train'/'val': the held-out design (no H1/H2 blocks, no HC combo). H1/H2/HC/HCrev: region panels."""
    while True:
        if split == 'full': phi, ts = _u(rng), _u(rng); break
        if split in ('train', 'val'):
            phi, ts = _u(rng), _u(rng)
            if not free(phi) or in_hc(phi, ts): continue
        elif split == 'H1': phi, ts = _u(rng, *H1_ARC), _u(rng)
        elif split == 'H2': phi, ts = _u(rng, H2_C - H2_W, H2_C + H2_W), _u(rng)
        elif split == 'HC': phi, ts = _u(rng, *HC_BLOCK), _u(rng, *HC_START)
        elif split == 'HCrev': phi, ts = _u(rng, *HC_START), _u(rng, *HC_BLOCK)
        else: raise ValueError(split)
        if split in ('H1', 'H2') and in_hc(phi, ts): continue
        if split in ('HC', 'HCrev') and not free(phi): continue
        break
    has_d = bool(rng.random() < P_DISTRACTOR)
    while True:
        pq = _u(rng)
        if (split == 'full' or free(pq)) and abs(float(wrap(pq - phi))) >= MIN_SEP_DEG: break
    qcol = rand_nonred(rng)
    return scene_at(phi, ts, pq, has_d, split, qcol=qcol)


def pool_scene(i, split='train'): return make_scene(np.random.default_rng([POOL_SEED, int(i)]), split)


def panel(name, n, split=None, seeds=SEEDS):
    split = split or {'val': 'val', 'cam': 'val', 'disc': 'val'}.get(name, name)
    return [make_scene(np.random.default_rng([seeds[name], k]), split) for k in range(n)]


def text_prompt(sc):
    u, v = w2uv(sc['px'], sc['py'])
    return f"Move the arm to the red block at ({float(u):05.1f}, {float(v):05.1f})."


# ---------------------------------------------------------------- metrics
def angle_err(pred_deg, target_deg):
    """circular |error| in degrees; pred (..., NJ), target broadcastable."""
    return np.abs(wrap(np.asarray(pred_deg)[..., :NJ] - np.asarray(target_deg)[..., :NJ]))


def ee_err(pred_deg, scs):
    """pred (n, T, >=1) or (n, >=1) degrees. Tip positions are averaged over the T action positions (vector mean, wrap-safe);
    returns (distance of the mean tip to the block centre in px, success = mean tip inside the block square)."""
    p = np.asarray(pred_deg, float)
    if p.ndim == 2: p = p[:, None]
    th = np.radians(p[..., 0]); tx, ty = (R * np.cos(th)).mean(1), (R * np.sin(th)).mean(1)
    px = np.array([s['px'] for s in scs]); py = np.array([s['py'] for s in scs])
    return np.hypot(tx - px, ty - py), (np.abs(tx - px) <= HALF) & (np.abs(ty - py) <= HALF)


def outside_canonical(pred_deg):
    p = np.asarray(pred_deg)[..., 0]; return (p < -180) | (p >= 180)


# ---------------------------------------------------------------- camera + rendering
def _cam(cam):
    if cam is None: return None
    cam = tuple(float(c) for c in cam)
    return None if cam == (0.0, 0.0, 0.0, 1.0) else cam


def cam_fwd(u0, v0, cam):
    if cam is None: return u0, v0
    du, dv, rot, s = cam; c, sn = np.cos(np.radians(rot)), np.sin(np.radians(rot)); C = IMG / 2
    a, b = np.asarray(u0) - C, np.asarray(v0) - C
    return C + s * (c * a - sn * b) + du, C + s * (sn * a + c * b) + dv


def cam_inv(u, v, cam):
    if cam is None: return u, v
    du, dv, rot, s = cam; c, sn = np.cos(np.radians(rot)), np.sin(np.radians(rot)); C = IMG / 2
    a, b = (np.asarray(u) - C - du) / s, (np.asarray(v) - C - dv) / s
    return C + c * a + sn * b, C - sn * a + c * b


_BG = {}


def background(bg, cam=None):
    cam = _cam(cam); key = (bg, cam)
    if key not in _BG:
        n = IMG * SS
        if bg == 'plain':
            im = np.empty((n, n, 3), np.uint8); im[:] = COL['table']
        elif bg == 'struct':
            g = (np.arange(n) + 0.5) / SS; U, V = np.meshgrid(g, g)
            x, y = uv2w(*cam_inv(U, V, cam))
            tile = ((np.floor((x - TILE_OFF[0]) / TILE) + np.floor((y - TILE_OFF[1]) / TILE)) % 2).astype(bool)
            im = np.where(tile[..., None], np.array(COL['tile'], np.uint8), np.array(COL['table'], np.uint8))
            im[np.abs(y - TAPE_Y) <= TAPE_W / 2] = COL['tape']
            (gx, gy), gr = GREEN_DISK; im[np.hypot(x - gx, y - gy) <= gr] = COL['green']
            x0, y0, x1, y1 = YELLOW_RECT; im[(x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)] = COL['yellow']
            im = np.ascontiguousarray(im, dtype=np.uint8)
        else: raise ValueError(bg)
        if len(_BG) >= 24: _BG.clear()
        _BG[key] = im
    return _BG[key].copy()


def render(sc, bg='plain', cam=None, blocks=True, arm=True, red=True, blue=False, black=False, dcol='blue'):
    """uint8 (512, 512, 3). Link drawn at the START angle; blocks drawn on top. blue=True only in the distractor suite.
    black=True: a fixed black block on the ring at BLACK_PHI, part of the background (drawn before the arm; the red block covers it).
    dcol: distractor colour when blue=True: 'blue' (default) or 'random' (the scene's own random non-red colour sc['qcol'])."""
    cam = _cam(cam); s = 1.0 if cam is None else cam[3]
    im = Image.fromarray(background(bg, cam)); d = ImageDraw.Draw(im)
    P = lambda x, y: tuple(float(z) * SS for z in cam_fwd(*w2uv(x, y), cam))
    circ = lambda c, r, **kw: d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], **kw)
    if black:
        bx, by = block_xy(BLACK_PHI)
        d.polygon([P(bx + a, by + b_) for a, b_ in ((-HALF, -HALF), (HALF, -HALF), (HALF, HALF), (-HALF, HALF))], fill=COL['black'])
    if arm:
        b, h = P(0.0, 0.0), P(*block_xy(sc['t1s']))
        circ(b, 16 * SS * s, fill=COL['base'], outline=COL['outline'], width=2 * SS)
        d.line([b, h], fill=COL['arm'], width=max(1, int(round(8 * SS * s))))
        circ(b, 8 * SS * s, fill=COL['joint'], outline=COL['outline'], width=2 * SS)
        circ(h, 6 * SS * s, fill=COL['outline'])
    if blocks:
        qc = COL['blue'] if dcol == 'blue' else tuple(sc['qcol'])
        todo = ([(sc['qx'], sc['qy'], qc)] if (sc['has_d'] and blue) else []) + ([(sc['px'], sc['py'], COL['red'])] if red else [])
        for x, y, col in todo:
            d.polygon([P(x + a, y + b_) for a, b_ in ((-HALF, -HALF), (HALF, -HALF), (HALF, HALF), (-HALF, HALF))], fill=col)
    return np.array(im.resize((IMG, IMG), Image.BOX))


def token_of(phi, cam=None):
    u, v = cam_fwd(*w2uv(*block_xy(phi)), _cam(cam)); return int(np.floor(v / 64)), int(np.floor(u / 64))


def phi_grid(n=360, exclude_holes=True, offset=0.0):
    g = wrap(np.arange(n) * 360.0 / n - 180.0 + offset)
    return g[free(g)] if exclude_holes else g


if __name__ == '__main__':
    g = np.arange(-180, 180, 0.01); h = g[_touches_token(g, H1_TOKEN, H1_MARGIN)]
    assert np.all(np.diff(h) < 0.02), 'H1 arc not contiguous'
    print('H1 arc', H1_ARC, 'width %.1f deg' % (H1_ARC[1] - H1_ARC[0]))
    scs = [pool_scene(i) for i in range(20000)]
    ph = np.array([s['phi'] for s in scs]); ts = np.array([s['t1s'] for s in scs])
    assert free(ph).all() and not in_hc(ph, ts).any()
    toks = np.array([token_of(p) for p in ph]); print('train blocks in H1 token centre-cell', ((toks[:, 0] == 2) & (toks[:, 1] == 6)).sum())
    assert not _touches_token(ph, H1_TOKEN, 0.0).any(), 'a training block touches the H1 token'
    print('train phi range', ph.min(), ph.max(), 'wrap-adjacent (|phi|>175) frac %.3f' % (np.abs(ph) > 175).mean())
    for sp in ('H1', 'H2', 'HC', 'HCrev'):
        pp = panel(sp, 300); a = np.array([s['phi'] for s in pp]); b = np.array([s['t1s'] for s in pp])
        print(sp, 'phi [%.1f, %.1f]' % (a.min(), a.max()), 'ts [%.1f, %.1f]' % (b.min(), b.max()), 'inH1 %.2f inH2 %.2f inHC %.2f' % (in_h1(a).mean(), in_h2(a).mean(), in_hc(a, b).mean()))
    print('prompt lengths', {len(text_prompt(s)) for s in scs[:3000]}, text_prompt(scs[0]))
    e, ok = ee_err(np.array([[[s['phi']]] * 50 for s in scs[:100]]).reshape(100, 50, 1), scs[:100]); print('self ee err', e.max(), ok.mean())
    print('token rows of wrap-line blocks (|phi|>177):', sorted({token_of(p) for p in ph[np.abs(ph) > 177]}))
