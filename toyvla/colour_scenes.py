"""Scenes with extra coloured squares and the colour constants of the colour-switch experiment."""
import colorsys, numpy as np
from PIL import Image, ImageDraw

ODD = list(range(1, 16, 2)); GL = 3; GLI = ODD.index(GL)                 # the detector reads the keys of layer 3
NSL = 113                                                                # prefix slots: 64 image, 9 prompt tokens, padding, state at 112


def rope_rot(d, n=64):
    """content-space direction(s) d (..., 320) -> (n, ..., 320) the direction at slot j in post-RoPE key space (lerobot apply_rope,
    per KV head of 64: pairs (i, i + 32), w_i = 1e4^(-2 i / 64))."""
    d = np.asarray(d, np.float64); w = 10000.0 ** (-(2.0 / 64) * np.arange(32)); out = np.zeros((n,) + d.shape)
    x = d.reshape(d.shape[:-1] + (5, 64))
    for j in range(n):
        c, s = np.cos(j * w), np.sin(j * w); a, b = x[..., :32], x[..., 32:]
        out[j] = np.concatenate([a * c - b * s, b * c + a * s], -1).reshape(d.shape)
    return out


def rope_back(K):
    """post-RoPE image-slot keys K (n, 64, 320) -> content keys R_j^T k_j (inverse of rope_rot's per-slot rotation)."""
    K = np.asarray(K, np.float64); n = K.shape[1]; w = 10000.0 ** (-(2.0 / 64) * np.arange(32)); x = K.reshape(K.shape[:-1] + (5, 64)); out = np.empty_like(x)
    for j in range(n):
        c, s_ = np.cos(j * w), np.sin(j * w); a, b = x[:, j, :, :32], x[:, j, :, 32:]
        out[:, j, :, :32] = a * c + b * s_; out[:, j, :, 32:] = b * c - a * s_
    return out.reshape(K.shape)


def hsv(h, s=0.864, v=0.863):
    r, g, b = colorsys.hsv_to_rgb((h % 360) / 360.0, s, v); return (int(round(255 * r)), int(round(255 * g)), int(round(255 * b)))


def palette(S): return {'red': ('red', tuple(S.COL['red'])), 'hue120': ('green', hsv(120)), 'hue240': ('blue', hsv(240))}


def auc(a, b):
    x = np.r_[a, b]; rk = x.argsort().argsort() + 1.0
    return float((rk[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


MIN_SEP = 40.0


def wrap(a): return (np.asarray(a, float) + 180) % 360 - 180


def obj(kind, x, y, col, **kw): d = dict(kind=kind, x=float(x), y=float(y), col=tuple(int(c) for c in col)); d.update(kw); return d


def render_objs(S, sc, bg='plain', objs=(), red=True, arm=True, red_col=None):
    base = bg if isinstance(bg, np.ndarray) else S.background(bg)
    im = Image.fromarray(np.ascontiguousarray(base, dtype=np.uint8)); d = ImageDraw.Draw(im); SS = S.SS
    P = lambda x, y: tuple(float(z) * SS for z in S.w2uv(x, y))
    circ = lambda c, r, **kw: d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], **kw)
    if arm:
        b, h = P(0.0, 0.0), P(*S.block_xy(sc['t1s']))
        circ(b, 16 * SS, fill=S.COL['base'], outline=S.COL['outline'], width=2 * SS)
        d.line([b, h], fill=S.COL['arm'], width=8 * SS)
        circ(b, 8 * SS, fill=S.COL['joint'], outline=S.COL['outline'], width=2 * SS)
        circ(h, 6 * SS, fill=S.COL['outline'])
    for o in objs:
        x, y, hh = o['x'], o['y'], o.get('half', S.HALF)
        d.polygon([P(x + a, y + b_) for a, b_ in ((-hh, -hh), (hh, -hh), (hh, hh), (-hh, hh))], fill=o['col'])
    if red:
        x, y, H_ = sc['px'], sc['py'], S.HALF
        d.polygon([P(x + a, y + b_) for a, b_ in ((-H_, -H_), (H_, -H_), (H_, H_), (-H_, H_))], fill=tuple(red_col or S.COL['red']))
    return np.array(im.resize((S.IMG, S.IMG), Image.BOX))


def compose(S, rng, PAL, k=2, cols=None, min_sep=MIN_SEP, lo=-180.0, hi=180.0):
    """S.make_scene(rng, 'full') + k extra squares of distinct non-red palette colours (or the given cols) at ring angles >= min_sep apart
    from the block and each other. Returns sc, [(cond, angle)], objs."""
    sc = S.make_scene(rng, 'full'); names = [c for c in PAL if c != 'red']
    cols = list(cols) if cols is not None else [names[i] for i in rng.choice(len(names), k, replace=False)]
    angs = [sc['phi']]
    while len(angs) < 1 + len(cols):
        a = float(rng.uniform(lo, hi))
        if all(abs(wrap(a - b)) >= min_sep for b in angs): angs.append(a)
    sq = list(zip(cols, angs[1:])); objs = [obj('square', *S.block_xy(a), PAL[c][1]) for c, a in sq]
    return sc, sq, objs
