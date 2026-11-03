"""Digit scenes of the pointer test: digits on a ring around the pivot, on the plain table, with no link and no block. No torch needed."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
from toyvla import scene as SCENE
FONT = f"{matplotlib.get_data_path()}/fonts/ttf/DejaVuSans-Bold.ttf"
SS = 2; IMG = 512; PIV = (256.0, 288.0); R0 = 176.0; DIG = [str(d) for d in range(1, 9)]
BG = {"plain": SCENE.background("plain")}
COL = SCENE.COL
BOXSCALE = 1.6; BOXW = 3

def wrap(a): return (np.asarray(a, float) + 180.0) % 360.0 - 180.0
def uv(phi, rad=R0, c=PIV): return c[0] + rad * np.cos(np.radians(phi)), c[1] - rad * np.sin(np.radians(phi))
def tok_of(u, v): return int(min(7, max(0, v // 64))) * 8 + int(min(7, max(0, u // 64)))
def nbhd(t):
    r, c = divmod(t, 8); return [rr * 8 + cc for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2) if 0 <= rr < 8 and 0 <= cc < 8]

# ---------------------------------------------------------------- rendering
_DG = {}
def glyph(d, size):
    if (d, size) not in _DG:
        f = ImageFont.truetype(FONT, 200); im = Image.new("L", (300, 300), 0); ImageDraw.Draw(im).text((20, 0), d, font=f, fill=255); g = im.crop(im.getbbox())
        w, h = g.size; s = size * SS / h; _DG[(d, size)] = g.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    return _DG[(d, size)]
def glyph_xy(d, u, v, size):
    g = glyph(d, size); return g, int(round(u * SS - g.size[0] / 2)), int(round(v * SS - g.size[1] / 2))
def draw_arm(d, theta):
    P = lambda x, y: (float(PIV[0] + x) * SS, float(PIV[1] - y) * SS)
    circ = lambda c, r, **kw: d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], **kw)
    b, h = P(0.0, 0.0), P(R0 * np.cos(np.radians(theta)), R0 * np.sin(np.radians(theta)))
    circ(b, 16 * SS, fill=COL['base'], outline=COL['outline'], width=2 * SS); d.line([b, h], fill=COL['arm'], width=8 * SS)
    circ(b, 8 * SS, fill=COL['joint'], outline=COL['outline'], width=2 * SS); circ(h, 6 * SS, fill=COL['outline'])
BACK = dict(col=(240, 236, 228), scale=1.25)      # light backing square (only when asked; side = scale x size)
def render(objs, size, box=None, bg="plain", backing=False, arm=None):
    """objs: list of (digit, u, v) (u None = not drawn); box: (u, v) of the box centre or None. -> uint8 (512, 512, 3)"""
    im = Image.fromarray(BG[bg].copy()); d = ImageDraw.Draw(im)
    if arm is not None: draw_arm(d, arm)
    if backing:
        h = BACK["scale"] * size / 2 * SS
        for dg, u, v in objs:
            if u is not None: d.rectangle([u * SS - h, v * SS - h, u * SS + h, v * SS + h], fill=BACK["col"])
    for dg, u, v in objs:
        if u is None: continue
        g, x0, y0 = glyph_xy(dg, u, v, size); im.paste(Image.new("RGB", g.size, (0, 0, 0)), (x0, y0), g)
    if box is not None:
        d = ImageDraw.Draw(im); h = BOXSCALE * size / 2 * SS; u, v = box
        d.rectangle([u * SS - h, v * SS - h, u * SS + h, v * SS + h], outline=(0, 0, 0), width=BOXW * SS)
    return np.array(im.resize((IMG, IMG), Image.BOX))
def ink(dg, u, v, size):
    """(64,) fraction of the digit's ink (glyph mask mass on the SS canvas) in each image token."""
    g, x0, y0 = glyph_xy(dg, u, v, size); a = np.asarray(g, np.float32) / 255.0; can = np.zeros((1024 + 400, 1024 + 400), np.float32)
    can[200 + y0:200 + y0 + a.shape[0], 200 + x0:200 + x0 + a.shape[1]] += a; can = can[200:1224, 200:1224]
    t = can.reshape(8, 128, 8, 128).sum((1, 3)).reshape(64); return t / max(a.sum(), 1e-9)
def bbox(dg, u, v, size):
    g, x0, y0 = glyph_xy(dg, u, v, size); return np.array([x0, y0, x0 + g.size[0], y0 + g.size[1]], float) / SS
def overlap(b1, b2, m=2.0): return not (b1[2] + m <= b2[0] or b2[2] + m <= b1[0] or b1[3] + m <= b2[1] or b2[3] + m <= b1[1])

