"""Figure 1: (a) the toy task, (b) the same scene with a green and a blue square, (c) the pointer test's digit scene with the question, (d) the simulated arm's first observation
(results/arm/fig1_arm_panel.png, rendered by figures/make_fig1d_arm_panel.py in the arm environment). Panels are 512 px, composed with PIL. The label font is matplotlib's DejaVu Sans."""
from common import OUT, RES
import numpy as np, matplotlib
from PIL import Image, ImageDraw, ImageFont
from toyvla import scene as S, digit_scenes as DS
PHI, START = 60.0, 30.0
sc = S.scene_at(PHI, START); a = Image.fromarray(S.render(sc)); b = a.copy(); db = ImageDraw.Draw(b); db.rectangle([150, 120, 172, 142], fill=(30, 220, 30)); db.rectangle([400, 370, 422, 392], fill=(30, 30, 220))
ring = ['5', '2', '3', '8', '4', '7', '6', '1']
c = Image.fromarray(DS.render([(dg, *map(float, DS.uv(45.0 * i - 0.55))) for i, dg in enumerate(ring)], 52))
dp = Image.open(RES / 'arm' / 'fig1_arm_panel.png').convert('RGB')
font = lambda n: ImageFont.truetype(f"{matplotlib.get_data_path()}/fonts/ttf/DejaVuSans.ttf", n)
P, GAP, TOP, BOT = 512, 28, 150, 70
W = 4 * P + 3 * GAP; fig = Image.new('RGB', (W, TOP + P + BOT), 'white'); d = ImageDraw.Draw(fig)
for i, (im, lab) in enumerate(((a, '(a) task'), (b, '(b) with distractors'), (c, '(c) pointer test'), (dp, '(d) SO-101 arm'))):
    x = i * (P + GAP); fig.paste(im.resize((P, P)), (x, TOP)); w = d.textlength(lab, font=font(32)); d.text((x + (P - w) / 2, TOP + P + 14), lab, fill='black', font=font(32))
x0 = 2 * (P + GAP) - 30; box = [x0, 18, x0 + 572, 124]
d.rounded_rectangle(box, radius=22, fill='white', outline='black', width=2); d.polygon([(x0 + 70, 122), (x0 + 74, 150), (x0 + 104, 122)], fill='white'); d.line([(x0 + 70, 124), (x0 + 74, 150), (x0 + 104, 124)], fill='black', width=2)
d.text((x0 + 22, 28), 'Which number is marked in this image?\nAnswer with the number.', fill='black', font=font(26), spacing=8)
fig.save(OUT / 'fig1_overview.png'); print('wrote', OUT / 'fig1_overview.png', fig.size)
