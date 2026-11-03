"""Figure 3b: closed-loop success by cube position on the simulated arm for the unedited policy and for the pan query weights folded to a fixed
estimate of +60 and of -60 degrees (results/arm/closed_loop.npz, 3,000 layouts per condition, seeds shared), over the top-down render results/arm/bg.png with
its mapping results/arm/map.json. Shading is a Gaussian 2.5 cm kernel estimate of the local success rate."""
from common import OUT, RES
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
D = RES / "arm"; Z = np.load(D / "closed_loop.npz")
mp = json.load(open(D / "map.json")); bg = np.asarray(Image.open(D / "bg.png")); sc = mp["scale_px_per_m"]; cx, cy = mp["camera_pos"][:2]; H, W = bg.shape[:2]
def px(x, y): return W / 2 - (np.asarray(y) - cy) * sc, H / 2 - (np.asarray(x) - cx) * sc
COLS = [("unedited", "big_unedited", None), ("fixed estimate $+60°$", "big_plus60", (-60, -20)), ("fixed estimate $-60°$", "big_minus60", (20, 60))]
V0, V1 = 40, 420; sig = 0.025; crop = bg[V0:V1]
uu, vv = np.meshgrid(np.arange(W), np.arange(V0, V1)); gx = cx + (H / 2 - vv) / sc; gy = cy - (uu - W / 2) / sc
def smooth(Pp, ok):
    d2 = (gx[..., None] - Pp[:, 0]) ** 2 + (gy[..., None] - Pp[:, 1]) ** 2; w = np.exp(-d2 / (2 * sig * sig)); m = w.sum(-1)
    return (w * ok).sum(-1) / np.maximum(m, 1e-9), m
plt.rcParams.update({"font.size": 8}); fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.55)); cmap = plt.get_cmap("RdYlGn"); out = {}
for ax, (title, tag, kept) in zip(axs, COLS):
    ok = Z[f"{tag}_success"].astype(float); Pp = Z[f"{tag}_cube_xy"].astype(np.float64); n = len(ok); b = np.degrees(np.arctan2(Pp[:, 1], Pp[:, 0])); rate, mass = smooth(Pp, ok)
    ax.imshow(crop, extent=(0, W, V1, V0)); over = cmap(rate); over[..., 3] = np.where(mass >= 1.5, 0.65, 0.0); ax.imshow(over, extent=(0, W, V1, V0), interpolation="bilinear")
    u, v = px(Pp[:, 0], Pp[:, 1]); ax.scatter(u[ok > 0], v[ok > 0], s=1.5, c="#1a7f1a", edgecolors="none", zorder=3, alpha=.5, label="success"); ax.scatter(u[ok == 0], v[ok == 0], s=4, c="#b00000", marker="x", linewidths=0.4, zorder=3, alpha=.6, label="failure")
    ou, ov = mp["origin"]
    for pan in (-60, -20, 0, 20, 60):
        bb = -pan; u1, v1 = px(0.36 * np.cos(np.radians(bb)), 0.36 * np.sin(np.radians(bb))); u2, v2 = px(0.295 * np.cos(np.radians(bb)), 0.295 * np.sin(np.radians(bb)))
        ax.plot([ou, u1], [ov, v1], ls=":", c="k", lw=0.7, alpha=0.6); ax.text(u2, v2 - 5, f"pan {pan:+d}°" if pan else "pan 0°", fontsize=5.5, clip_on=True, ha="center", va="bottom", bbox=dict(fc="w", ec="none", alpha=0.7, pad=0.8))
    rec = dict(n=int(n), success=float(ok.mean()))
    if kept:
        km = (b >= kept[0]) & (b <= kept[1]); rec.update(kept=float(ok[km].mean()), n_kept=int(km.sum()), elsewhere=float(ok[~km].mean()), n_elsewhere=int((~km).sum()))
        sub = f"kept {ok[km].mean():.2f}, elsewhere {ok[~km].mean():.2f}"
    else: sub = f"{ok.mean():.2f} overall"
    out[tag] = rec; ax.set_xlim(0, W); ax.set_ylim(V1, V0); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{title}\n{sub}", fontsize=8)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)); cb = fig.colorbar(sm, ax=axs, fraction=0.02, pad=0.01, shrink=0.7); cb.set_label("local success rate", fontsize=7); cb.ax.tick_params(labelsize=6)
axs[0].legend(loc="lower left", fontsize=6, frameon=True, markerscale=2.5, handletextpad=0.2)
fig.savefig(OUT / "fig4b_arm.png", dpi=200, bbox_inches="tight"); fig.savefig(OUT / "fig4b_arm.pdf", bbox_inches="tight"); json.dump(out, open(OUT / "fig4b_arm.json", "w"), indent=1); print(out)
