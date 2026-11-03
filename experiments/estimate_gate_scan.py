"""Scan of the stream's estimate. The estimate coordinate of the query input at layers 5-15 and the last flow step is set to an absolute angle A on a 15-degree grid, rows keep their
offsets, and per head the attention to the block's tile is recorded. Conditions: intact; identity check (plane rotated by 0); tmpl A (row-mean coordinate := fit-panel mean
coordinate of scenes near A); line A (the least-squares line of the coordinate on phi); rand A (the tmpl edit in a random 2-D subspace, a short list of A).
Panels: 128 scenes with |phi| >= 90 (seed [61003, i], noise 10000 + i) + 48 scenes (seed [61002, i], noise 7000 + i); fit 96 (seed [61004, i], noise 11000 + i).
Output: data/estimate_gate_scan/{attention.npy, answers.npy, meta.json}, read by experiments/head_gating.py. Runtime about 2 min on a GPU (2.1 GB)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, time, numpy as np, torch
from toyvla import query_tools as Q, policy_tools
from toyvla.paths import DEVICE, DATA, RESULTS, ESTIMATE_FIT, SMOKE
from lerobot.policies.smolvla.smolvlm_with_expert import apply_rope
V, R = Q.V, Q.R
BS = int(os.environ.get("BATCH", 16)); OD = DATA / "estimate_gate_scan"; OD.mkdir(parents=True, exist_ok=True)
S, pol, m, H, st = V.setup()
NS, ODD = Q.NS, Q.ODD; dev = DEVICE; POS = torch.arange(50, device=dev)[None]; EL = ODD[2:]            # edited layers 5..15
ENDS_ABS = 90.0; D = 720
Q.log("edited / recorded layers", EL)


class P:
    X = {}; edit = None; rec = None; idmode = False; qdiff = []


def resid_fn(L, k, h):
    if L % 2 == 1: P.X[L] = h.clone()
    return None
H.resid_fn = resid_fn


def qcall(mask, bs, hd, q, k, v):
    L = H.layer; ks = H.step
    if H.phase == "denoise" and L % 2 == 1 and ks == 9:
        if P.edit is not None:
            xa = P.edit(L, P.X[L].float())
            if xa is not None:
                lay = H.ex.layers[L]; hs = lay.input_layernorm(xa.to(P.X[L].dtype)).to(lay.self_attn.q_proj.weight.dtype)
                q2 = apply_rope(lay.self_attn.q_proj(hs).view(bs, -1, q.shape[2], hd), POS.expand(bs, -1)).to(q.dtype)
                if P.idmode: P.qdiff.append(float((q2.float() - q.float()).abs().max()))
                q = q2
        if P.rec is not None and L in EL:
            with torch.no_grad():
                kr = k.repeat_interleave(q.shape[2] // k.shape[2], dim=2)                                  # (B,113,15,64)
                lg = torch.einsum("brhd,bthd->bhrt", q.float(), kr.float()) / 8.0
                mb = mask if mask.dtype == torch.bool else (mask > 0.5)
                if mb.dim() == 4: mb = mb[:, 0]
                if mb.shape[1] == 1: mb = mb.expand(bs, q.shape[1], mb.shape[-1])
                lg = lg.masked_fill(~mb[:, None], float("-inf")); pr = torch.softmax(lg, -1).mean(2)     # (B,15,113)
                P.rec[L] = torch.cat([pr[..., :64], pr[..., 64:].sum(-1, keepdim=True)], -1).half().cpu().numpy()   # (B,15,65)
    return H._eager(mask, bs, hd, q, k, v)
H.ve.eager_attention_forward = qcall


def flow(pm, kv, noise, edit=None, rec=None):
    x = noise.clone(); t = torch.tensor(1.0, device=dev); dt = torch.tensor(-1.0 / NS, device=dev)
    P.edit, P.rec = None, None
    try:
        with torch.no_grad():
            for k in range(NS):
                P.X = {}
                if k == 9: P.edit, P.rec = edit, rec
                v = Q.step(m, H, pm, kv, x, t, k); x = x + dt * v; t = t + dt
    finally: P.edit, P.rec = None, None
    return V.y_eval_np(x)


def block_token(phi):
    u, vv = S.w2uv(*S.block_xy(float(phi))); return int(min(7, max(0, vv // 64))) * 8 + int(min(7, max(0, u // 64)))
def tgt_of(scs): return np.array([s_["phi"] for s_ in scs])
def wrap(a): return (a + 180.0) % 360.0 - 180.0


# ---------------- fit panel: MU, plane, template, line, random-subspace template
NF = 6 if SMOKE else 96; fit = [S.make_scene(np.random.default_rng([61004, i]), "full") for i in range(NF)]
fphi = np.array([s_["phi"] for s_ in fit]); ftgt = tgt_of(fit); XF = {L: np.zeros((NF, 50, D), np.float32) for L in ODD}; t0 = time.time()
for j in range(0, NF, BS):
    sub = fit[j:j + BS]; B = len(sub); noise = R.noise_per_scene(list(range(11000 + j, 11000 + j + B))); E_ = V.Embs(H, pol, sub); kv = E_.kv()
    def cap(L, x, j=j, B=B):
        XF[L][j:j + B] = x.float().cpu().numpy(); return None
    flow(E_.pm, kv, noise, edit=cap); del kv
Q.log(f"fit panel recorded {NF} scenes, {time.time() - t0:.0f} s")
RNDn = {L: np.linalg.qr(np.random.default_rng([700, L]).standard_normal((D, 2)))[0].T for L in ODD}
MU = {}; BAS = {}; RND = {}; FITC = {}; FITR = {}; LIN = {}; fitinfo = {}
for L in ODD:
    X = XF[L]; mu = X.mean(0); d = (X - mu).reshape(-1, D).astype(np.float64); G = d.T @ d; w, U = np.linalg.eigh(G); U = U[:, ::-1]; w = w[::-1]
    Bl = np.ascontiguousarray(U[:, :2].T)
    c = (X - mu).mean(1) @ Bl.T; cr = (X - mu).mean(1) @ RNDn[L].T                                      # (NF,2) row-mean coordinates
    A_ = np.stack([np.ones(NF), fphi], 1); coef, res, _, _ = np.linalg.lstsq(A_, c, rcond=None)            # c ~ a + b phi
    pred = A_ @ coef; r2 = 1 - ((c - pred) ** 2).sum() / ((c - c.mean(0)) ** 2).sum()
    MU[L] = torch.tensor(mu, dtype=torch.float32, device=dev); BAS[L] = torch.tensor(Bl, dtype=torch.float32, device=dev); RND[L] = torch.tensor(RNDn[L], dtype=torch.float32, device=dev)
    FITC[L] = c; FITR[L] = cr; LIN[L] = coef
    fitinfo[f"L{L}"] = dict(line_r2=float(r2), line_a=coef[0].tolist(), line_b=coef[1].tolist(), ev_frac_top2=float(w[:2].sum() / w.sum()),
                            radius_med=float(np.median(np.hypot(c[:, 0], c[:, 1])))); Q.log(f"L{L}", fitinfo[f"L{L}"])
del XF


def template(c, tph):
    """mean fit-panel coordinate (c: NF x 2) at angles tph: scenes within 15 deg, else the nearest 3."""
    out = np.zeros((len(tph), 2))
    for i, t in enumerate(tph):
        dd = np.abs(wrap(fphi - t)); sel = dd <= 15
        if sel.sum() < 3: sel = np.argsort(dd)[:3]
        out[i] = c[sel].mean(0)
    return torch.tensor(out, dtype=torch.float32, device=dev)


def mk(kind, A, idmode=False):
    def ed(L, x):
        if L not in EL: return None
        P.idmode = idmode; d = x - MU[L]
        if kind == "rot":
            Bl = BAS[L]; cp = d @ Bl.T; return x + (cp - cp) @ Bl                                        # delta 0: identity path through the edit
        if kind == "tmpl":
            Bl = BAS[L]; cp = d @ Bl.T; ct = template(FITC[L], np.full(x.shape[0], float(A)))
        elif kind == "line":
            Bl = BAS[L]; cp = d @ Bl.T; ct = torch.tensor(LIN[L][0] + LIN[L][1] * float(A), dtype=torch.float32, device=dev)[None].expand(x.shape[0], -1)
        elif kind == "rand":
            Bl = RND[L]; cp = d @ Bl.T; ct = template(FITR[L], np.full(x.shape[0], float(A)))
        else: raise ValueError(kind)
        c2 = cp - cp.mean(1, keepdim=True) + ct[:, None]; return x + (c2 - cp) @ Bl
    return ed


# ---------------- eval panels
A = V.rand_scenes(61002, 1 if SMOKE else 48); A_noise = list(range(7000, 7000 + len(A)))
Bsc, B_noise, i = [], [], 0; NB = 1 if SMOKE else 128
while len(Bsc) < NB:
    s_ = S.make_scene(np.random.default_rng([61003, i]), "full")
    if abs(s_["phi"]) >= ENDS_ABS: Bsc.append(s_); B_noise.append(10000 + i)
    i += 1
scs = Bsc + A; nseeds = B_noise + A_noise; panel = ["ends"] * len(Bsc) + ["held"] * len(A); N = len(scs)
phi = np.array([s_["phi"] for s_ in scs]); tgt = tgt_of(scs); tok = np.array([block_token(p) for p in phi])
GRID = list(range(-165, 181, 15)); RG = [-150, -90, 0, 90, 150]
if SMOKE: GRID, RG = [-90, 90], [90]
CONDS = [("intact", 0), ("rot", 0)] + [("tmpl", a) for a in GRID] + [("line", a) for a in GRID] + [("rand", a) for a in RG]
ATT = np.zeros((len(CONDS), N, len(EL), 15, 65), np.float16); Y = np.zeros((len(CONDS), N), np.float32); COORD = np.zeros((N, 8, 2), np.float32); t0 = time.time()
Q.log(len(CONDS), "conditions x", N, "scenes")
for j in range(0, N, BS):
    sub = scs[j:j + BS]; B = len(sub); noise = R.noise_per_scene(nseeds[j:j + B]); E_ = V.Embs(H, pol, sub); kv = E_.kv()
    def cap(L, x, j=j, B=B):
        li = ODD.index(L); COORD[j:j + B, li] = ((x - MU[L]).mean(1) @ BAS[L].T).cpu().numpy(); return None
    for ci, (kind, a) in enumerate(CONDS):
        rec = {}
        ed = cap if kind == "intact" else mk(kind, a, idmode=(kind == "rot"))
        Y[ci, j:j + B] = flow(E_.pm, kv, noise, edit=ed, rec=rec)
        for li, L in enumerate(EL): ATT[ci, j:j + B, li] = rec[L]
    del kv
    e0 = np.abs(V.adiff(Y[0, :j + B], tgt[:j + B])); Q.log(f"{j + B}/{N}: intact median {np.median(e0):.3f}; id(rot0) max|dy| {np.abs(V.adiff(Y[1, :j + B], Y[0, :j + B])).max():.5f}; {time.time() - t0:.0f} s; {Q.peak()}")
    if (j // BS) % 4 == 3: np.save(f"{OD}/attention.npy", ATT); np.save(f"{OD}/answers.npy", Y)
np.save(f"{OD}/attention.npy", ATT); np.save(f"{OD}/answers.npy", Y)
idc = dict(max_abs_dy=float(np.abs(V.adiff(Y[1], Y[0])).max()), q_recompute_maxabs=float(max(P.qdiff)) if P.qdiff else None, att_maxabs=float(np.abs(ATT[1].astype(np.float32) - ATT[0].astype(np.float32)).max()))
meta = dict(odd=list(map(int, ODD)), edited_layers=list(map(int, EL)), N=N, panel=panel, phi=phi.tolist(), tgt=tgt.tolist(), tok=tok.tolist(),
            noise_seeds=list(map(int, nseeds)), conds=[[k, d] for k, d in CONDS], grid=GRID, fit=fitinfo, fit_N=NF, identity=idc,
            intact_med=float(np.median(np.abs(V.adiff(Y[0], tgt)))), ends_abs=ENDS_ABS, att_cols="64 image tokens + summed non-image mass")
json.dump(meta, open(f"{OD}/meta.json", "w"), indent=1)
Q.log("identity", idc, "intact med", meta["intact_med"]); Q.log(Q.peak()); Q.log("done")
