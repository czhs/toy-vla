"""The fold on the arm: Equation (fold) of the paper applied to the query weights of the arm policy at layers 5-15, with the removal and the write
scaled by lam. n is the row mean of the normalised stream entering q_proj (the query is n W_q); over the fit frames at the last flow step
n ~ nbar + c Bn, where c is the two-dimensional coordinate of the stream on its two leading principal directions. P = Bn^+ Bn is the projector on the
rows of Bn, g is fitted by least squares so that n g^T ~ 1 over action rows and flow steps, and c_A is the mean coordinate of fit frames whose reach pan
is within win degrees of the fixed angle A:
    M = I - lam P + lam g^T (nbar P + c_A Bn),     W_q' = W_q M^T   (torch's row convention: q = W n)
The reach pan of a frame is the expert's pan at the first row of its action chunk where the TCP goes below LOW_Z. Frames: the 192 fit episodes are
drawn from the dataset with fixed seeds; g uses one approach frame per episode, the plane and regression use the first three frames of each episode
that keep the table reach at row >= 3 and within 3 cm of the cube.

  python arm/fit_fold.py panels          select fit frames, cache their prefix K/V           $TOYVLA_DATA/arm_fold/{frames,kv}_{g,plane}.*   GPU
  python arm/fit_fold.py streams         record the row-mean streams at the last flow step   $TOYVLA_DATA/arm_fold/streams.npz               GPU
  python arm/fit_fold.py fit_g           g over action rows and flow steps                   $TOYVLA_DATA/arm_fold/g.npz                     GPU
  python arm/fit_fold.py export A LAM    write a folded checkpoint and its q_proj weights    $TOYVLA_DATA/arm_fold/fold_A{A}_lam{LAM}/       CPU
The paper's edits: A = +60 and A = -60 at LAM = 1.25; LAM 0.75 and 1.5 are the other candidates on the development layouts."""
import json, shutil, sys, time
import numpy as np, torch
import paths

STAGE = sys.argv[1]
OUT = paths.FOLD; OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261101; NEV = 192; NFIT0 = 128; NX = 192; BS = 16; NS = 10; DM = 720; NL = 16; ODD = list(range(1, 16, 2)); EL = [5, 7, 9, 11, 13, 15]; WIN = 10.0
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

# ------------------------------------------------------------------------------------------------ export: numpy and safetensors only
if STAGE == "export":
    from safetensors.torch import load_file, save_file
    A, lam = float(sys.argv[2]), float(sys.argv[3]); name = f"fold_A{A:+g}_lam{lam:g}"
    ST = np.load(OUT / "streams.npz"); GF = np.load(OUT / "g.npz"); PAN = ST["pan"]
    def fold_M(L):
        Xc = ST[f"xm_L{L}"].astype(np.float64); Xc = Xc - Xc.mean(0); n = ST[f"nm_L{L}"].astype(np.float64); Nc = n - n.mean(0)
        w, U = np.linalg.eigh(Xc.T @ Xc); w, U = w[::-1], U[:, ::-1]; B = np.ascontiguousarray(U[:, :2].T)
        C = Xc @ B.T; Bn = np.linalg.solve(C.T @ C, C.T @ Nc); nbar = n.mean(0)
        dd = np.abs(PAN - A); sel = dd <= WIN
        if sel.sum() < 3: sel = np.argsort(dd)[:3]
        cA = C[sel].mean(0); g = GF[f"g_L{L}"].astype(np.float64); Pm = np.linalg.pinv(Bn) @ Bn
        log(f"L{L}: top-2 variance share {w[:2].sum() / w.sum():.3f}, n on c r2 {1 - ((Nc - C @ Bn) ** 2).sum() / (Nc ** 2).sum():.3f}, template frames {int(sel.sum())}")
        return np.eye(DM) - lam * Pm + lam * np.outer(g, nbar @ Pm + cA @ Bn)
    src = paths.POLICY; sd0 = load_file(str(src / "model.safetensors")); pre = "model.vlm_with_expert.lm_expert.layers."; sd = dict(sd0); changed = {}
    for L in EL:
        kq = f"{pre}{L}.self_attn.q_proj.weight"; M = torch.tensor(fold_M(L), dtype=torch.float64)
        sd[kq] = (sd0[kq].double() @ M.T).to(sd0[kq].dtype).contiguous(); changed[kq] = sd[kq]
    dst = OUT / name; dst.mkdir(parents=True, exist_ok=True)
    for fp in src.iterdir():
        if fp.name not in ("model.safetensors", "delta.safetensors"): shutil.copy2(fp, dst / fp.name)
    save_file(sd, str(dst / "model.safetensors"), metadata={"format": "pt"}); save_file(changed, str(OUT / f"{name}_q_proj.safetensors"), metadata={"format": "pt"})
    log(f"wrote {dst}: {len(changed)} q_proj tensors differ from the policy"); sys.exit(0)

# ------------------------------------------------------------------------------------------------ everything else runs the policy
import policy_tools as X
from policy_tools import apply_rope
Mdl = X.Model(ckpt=str(paths.POLICY))

# ------------------------------------------------------------------------------------------------ panels
if STAGE == "panels":
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root = paths.DATASET
    ds = LeRobotDataset("local/so101_redcube_angled_shadow", root=str(root), delta_timestamps={"action": [i / 10 for i in range(50)]})
    side = [json.loads(l) for l in open(root / "meta/so101_episodes.jsonl")]
    ep = np.array(ds.hf_dataset["episode_index"]); starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]]); lens = np.diff(np.r_[starts, len(ep)])
    rng = np.random.default_rng(SEED); eps_all = rng.choice(len(starts), size=3 * NEV + NFIT0, replace=False)     # episodes held out of the fit
    sc = X.Scene(X.spec_of("plain")); sc.env.reset(seed=0)
    def reach_row(acts):
        tcp = sc.tcp_of(acts); i = np.nonzero(tcp[:, 2] < X.LOW_Z)[0]
        return (int(i[0]), tcp[int(i[0]), :2]) if len(i) else (-1, tcp[np.argmin(tcp[:, 2]), :2])
    def pull(e, i):
        it = ds[int(i)]; s = side[int(ep[starts[e]])]; assert s["episode_index"] == int(ep[starts[e]])
        return ((it["observation.images.overhead"].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8), it["observation.state"].numpy(),
                it["action"].numpy(), it["action_is_pad"].numpy(), np.array(s["cube_xy"]))
    def approach(eps_, n, r):
        rows = []
        for e in eps_:
            i = starts[e] + r.integers(3); im, st, gt, pd, cx = pull(e, i); jr, rxy = reach_row(gt[~pd])
            if jr < 3 or np.linalg.norm(rxy - cx) > 0.03 or pd[:10].any(): continue
            rows.append((e, i, im, st, gt, pd, cx, jr))
            if len(rows) == n: break
        assert len(rows) == n, (len(rows), n)
        return rows
    def first_frames(e):
        """frames among the first three of episode e whose chunk reaches the table at row >= 3, within 3 cm of the cube."""
        rows = []; opened = False
        for i in range(starts[e], starts[e] + min(3, lens[e])):
            im, st, gt, pd, cx = pull(e, i); grip = float(st[5])
            if grip > 85: opened = True
            if opened and grip < 25: break
            if pd[:10].any(): break
            tcp = sc.tcp_of(gt[~pd]); j = np.nonzero(tcp[:, 2] < X.LOW_Z)[0]
            if not len(j): continue
            jr = int(j[0])
            if np.linalg.norm(tcp[jr, :2] - cx) > 0.03 or jr < 3: continue
            rows.append((e, i, im, st, gt, pd, cx, jr))
        return rows
    def save(name, rows):
        N = len(rows); idx = np.array([r[1] for r in rows]); imgs = np.stack([r[2] for r in rows]); states = np.stack([r[3] for r in rows]); GT = np.stack([r[4] for r in rows])
        cube = np.stack([r[6] for r in rows]); JR = np.array([r[7] for r in rows])
        KV = np.lib.format.open_memmap(OUT / f"kv_{name}.npy", mode="w+", dtype=np.float16, shape=(NL, 2, N, 78, 5, 64))
        for j in range(0, N, BS):
            kv = Mdl.prefix(imgs[j:j + BS], states[j:j + BS]); assert kv[0][0].shape[1] == 78
            for l in range(NL): KV[l, 0, j:j + BS] = kv[l][0].float().cpu().numpy().astype(np.float16); KV[l, 1, j:j + BS] = kv[l][1].float().cpu().numpy().astype(np.float16)
            del kv
        KV.flush(); del KV
        np.savez(OUT / f"frames_{name}.npz", idx=idx, episode=np.array([r[0] for r in rows]), cube_xy=cube, bearing=np.degrees(np.arctan2(cube[:, 1], cube[:, 0])), reach_row=JR, pan_target_reach=GT[np.arange(N), JR, 0])
        log(name, N, "frames; reach pan range", GT[np.arange(N), JR, 0].min().round(1), GT[np.arange(N), JR, 0].max().round(1))
    rest = np.setdiff1d(np.arange(len(starts)), eps_all); rng2 = np.random.default_rng(SEED + 7); rest = rng2.permutation(rest)
    fit = approach(rest[: len(rest) // 2], NX, rng2); save("g", fit)
    plane = []
    for e in np.unique([r[0] for r in fit]): plane += first_frames(int(e))
    save("plane", plane); sc.close(); log("stage panels done"); sys.exit(0)

# ------------------------------------------------------------------------------------------------ expert in fp32 (the prefix cache stays as computed under bf16)
TV = X.tvals(Mdl); m, vwe = Mdl.m, Mdl.vwe; ex = vwe.lm_expert; dev = Mdl.dev
for mod in (ex, m.action_in_proj, m.action_time_mlp_in, m.action_time_mlp_out, m.action_out_proj): mod.float()
for L in ODD: assert ex.layers[L].self_attn.q_proj.bias is None

def fwd(kv, x_t, t, tap=None):
    h = X.suffix_emb(Mdl, x_t, t)
    B, Ls = h.shape[:2]; Pp = kv[0][0].shape[1]
    pos_a = (Pp + torch.arange(Ls, device=dev))[None].expand(B, Ls); pos_e = torch.arange(Ls, device=dev)[None].expand(B, Ls)
    causal = torch.tril(torch.ones(Ls, Ls, dtype=torch.bool, device=dev))
    m_self = torch.cat([torch.ones(B, Ls, Pp, dtype=torch.bool, device=dev), causal[None].expand(B, Ls, Ls)], 2)
    m_cross = torch.ones(B, Ls, Pp, dtype=torch.bool, device=dev)
    for l, el in enumerate(ex.layers):
        K, V = kv[l]; shp = (B, Ls, -1, 64)
        if l % 2 == 0:
            x = el.input_layernorm(h)
            q = apply_rope(el.self_attn.q_proj(x).view(shp), pos_a); k = apply_rope(el.self_attn.k_proj(x).view(shp), pos_a)
            v = el.self_attn.v_proj(x).view(shp)
            att = vwe.eager_attention_forward(m_self, B, 64, q, torch.cat([K, k], 1), torch.cat([V, v], 1))
        else:
            if tap is not None: tap(l, h, el.input_layernorm(h))
            q = apply_rope(el.self_attn.q_proj(el.input_layernorm(h)).view(shp), pos_e)
            Kp = el.self_attn.k_proj(K.reshape(B, Pp, -1)).view(B, Pp, -1, 64); Vp = el.self_attn.v_proj(V.reshape(B, Pp, -1)).view(B, Pp, -1, 64)
            att = vwe.eager_attention_forward(m_cross, B, 64, q, Kp, Vp)
        h = h + el.self_attn.o_proj(att.float())
        h = h + el.mlp(el.post_attention_layernorm(h))
    return m.action_out_proj(ex.norm(h))

CUR = dict(k=0)
def tvec(k, B): return torch.full((B,), float(TV[k]), device=dev)
@torch.no_grad()
def flow(kv, noise, tap=None):
    x = noise.clone()
    for k in range(NS):
        CUR["k"] = k; v = fwd(kv, x, tvec(k, len(x)), tap=tap); x = x - 0.1 * v
    return x
def kv_to_dev(KV, sl): return [(torch.from_numpy(np.ascontiguousarray(KV[l, 0, sl])).to(dev).float(), torch.from_numpy(np.ascontiguousarray(KV[l, 1, sl])).to(dev).float()) for l in range(NL)]
def load_set(name): return np.load(OUT / f"frames_{name}.npz"), np.load(OUT / f"kv_{name}.npy", mmap_mode="r")
def noise_for(idx): return Mdl.noise([7100000 + int(i) for i in idx])

# ------------------------------------------------------------------------------------------------ streams: row means at the last flow step
if STAGE == "streams":
    z, KV = load_set("plane"); N = len(z["idx"]); XM = {L: np.zeros((N, DM), np.float16) for L in EL}; NM = {L: np.zeros((N, DM), np.float16) for L in EL}
    for j in range(0, N, BS):
        sl = slice(j, j + BS); kv = kv_to_dev(KV, sl)
        def tap(l, h, n, j=j):
            if l in EL and CUR["k"] == NS - 1: XM[l][j:j + len(h)] = h.mean(1).cpu().numpy().astype(np.float16); NM[l][j:j + len(h)] = n.mean(1).cpu().numpy().astype(np.float16)
        flow(kv, noise_for(z["idx"][sl]), tap=tap); del kv
    np.savez(OUT / "streams.npz", pan=z["pan_target_reach"], bearing=z["bearing"], **{f"xm_L{L}": XM[L] for L in EL}, **{f"nm_L{L}": NM[L] for L in EL})
    log("stage streams done", N, "frames"); sys.exit(0)

# ------------------------------------------------------------------------------------------------ g: least squares n g^T ~ 1 over every 5th action row of all flow steps
if STAGE == "fit_g":
    z, KV = load_set("g"); N = len(z["idx"]); NN = {L: [] for L in EL}
    for j in range(0, N, BS):
        sl = slice(j, j + BS); kv = kv_to_dev(KV, sl)
        def tap(l, h, n):
            if l in EL: NN[l].append(n[:, ::5].reshape(-1, DM).cpu().numpy())
        flow(kv, noise_for(z["idx"][sl]), tap=tap); del kv
    G = {}
    for L in EL:
        Nm = np.concatenate(NN[L]).astype(np.float64); G[f"g_L{L}"] = np.linalg.lstsq(Nm, np.ones(len(Nm)), rcond=None)[0]; r = Nm @ G[f"g_L{L}"]
        log(f"g fitted L{L}: g.n p1/p50/p99 {np.percentile(r, [1, 50, 99]).round(3).tolist()}")
    np.savez(OUT / "g.npz", **G); log("stage fit_g done"); sys.exit(0)
raise SystemExit(f"unknown stage {STAGE}")
