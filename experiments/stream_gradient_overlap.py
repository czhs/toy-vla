"""How much of the stream gradient lies in the pulled-back read directions, CPU only. PB_L[j]: the pulled-back directions of layer L at image token j, orthonormalised; union: the concatenation over odd layers
L' >= L. Energy share inside PB: sum over rows (scene, token j) of |Q_j g|^2 / sum |g|^2; random control: 4 random orthonormal subspaces of the same dimension, mean. ov_local_noJ: overlap of the local term's
90% directions with and without the normalisation Jacobian. Output: results/stream_gradient_overlap.json. Runtime about 1 min on 4 CPU threads; needs data/stream_gradients/."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import json, numpy as np, torch
from toyvla import stream_tools as V
from toyvla.paths import DATA, RESULTS, GRADIENT_DIRECTIONS
OD = DATA / "stream_gradients"
torch.set_num_threads(4)
NI, D = 64, 960; LS = list(range(16)); ODD = V.ODD; RNG = np.random.default_rng(5)
Z = {nm: np.load(f"{OD}/eig_{nm}.npz") for nm in ("grid", "rand")}
MG = json.load(open(f"{OD}/meta_grid.json")); MR = json.load(open(f"{OD}/meta_rand.json"))
scg = np.load(f"{OD}/scales_grid.npz"); scr = np.load(f"{OD}/scales_rand.npz")
def stack(name, var, sc):
    a = np.load(f"{OD}/{name}_{var}.npy", mmap_mode="r"); return lambda L: a[L].astype(np.float64) * sc[name][L][:, None, None]
AF = stack("A_full", "grid", scg); AL = stack("A_loc", "grid", scg); AN = stack("A_noJ", "grid", scg); AFr = stack("A_full", "rand", scr)
k90 = lambda w: V.kfrac(w, 0.90); k99 = lambda w: V.kfrac(w, 0.99)
def e_basis(Zg, key, L, frac=0.90):
    w = Zg[f"ev{key}"][L]; k = V.kfrac(w, frac); return Zg[f"vec{key}"][L][:k]
PBd = V.pulled_back(LS)
def orth_tok(M):  # (64, k, 960) -> list of orthonormal (k', 960) per token
    return [V.orth_rows(M[j]) for j in range(NI)]
PB = {L: orth_tok(PBd[L]["S"]) for L in LS}
UN = {}
for L in LS:
    for nm, src in (("odd", [q for q in ODD if q >= L]), ("all", [q for q in LS if q >= L])):
        UN[(nm, L)] = [V.orth_rows(np.concatenate([PBd[q]["S"][j] for q in src], 0)) for j in range(NI)]
def ov_tok(A, Qs): return float(np.mean([V.mean_cos2(A, Q) for Q in Qs]))
def share_tok(G, Qs):
    """G (n, 64, 960) gradient rows; Qs list of 64 orthonormal bases -> energy share inside."""
    num = sum(float(((G[:, j] @ Qs[j].T) ** 2).sum()) for j in range(NI)); return num / float((G ** 2).sum())
def share(G, Q): return float(((G.reshape(-1, D) @ Q.T) ** 2).sum() / (G ** 2).sum())
def rand_orth(k): return V.orth_rows(RNG.standard_normal((k, D)))
out = dict(n_grid=MG["n"], n_rand=MR["n"], layers={})
for L in LS:
    G = AF(L); Gl = AL(L); Gn = AN(L); eA = e_basis(Z["grid"], "A", L); kS = len(eA)
    r = dict(k90=kS, k99=k99(Z["grid"]["evA"][L]), kPB=int(PBd[L]["kV"] + PBd[L]["kK"]), kV=PBd[L]["kV"], kK=PBd[L]["kK"])
    r["ov_PB"] = ov_tok(eA, PB[L]); r["ov_union_odd"] = ov_tok(eA, UN[("odd", L)]); r["ov_union_all"] = ov_tok(eA, UN[("all", L)])
    r["k_union_odd"] = int(np.median([len(q) for q in UN[("odd", L)]])); r["k_union_all"] = int(np.median([len(q) for q in UN[("all", L)]]))
    rs = [rand_orth(kS) for _ in range(4)]
    r["ov_PB_rand"] = float(np.mean([ov_tok(q, PB[L][::8]) for q in rs])); r["ov_union_odd_rand"] = float(np.mean([ov_tok(q, UN[("odd", L)][::8]) for q in rs]))
    r["share_PB"] = share_tok(G, PB[L]); r["share_union_odd"] = share_tok(G, UN[("odd", L)]); r["share_union_all"] = share_tok(G, UN[("all", L)])
    r["share_PB_rand"] = float(np.mean([share(G, rand_orth(r["kPB"])) for _ in range(4)]))
    r["share_union_odd_rand"] = float(np.mean([share(G, rand_orth(r["k_union_odd"])) for _ in range(4)])) if r["k_union_odd"] > 0 else 0.0
    r["share_k90_self"] = share(G, eA)
    # local term
    eF, eLc, eR = float((G ** 2).sum()), float((Gl ** 2).sum()), float(((G - Gl) ** 2).sum())
    r["local_energy_frac"] = eLc / eF; r["rest_energy_frac"] = eR / eF; r["cos_local_full"] = float((G * Gl).sum() / np.sqrt(eF * eLc))
    r["cross_term_frac"] = float(2 * (Gl * (G - Gl)).sum() / eF)
    rowc = (G * Gl).sum(-1) / np.maximum(np.linalg.norm(G, axis=-1) * np.linalg.norm(Gl, axis=-1), 1e-30); r["cos_local_full_row_median"] = float(np.median(rowc))
    eLb = e_basis(Z["grid"], "L", L); eNb = e_basis(Z["grid"], "N", L); r["k90_local"] = len(eLb); r["k90_noJ"] = len(eNb)
    r["ov_local_PB"] = ov_tok(eLb, PB[L]); r["ov_noJ_PB"] = ov_tok(eNb, PB[L]); r["ov_local_noJ"] = V.mean_cos2(eLb, eNb)
    r["share_local_in_PB"] = share_tok(Gl, PB[L]); r["share_noJ_in_PB"] = share_tok(Gn, PB[L])
    r["share_local_in_streamK90"] = share(Gl, eA); r["share_full_in_localK90"] = share(G, eLb)
    r["ov_k90_local_vs_full"] = V.mean_cos2(eA, eLb)
    # cache-space check: fp32 cache gradients (this run) vs gradient_directions.json (bf16-prefix cache gradients) 90% directions
    Gj = json.load(open(GRADIENT_DIRECTIONS)) if L == 0 else Gj
    for kind, f in (("V", "gV"), ("K", "gK")):
        g = np.load(f"{OD}/{f}_grid.npy", mmap_mode="r")[L].reshape(-1, 320).astype(np.float64)
        w, Q = np.linalg.eigh(g.T @ g); w, Q = w[::-1], Q[:, ::-1].T; kk = V.kfrac(w, 0.90)
        DJ = np.array(Gj["D" + kind][str(L)][:V.kept(Gj, L, kind)])
        r[f"cache_{kind}_k90_fp32"] = kk; r[f"cache_{kind}_k90_json"] = len(DJ); r[f"cache_{kind}_ov_fp32_json"] = V.mean_cos2(Q[:kk], DJ)
        r[f"cache_{kind}_energy_fp32"] = float(w.sum())
    # second scene set and pooled / per-step relations
    Gr = AFr(L); r["k90_rand"] = k90(Z["rand"]["evA"][L]); r["k99_rand"] = k99(Z["rand"]["evA"][L])
    r["share_rand_in_gridK90"] = share(Gr, eA); r["share_rand_in_gridK99"] = share(Gr, e_basis(Z["grid"], "A", L, 0.99))
    r["ov_grid_rand_k90"] = V.mean_cos2(eA, e_basis(Z["rand"], "A", L))
    r["ov_a_pool_k90"] = V.mean_cos2(eA, e_basis(Z["grid"], "P", L))
    r["ov_a_step9_k90"] = V.mean_cos2(eA, Z["grid"]["vecB"][L, 9][:k90(Z["grid"]["evB"][L, 9])])
    r["ov_a_step0_k90"] = V.mean_cos2(eA, Z["grid"]["vecB"][L, 0][:k90(Z["grid"]["evB"][L, 0])])
    # sample size: 90% / 99% counts of random subsets of the grid scenes (3 draws per size)
    nG = G.shape[0]; ss = {}
    for nn in sorted({n_ for n_ in (12, 24, 36, 48, 60) if n_ < nG} | {nG}):
        ks = []
        for d in range(3 if nn < nG else 1):
            idx = RNG.choice(nG, nn, replace=False) if nn < nG else np.arange(nG); g2 = G[idx].reshape(-1, D)
            w = np.linalg.eigvalsh(g2.T @ g2)[::-1]; ks.append((k90(w), k99(w)))
        ss[nn] = np.mean(ks, 0).tolist()
    r["k_vs_n"] = ss
    out["layers"][L] = r
    print(f"L{L:2d} k90 {kS:3d} kPB {r['kPB']:2d} | ov PB {r['ov_PB']:.3f} (rand {r['ov_PB_rand']:.3f}) union_odd {r['ov_union_odd']:.3f} (k {r['k_union_odd']}, rand {r['ov_union_odd_rand']:.3f}) | "
          f"share PB {r['share_PB']:.3f} (rand {r['share_PB_rand']:.3f}) union_odd {r['share_union_odd']:.3f} all {r['share_union_all']:.3f} | local frac {r['local_energy_frac']:.3f} "
          f"cos {r['cos_local_full']:.3f} | ov local-PB {r['ov_local_PB']:.3f} noJ-PB {r['ov_noJ_PB']:.3f} | cacheV ov {r['cache_V_ov_fp32_json']:.3f} K {r['cache_K_ov_fp32_json']:.3f} | "
          f"rand in grid 90% {r['share_rand_in_gridK90']:.3f} | k_vs_n {ss}", flush=True)
json.dump(out, open(RESULTS / "stream_gradient_overlap.json", "w"), indent=1); print("done")
