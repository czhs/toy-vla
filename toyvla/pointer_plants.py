"""Scenes, conditions and the runner of the pointer test. A condition edits the residual stream of the marked digit's token at chosen layers:
  plan = list of (edit layer Le, content layer Lc): the stream entering Le is edited with layer Lc's read directions S_Lc (rows: all / V only / K only)
  and target m * T_Lc; for a plant with nothing edited upstream the added vector is m * r * M_Lc T_Lc, r = rms of the token at Le.
The stream entering every odd layer is recorded after the edit, so the achieved change in units of T_L is logged with the answers."""
import json, time, numpy as np, torch
from toyvla import vlm_pointer as C
from toyvla.paths import DEVICE, peak_gpu_bytes, reset_peak_gpu
SIZE = 52; ODD = C.ODD

# ------------------------------------------------------------ scenes
def ring8_draw(i, seed=20261200):
    rng = np.random.default_rng([seed, i]); th0 = float(rng.uniform(0, 45)); perm = rng.permutation(8); tgt = int(rng.integers(8)); return th0, perm, tgt
def scene_from_phis(phis, digits, tgt, rad=C.R0, c=C.PIV, size=SIZE):
    objs = [(d, *map(float, C.uv(p, rad, c)), float(C.wrap(p))) for d, p in zip(digits, phis)]; return dict(objs=objs, tgt=int(tgt), size=size)
def ring8_scenes(N, seed=20261200, off=0.0):
    """perm, tgt from the draw; the ring offset is off (off=None keeps the drawn th0)."""
    scs = []
    for i in range(N):
        th0, perm, tgt = ring8_draw(i, seed); o = th0 if off is None else off
        s = scene_from_phis(C.wrap(o + 45.0 * np.arange(8)), [C.DIG[p] for p in perm], tgt); d, u, v, _ = s["objs"][tgt]
        s["tok"] = C.tok_of(u, v); s["digit"] = d; scs.append(s)
    return scs
def images(scs):
    return dict(plain=[C.render([(d, u, v) for d, u, v, _ in s["objs"]], s["size"]) for s in scs],
                box=[C.render([(d, u, v) for d, u, v, _ in s["objs"]], s["size"], box=s["objs"][s["tgt"]][1:3]) for s in scs])

# ------------------------------------------------------------ model + expert
class Env:
    def __init__(self, bs=40):
        self.V = C.VLM(); self.P = C.prompts(self.V); self.EX = C.Expert(); self.bs = bs
        D = C.directions(); self.kV = {L: D[L]["kV"] for L in ODD}; self.kK = {L: D[L]["kK"] for L in ODD}
        for L in ODD: assert self.EX.Sd[L][0].shape[1] == self.kV[L] + self.kK[L] == len(self.EX.T[L])
        self.UA = {L: C.unit_rand(1000 + L, 4096) for L in range(0, 32)}
        self._rows = {}
    def rows(self, L, which):
        k = self.kV[L] + self.kK[L]
        return {"all": np.arange(k), "V": np.arange(self.kV[L]), "K": np.arange(self.kV[L], k)}[which]
    def SM(self, L, which):
        """(S, M) of content layer L restricted to rows; 'all' returns the Expert's own tensors."""
        if which == "all": return self.EX.Sd[L]
        if (L, which) not in self._rows:
            S = self.EX.Sd[L][0][:, torch.tensor(self.rows(L, which), device=DEVICE)].double()
            M = S.transpose(1, 2) @ torch.linalg.inv(S @ S.transpose(1, 2)); self._rows[(L, which)] = (S.float().contiguous(), M.float())
        return self._rows[(L, which)]
    def Trow(self, L, which): return self.EX.T[L][torch.tensor(self.rows(L, which), device=DEVICE)]

    def capture(self, imgs, layers):
        V = self.V; B, kw = V._kw(imgs); ii = V.PRE[None].expand(B, -1).contiguous(); V.full = V.NPRE; V.cap = set(layers); V.caps = {}
        with torch.no_grad(): V.m(input_ids=ii, use_cache=False, **kw)
        V.cap = None; V.full = None; out = {L: V.caps[L] for L in layers}; V.caps = {}; return out
    def prefix_meas(self, imgs, ed, tk, meas):
        V = self.V; B, kw = V._kw(imgs); ii = V.PRE[None].expand(B, -1).contiguous(); post = {}
        gi = tk.clamp(min=0)[..., None].expand(-1, -1, 960)
        def f(L, X):
            new = ed.apply(L, X) if ed is not None else None
            if L in meas: post[L] = (new if new is not None else X.float()).gather(1, gi).clone()
            return new
        V.edit = f; V.full = V.NPRE
        with torch.no_grad(): o = V.m(input_ids=ii, use_cache=True, **kw)
        V.edit = None; V.full = None; return o.past_key_values.to_legacy_cache(), post

def ptok(S, x, tk, eps):
    """c (B, n, k) = S_j x_j / rms(x_j) for tokens j = tk (x already gathered: (B, n, 960))."""
    Sj = S[tk.clamp(min=0)]; r = x.pow(2).mean(-1, keepdim=True).add(eps).sqrt(); return torch.einsum("bnkd,bnd->bnk", Sj, x / r)

def run(E, imgsets, conds, scs, out_path=None, meta=None, log=print):
    """conds: list of (key, spec). spec: dict(img=...) [no edit]; dict(kind='calib', img, toks, m, plan=[(Le, Lc)], rows='all'|'V'|'K');
    dict(kind='rand', img, toks, src=key of an earlier calib cond) = random unit direction per scene and layer (seed 1000+Le, scene index),
    per-token norm = that of the source cond's edit, at the source's edit layers. Returns res[key] = dict(prompt -> text/got/lp, meas)."""
    V = E.V; N = len(scs); eps = V.eps; tg = [s["digit"] for s in scs]
    res = {k: dict(**{p: dict(text=[], got=[], lp=[]) for p in E.P}, meas={}) for k, _ in conds}
    t0 = time.time(); reset_peak_gpu()
    for j in range(0, N, E.bs):
        ii = list(range(j, min(N, j + E.bs))); X = {}; recs = {}; plans = {}
        need = {}
        for key, c in conds:
            if c.get("kind") in ("calib",): need.setdefault(c.get("img", "plain"), set()).update(ODD + [Le for Le, _ in c["plan"]])
            if c.get("kind") == "rand": need.setdefault(c.get("img", "plain"), set()).update(ODD + [Le for Le, _ in plans_of(conds, c["src"])])
        for nm, Ls in need.items(): X[nm] = E.capture([imgsets[nm][i] for i in ii], sorted(Ls))
        for key, c in conds:
            nm = c.get("img", "plain"); im = [imgsets[nm][i] for i in ii]; ed = None; kind = c.get("kind", "none")
            if kind == "none":
                cache = V.prefix(im, None); post = None
            else:
                tk = C.pad_tokens([c["toks"][i] for i in ii])
                if kind == "calib":
                    Sd, T, c0 = {}, {}, {}
                    for Le, Lc in c["plan"]:
                        S, M = E.SM(Lc, c.get("rows", "all")); Sd[Le] = (S, M); T[Le] = E.EX.T[Lc] if c.get("rows", "all") == "all" else E.Trow(Lc, c["rows"])
                        c0[Le] = torch.gather(C.proj(Sd, Le, X[nm][Le], eps), 1, tk.clamp(min=0)[..., None].expand(-1, -1, S.shape[1]))
                    ed = C.Editor([Le for Le, _ in c["plan"]], tk, eps, mode="calib", Sd=Sd, T=T, m=c["m"], c0=c0)
                    elayers = [Le for Le, _ in c["plan"]]
                else:
                    src = recs[c["src"]]; elayers = sorted(src)
                    ed = C.Editor(elayers, tk, eps, mode="rand", norms={L: src[L].norm(dim=-1) for L in elayers}, U={L: E.UA[L][torch.tensor(ii, device=DEVICE)] for L in elayers})
                meas = set(ODD) | set(elayers)
                cache, post = E.prefix_meas(im, ed, tk, meas); recs[key] = {L: v.clone() for L, v in ed.rec.items()}
                # ---- measurements (per scene, mean over the valid planted tokens)
                vm = (tk >= 0).float(); nv = vm.sum(1)
                def tm(a): return ((a * vm).sum(1) / nv).cpu().numpy().tolist()
                gi = tk.clamp(min=0)[..., None].expand(-1, -1, 960); M_ = res[key]["meas"]
                for L in ODD:
                    S = E.EX.Sd[L][0]; Tl = E.EX.T[L]; x0 = X[nm][L].gather(1, gi); dc = ptok(S, post[L], tk, eps) - ptok(S, x0, tk, eps)
                    a = (dc * Tl).sum(-1) / Tl.pow(2).sum(); kv = E.kV[L]
                    aV = (dc[..., :kv] * Tl[:kv]).sum(-1) / Tl[:kv].pow(2).sum(); aK = (dc[..., kv:] * Tl[kv:]).sum(-1) / Tl[kv:].pow(2).sum()
                    perp = (dc - a[..., None] * Tl).norm(dim=-1) / Tl.norm()
                    d = M_.setdefault(f"L{L}", dict(ach=[], achV=[], achK=[], perp=[]))
                    d["ach"] += tm(a); d["achV"] += tm(aV); d["achK"] += tm(aK); d["perp"] += tm(perp)
                for Le in elayers:
                    x0 = X[nm][Le].gather(1, gi); r0 = x0.pow(2).mean(-1).sqrt(); r1 = post[Le].pow(2).mean(-1).sqrt(); dr = recs[key][Le].pow(2).mean(-1).sqrt()
                    d = M_.setdefault(f"edit{Le}", dict(drms=[], rms_ratio=[], rms0=[]))
                    d["drms"] += tm(dr / r0); d["rms_ratio"] += tm(r1 / r0); d["rms0"] += tm(r0)
            for p, suf in E.P.items():
                txt, lp = V.answer(cache, suf); r = res[key][p]; r["text"] += txt; r["got"] += [C.parse(t) for t in txt]; r["lp"] += np.round(lp, 3).tolist()
            del cache, post
        n = len(res[conds[0][0]][next(iter(E.P))]["got"])
        log(f"batch {j} done ({n} scenes, {len(conds)} conds) {time.time()-t0:.0f}s peak {peak_gpu_bytes()/2**30:.2f} GiB")
        if out_path: save(out_path, E, res, conds, scs, meta)
    return res

def plans_of(conds, key):
    return dict(conds)[key]["plan"]

def acc(r, scs, p): return float(np.mean([a == s["digit"] for a, s in zip(r[p]["got"], scs)]))
def tlp(r, scs, p): return float(np.mean([lp[C.DIG.index(s["digit"])] for lp, s in zip(r[p]["lp"], scs)]))
def summary(res, scs, P):
    out = {}
    for k, r in res.items():
        n = len(r[next(iter(P))]["got"]); sc = scs[:n]
        o = {p: dict(acc=acc(r, sc, p), tlp=tlp(r, sc, p), n=n) for p in P}
        if r["meas"]: o["meas"] = {mk: {q: float(np.mean(v)) for q, v in mv.items()} for mk, mv in r["meas"].items()}
        out[k] = o
    return out
def spec_json(c):
    return {k: v for k, v in c.items() if k not in ("toks",)} | ({"ntok": [len(t) for t in c["toks"][:3]]} if "toks" in c else {})
def save(path, E, res, conds, scs, meta):
    json.dump(dict(meta=dict(prompts={k: E.V.tok.decode(v) for k, v in E.P.items()}, **(meta or {})),
                   conds={k: spec_json(c) for k, c in conds}, scenes=[dict(digit=s["digit"], tok=s["tok"], tgt=s["tgt"]) for s in scs],
                   summary=summary(res, scs, E.P), res=res), open(path, "w"))

# ------------------------------------------------------------ condition builders
def calib(toks, plan, m, rows="all", img="plain"): return dict(kind="calib", img=img, toks=toks, plan=[(int(a), int(b)) for a, b in plan], m=m, rows=rows)
def rand(toks, src, img="plain"): return dict(kind="rand", img=img, toks=toks, src=src)
def with_rand(key, spec): return [(key, spec), ("rand_" + key, rand(spec["toks"], key, spec.get("img", "plain")))]
def ref_conds(scs):
    t1 = [[s["tok"]] for s in scs]; own = [(L, L) for L in ODD]
    return [("none", dict(img="plain")), ("box", dict(img="box"))] + with_rand("all_odd_m2", calib(t1, own, 2))
