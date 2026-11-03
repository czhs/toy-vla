"""The pointer test on the frozen vision-language model: digit scenes, the pull-back of the expert's read directions into the residual stream,
the calibrated editor and the prefix-cache answer of the full SmolVLM2 with its language head."""
import json, re, numpy as np, torch
from huggingface_hub import snapshot_download
from toyvla import policy_tools
from toyvla.paths import DEVICE, GRADIENT_DIRECTIONS, VLM_WEIGHTS, BLOCK_CALIBRATION
from toyvla.digit_scenes import *
from toyvla.digit_scenes import SS, IMG, PIV, R0, DIG, BG, COL, FONT
ODD = list(range(1, 16, 2))

# ---------------------------------------------------------------- read directions + calibrated editor
from lerobot.policies.smolvla.smolvlm_with_expert import apply_rope
_Z = None
def Z():
    global _Z
    if _Z is None: _Z = np.load(VLM_WEIGHTS)
    return _Z
def rope_back(D):
    d = torch.tensor(np.asarray(D), dtype=torch.float32).reshape(len(D), 1, 5, 64).expand(len(D), 64, 5, 64); pos = torch.arange(64)[None].float()
    return apply_rope(d, -pos.expand(len(D), 64)).reshape(len(D), 64, 320).permute(1, 0, 2).numpy().astype(np.float64)
def kept():
    G = json.load(open(GRADIENT_DIRECTIONS)); out = {}
    for L in ODD:
        for kind in "VK":
            s = np.array(G["s" + kind][str(L)]); tot = G["grad_share"][kind][L] * G["grad_energy_total"]
            out[(L, kind)] = int(np.searchsorted(np.cumsum(s ** 2) / tot, 0.90) + 1)
    return out
def directions():
    G = json.load(open(GRADIENT_DIRECTIONS)); out = {}; K = kept(); z = Z()
    for L in ODD:
        kv, kk = K[(L, "V")], K[(L, "K")]; DV = np.array(G["DV"][str(L)][:kv]); DK = np.array(G["DK"][str(L)][:kk]); gam = z[f"gam{L}"]
        sV = (DV @ z[f"Wv{L}"]) * gam; sK = (rope_back(DK) @ z[f"Wk{L}"]) * gam
        out[L] = dict(S=np.concatenate([np.broadcast_to(sV, (64, kv, 960)), sK], 1), DV=DV, DK=DK, kV=kv, kK=kk)
    return out
def to_dev(dirs, dev=DEVICE):
    o = {}
    for L, v in dirs.items():
        S = torch.tensor(v["S"], dtype=torch.float64); M = S.transpose(1, 2) @ torch.linalg.inv(S @ S.transpose(1, 2)); o[L] = (S.float().to(dev), M.float().to(dev))
    return o
def proj(Sd, L, X, eps):
    S = Sd[L][0]; r = X.pow(2).mean(-1, keepdim=True).add(eps).sqrt(); return torch.einsum("jkd,bjd->bjk", S, X / r)
class Editor:   # modes calib / rand
    def __init__(self, layers, tokens, eps, mode="calib", Sd=None, T=None, m=1.0, c0=None, norms=None, U=None):
        self.layers = set(layers); self.tok = tokens; self.eps = eps; self.mode = mode; self.Sd = Sd; self.T = T; self.m = m; self.c0 = c0
        self.norms = norms; self.U = U; self.rec = {}
    def apply(self, L, X):
        if L not in self.layers: return None
        X = X.float(); v = (self.tok >= 0); idx = self.tok.clamp(min=0); gi = idx[..., None].expand(-1, -1, X.shape[-1]); x = X.gather(1, gi)
        if self.mode == "calib":
            S, M = self.Sd[L]; Sj = S[idx]; Mj = M[idx]; r = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
            c = torch.einsum("bnkd,bnd->bnk", Sj, x) / r; need = self.c0[L] + self.m * self.T[L][None, None] - c
            delta = r * torch.einsum("bndk,bnk->bnd", Mj, need)
        else: delta = self.norms[L][..., None] * self.U[L][:, None, :]
        delta = delta * v[..., None]; self.rec[L] = delta.detach(); out = X.clone(); out.scatter_add_(1, gi, delta); return out
def unit_rand(seed, n, d=960, dev=DEVICE):
    g = np.random.default_rng(seed); u = g.standard_normal((n, d)); u /= np.linalg.norm(u, axis=1, keepdims=True); return torch.tensor(u, dtype=torch.float32, device=dev)
def pad_tokens(lists, dev=DEVICE):
    n = max(len(t) for t in lists); return torch.tensor([list(t) + [-1] * (n - len(t)) for t in lists], device=dev)
class Expert:
    """the pulled-back read directions and the block calibration T of the red policy."""
    def __init__(self, T=None, dev=DEVICE):
        self.Sd = to_dev(directions(), dev)
        if T is None: T = json.load(open(BLOCK_CALIBRATION))["T"]
        self.T = {int(L): torch.tensor(v, dtype=torch.float32, device=dev) for L, v in T.items()}

# ---------------------------------------------------------------- the full vision-language model (prefix KV cache)
class VLM:
    def __init__(self, dev=DEVICE):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        HFP = snapshot_download(policy_tools.BASE["backbone"], revision=policy_tools.BASE["backbone_revision"])
        self.tok = AutoProcessor.from_pretrained(HFP).tokenizer
        self.m = AutoModelForImageTextToText.from_pretrained(HFP, torch_dtype=torch.float32).to(dev).eval(); self.m.requires_grad_(False)
        self.TM = self.m.model.text_model; self.IMG_ID = self.m.config.image_token_id; self.dev = dev; self.eps = float(self.m.config.text_config.rms_norm_eps)
        self.E = self.TM.get_input_embeddings().weight.detach(); self.edit = None; self.cap = None; self.caps = {}; self.full = None
        for L in range(len(self.TM.layers)): self.TM.layers[L].register_forward_pre_hook(self._pre(L), with_kwargs=True)
        self.SP = self.tok(" ", add_special_tokens=False)["input_ids"][0]; self.DID = torch.tensor([self.tok(d, add_special_tokens=False)["input_ids"][0] for d in DIG], device=dev)
        full = self.chat("x", ""); ids = torch.tensor(full, device=dev); self.imgpos = (ids == self.IMG_ID).nonzero()[:, 0]; self.NPRE = int(self.imgpos[-1]) + 2
        self.PRE = ids[:self.NPRE]; assert self.tok.decode(self.PRE[-1:]) == "<fake_token_around_image>"
    def chat(self, q, pre):
        s = "<|im_start|>User:<fake_token_around_image><global-img>" + "<image>" * 64 + "<fake_token_around_image>" + q + "<end_of_utterance>\nAssistant:" + pre
        return self.tok(s, add_special_tokens=False)["input_ids"]
    def suffix(self, q, pre): x = self.chat(q, pre); assert x[:self.NPRE] == self.PRE.tolist(); return x[self.NPRE:]
    def _pre(self, L):
        def f(mod, args, kwargs):
            hs = args[0] if args else kwargs["hidden_states"]
            if self.full is None or hs.shape[1] != self.full: return None
            if self.cap is not None and L in self.cap: self.caps[L] = hs[:, self.imgpos].detach().float().clone()
            if self.edit is None: return None
            new = self.edit(L, hs[:, self.imgpos])
            if new is None: return None
            hs = hs.clone(); hs[:, self.imgpos] = new.to(hs.dtype)
            if args: return (hs,) + tuple(args[1:]), kwargs
            kwargs["hidden_states"] = hs; return args, kwargs
        return f
    def _kw(self, imgs):
        pix = torch.from_numpy(np.stack(imgs)).to(self.dev).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0); B = len(imgs)
        return B, dict(pixel_values=pix[:, None], pixel_attention_mask=torch.ones(B, 1, 512, 512, dtype=torch.long, device=self.dev))
    def capture(self, imgs):
        B, kw = self._kw(imgs); ii = self.PRE[None].expand(B, -1).contiguous(); self.full = self.NPRE; self.cap = set(ODD)
        with torch.no_grad(): self.m(input_ids=ii, use_cache=False, **kw)
        self.cap = None; self.full = None; return {L: self.caps[L] for L in ODD}
    def prefix(self, imgs, ed=None):
        B, kw = self._kw(imgs); ii = self.PRE[None].expand(B, -1).contiguous(); self.edit = (lambda L, X: ed.apply(L, X)) if ed is not None else None; self.full = self.NPRE
        with torch.no_grad(): o = self.m(input_ids=ii, use_cache=True, **kw)
        self.edit = None; self.full = None; return o.past_key_values.to_legacy_cache()
    def _fwd(self, emb, cache, keep=2):
        from transformers import DynamicCache
        B = emb.shape[0]; am = torch.ones(B, cache[0][0].shape[2] + emb.shape[1], dtype=torch.long, device=self.dev)
        o = self.m.model(inputs_embeds=emb, past_key_values=DynamicCache.from_legacy_cache(cache), attention_mask=am, use_cache=True)
        return self.m.lm_head(o.last_hidden_state[:, -keep:]).float()
    def digit_lp(self, cache, suf_emb):
        """suf_emb (B, L, 960) -> (B, 8) log P(answer = digit d) = logaddexp(lp0[d], lp0[' '] + lp1[d | ' '])."""
        B = suf_emb.shape[0]; x = torch.cat([suf_emb, self.E[self.SP][None, None].expand(B, 1, -1)], 1)
        lg = torch.log_softmax(self._fwd(x, cache), -1); lp0, lp1 = lg[:, 0], lg[:, 1]
        return torch.logaddexp(lp0[:, self.DID], lp0[:, self.SP:self.SP + 1] + lp1[:, self.DID])
    def greedy(self, cache, suf_ids, n=4):
        from transformers import DynamicCache
        B = cache[0][0].shape[0]; c = DynamicCache.from_legacy_cache(cache); x = self.E[torch.tensor(suf_ids, device=self.dev)][None].expand(B, -1, -1); out = []; L0 = cache[0][0].shape[2]
        with torch.no_grad():
            for k in range(n):
                am = torch.ones(B, L0 + x.shape[1], dtype=torch.long, device=self.dev)
                o = self.m(inputs_embeds=x, past_key_values=c, attention_mask=am, use_cache=True); c = o.past_key_values
                t = o.logits[:, -1].argmax(-1); out.append(t); L0 += x.shape[1]; x = self.E[t][:, None]
        return self.tok.batch_decode(torch.stack(out, 1), skip_special_tokens=True)
    def answer(self, cache, suf_ids):
        with torch.no_grad():
            emb = self.E[torch.tensor(suf_ids, device=self.dev)][None].expand(cache[0][0].shape[0], -1, -1); lp = self.digit_lp(cache, emb).cpu().numpy()
        return self.greedy(cache, suf_ids), lp

def parse(text):
    """earliest bare digit 1-8 (word-boundary on letters/digits); None if none."""
    t = text.lower(); best = (10 ** 9, None)
    for d in DIG:
        m = re.search(r"(?<![a-z0-9])" + d + r"(e?s)?(?![a-z0-9])", t)
        if m and m.start() < best[0]: best = (m.start(), d)
    return best[1]

# ---------------------------------------------------------------- questions
SELECTION_Q = ("One of the numbers has a black box around it. Which number has the black box around it? Answer with the number.", " The boxed number is:")
PAPER_Q = ("Which number is marked in this image? Answer with the number.", " The marked number is:")
def prompts(V): return {"selection": V.suffix(*SELECTION_Q), "paper": V.suffix(*PAPER_Q)}
