"""Loading the toy policy and running it with hooks.

Prefix layout: slots 0-63 = image tokens (token k covers pixels u in [64*(k%8), +64), v in [64*(k//8), +64)), slots 64-111 = language (48,
right-padded), slot 112 = state token. Position ids: image 0-63, language 64.., state 64+L.
Expert: 16 layers; even layers self-attend over [prefix KV (the vision-language model's own K/V); suffix], odd layers cross-attend to the
prefix through the expert's own k_proj/v_proj applied to the cached (already RoPE-rotated) keys and values.
Flow: x_t = t*noise + (1-t)*action, t = 1, .9, ..., .1; x <- x - 0.1 v; clean estimate xhat_k = x_t - t v (xhat_9 = output).
Model coordinate 0 = theta / 180 (start angle in the state, target angle in the action chunk).

Main entry points
  pol, m = load(ckpt)                       base SmolVLA + the trained expert weights (final or resume format)
  inp = make_inputs(pol, scs, ...)          render + tokenize + state for a list of scene dicts (scene.py)
  H = Hooks(m)                              layer/step tracking, attention override, residual edits, captures
  out = H.run(inp, noise)                   -> dict(final (B,2), xhat (10,B,2): circular means over the 50 positions; final_raw / xhat_raw:
                                            linear means; final_pos (B,50,2) / xhat_pos (10,B,50,2): per position; actions (B,50,32)). Degrees.
  pm, kv = H.prefix(inp); H.denoise(pm, kv, noise)   split run (for cache edits)"""
import os, sys
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
from torch import nn
from huggingface_hub import snapshot_download
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks

from toyvla import scene as S
from toyvla.paths import DEVICE
NJ = 1



def pos_mean(a):
    """mean over the 50 action positions of per-position degrees a (..., T, 2): circular for coordinate 0 (wrap-safe, canonical
    [-180, 180)), linear otherwise."""
    a = np.asarray(a, float); m = a.mean(-2)
    r = np.radians(a[..., 0]); m[..., 0] = (np.degrees(np.arctan2(np.sin(r).mean(-1), np.cos(r).mean(-1))) + 180) % 360 - 180
    return m

BASE = dict(model="lerobot/smolvla_base", model_revision="d9f33c94a60fb382c90dea2164c96845bd955e28",
            backbone="HuggingFaceTB/SmolVLM2-500M-Video-Instruct", backbone_revision="7b375e1b73b11138ff12fe22c8f2822d8fe03467")
N_IMG, LANG0, N_LANG = 64, 64, 48
STATE_SLOT, PREFIX = 112, 113
NSTEP = 10
TS = [1.0 - 0.1 * k for k in range(NSTEP)]


def token_box(k):
    """pixel box (u0, v0, u1, v1) of image token k (row-major 8x8)."""
    r, c = divmod(int(k), 8); return 64 * c, 64 * r, 64 * (c + 1), 64 * (r + 1)


def load(ckpt=None, device=DEVICE):
    path = snapshot_download(BASE['model'], revision=BASE['model_revision'])
    cfg = PreTrainedConfig.from_pretrained(path); cfg.device = 'cpu'; cfg.load_vlm_weights = False
    cfg.vlm_model_name = snapshot_download(BASE['backbone'], revision=BASE['backbone_revision'],
                                           allow_patterns=['*.json', '*.txt', '*.model', '*.jinja'])
    pol = SmolVLAPolicy.from_pretrained(path, config=cfg, strict=False).float().to(device); m = pol.model
    if ckpt is not None:
        sub = torch.load(ckpt, map_location='cpu', weights_only=False)
        if isinstance(sub, dict) and 'model' in sub and isinstance(sub['model'], dict): sub = sub['model']
        params = dict(m.named_parameters()); missing = set(sub) - set(params)
        assert not missing, list(missing)[:5]
        with torch.no_grad():
            for n, t in sub.items(): params[n].copy_(t.to(params[n].device, params[n].dtype))
        m.loaded_names = set(sub)
    m.eval(); m.requires_grad_(False); return pol, m


def tokenize(pol, texts, device=DEVICE):
    t = pol.language_tokenizer([x + '\n' for x in texts], padding='max_length', max_length=48, return_tensors='pt')
    return t['input_ids'].to(device), t['attention_mask'].bool().to(device)


def images_to_tensor(imgs_u8, device=DEVICE):
    """list/array of uint8 (H, W, 3) -> float (B, 3, 512, 512) in [-1, 1] (SmolVLA's SigLIP convention)."""
    x = torch.from_numpy(np.stack(imgs_u8)).to(device).permute(0, 3, 1, 2).float()
    return x.div_(127.5).sub_(1.0)


def make_inputs(pol, scs, variant='plain', bg=None, cam=None, imgs=None, prompts=None, state=None, device=DEVICE, **render_kw):
    """variant plain/struct: fixed prompt + rendered scene; text: block (u, v) prompt + arm-only image.
    imgs: optional list of uint8 images (overrides rendering); state: optional (B, 2) start angles in degrees."""
    if imgs is None:
        if variant == 'text': imgs = [S.render(sc, bg or 'plain', cam=cam, blocks=False, **render_kw) for sc in scs]
        else: imgs = [S.render(sc, bg or variant, cam=cam, **render_kw) for sc in scs]
    if prompts is None: prompts = [S.text_prompt(sc) for sc in scs] if variant == 'text' else [S.PROMPT] * len(scs)
    ids, mk = tokenize(pol, prompts, device)
    B = len(scs)
    ang = np.array([[sc['t1s'], sc['t2s']] for sc in scs], float) if state is None else np.asarray(state, float).reshape(B, -1)
    st = torch.zeros(B, 32, device=device)
    st[:, :NJ] = torch.tensor(ang[:, :NJ] / 180, device=device, dtype=torch.float32)
    x = images_to_tensor(imgs, device)
    return dict(images=[x], img_masks=[torch.ones(B, dtype=torch.bool, device=device)], ids=ids, mk=mk, state=st,
                imgs_u8=imgs, prompts=prompts, scs=scs,
                target=np.array([[sc['t1'], sc['t2']] for sc in scs], float))


def subset(inp, idx):
    idx = np.asarray(idx); ti = torch.as_tensor(idx, device=inp['state'].device)
    return dict(images=[x[ti] for x in inp['images']], img_masks=[x[ti] for x in inp['img_masks']], ids=inp['ids'][ti], mk=inp['mk'][ti],
                state=inp['state'][ti], imgs_u8=[inp['imgs_u8'][i] for i in idx], prompts=[inp['prompts'][i] for i in idx],
                scs=[inp['scs'][i] for i in idx], target=inp['target'][idx])


def noise_for(B, seed=11, shared=True, device=DEVICE):
    """shared=True: one noise draw for every case (isolates input dependence); else per-case draws seeded seed+i."""
    if shared:
        return torch.randn(1, 50, 32, device=device, generator=torch.Generator(device=device).manual_seed(seed)).expand(B, -1, -1).clone()
    return torch.stack([torch.randn(50, 32, device=device, generator=torch.Generator(device=device).manual_seed(seed + i)) for i in range(B)])


def angle_error(pred_deg, target_deg):
    """circular |error| in degrees."""
    return S.angle_err(pred_deg, target_deg)


def ee_error(pred_deg, scs):
    """tip distance (px) to the block centre, and inside-block success."""
    return S.ee_err(pred_deg, scs)


class Hooks:
    """One instance per model. Tracks (phase, layer, step); optional attention override and residual edits in the denoise pass.
    attn_fn(layer, step, p, q, k, v, mask) -> new probs (B, H, Q, K) or None   (expert attention, denoise phase only)
    vlm_attn_fn(layer, p, q, k, v, mask)  -> new probs or None                 (VLM prefix attention)
    resid_fn(layer, step, h) -> new h or None   edits the TRUE expert residual entering layer `layer` (16 = final norm input)
    cap: dict filled by capture hooks; use H.capture(...) to register named captures."""

    def __init__(self, m):
        self.m, self.ve = m, m.vlm_with_expert; self.ex = self.ve.lm_expert; self.vlm = self.ve.get_vlm_model().text_model
        self.phase, self.layer, self.step = None, None, -1
        self.attn_fn = self.vlm_attn_fn = self.resid_fn = None
        self.cap, self._caps = {}, []
        ve = self.ve; self._orig_eager = ve.eager_attention_forward; ve.eager_attention_forward = self._eager
        self._olds = (ve.forward_attn_layer, ve.forward_cross_attn_layer)
        ve.forward_attn_layer, ve.forward_cross_attn_layer = self._wrap(self._olds[0]), self._wrap(self._olds[1])
        self._nh = self.ex.norm.register_forward_pre_hook(self._final_pre)

    # ---- plumbing
    def _wrap(self, old):
        def call(*a, **kw):
            self.layer = a[2]; emb = a[1]
            if self.phase == 'denoise' and self.resid_fn is not None and len(emb) > 1 and emb[1] is not None:
                h = self.resid_fn(self.layer, self.step, emb[1])
                if h is not None: emb[1] = h
            return old(*a, **kw)
        return call

    def _final_pre(self, mod, args):
        if self.phase == 'denoise' and self.resid_fn is not None:
            h = self.resid_fn(16, self.step, args[0])
            if h is not None: return (h,)

    def _eager(self, mask, bs, hd, q, k, v):
        fn = self.attn_fn if self.phase == 'denoise' else (self.vlm_attn_fn if self.phase == 'prefix' else None)
        if fn is None and not self._want_probs(): return self._orig_eager(mask, bs, hd, q, k, v)
        Hn = q.shape[2]; KH = k.shape[2]; g = Hn // KH; Sk = k.shape[1]
        k2 = k[:, :, :, None, :].expand(bs, Sk, KH, g, hd).reshape(bs, Sk, Hn, hd)
        v2 = v[:, :, :, None, :].expand(bs, Sk, KH, g, hd).reshape(bs, Sk, Hn, hd)
        att = torch.matmul(q.float().transpose(1, 2), k2.float().transpose(1, 2).transpose(2, 3)) * hd ** -0.5
        self._mask = mask; att_raw = att
        att = torch.where(mask[:, None], att, torch.finfo(att.dtype).min); p = nn.functional.softmax(att, dim=-1)
        self._record_probs(p, att_raw)
        if fn is not None:
            p2 = fn(self.layer, self.step, p, q, k2, v2, mask) if self.phase == 'denoise' else fn(self.layer, p, q, k2, v2, mask)
            if p2 is not None: p = p2
        return torch.matmul(p.to(v2.dtype), v2.permute(0, 2, 1, 3)).permute(0, 2, 1, 3).reshape(bs, -1, Hn * hd)

    # attention-pattern recording: self.rec_probs = {'phase': 'denoise', 'steps': {0}, 'layers': None} -> self.cap[f'probs_s{k}_l{L}']
    rec_probs = None
    def _want_probs(self):
        r = self.rec_probs
        return r is not None and r.get('phase', 'denoise') == self.phase and (self.phase == 'prefix' or self.step in r.get('steps', {0})) \
            and (r.get('layers') is None or self.layer in r['layers'])
    def _record_probs(self, p, att):
        if not self._want_probs(): return
        key = f"probs_l{self.layer}" if self.phase == 'prefix' else f"probs_s{self.step}_l{self.layer}"
        red = self.rec_probs.get('reduce', 'mean_q'); mask = self._mask
        valid = mask.any(-1)[:, None, :, None].float()                           # (B,1,Q,1): query rows with >= 1 visible key
        if red == 'mean_q': self.cap[key] = ((p.float() * valid).sum(2) / valid.sum(2).clamp_min(1)).cpu()   # padding rows excluded
        else: self.cap[key] = p.float().cpu()
        if self.rec_probs.get('logits'):
            lg = torch.where(mask[:, None], att.float(), torch.full_like(att.float(), float('nan')))   # masked entries -> nan
            self.cap['logits_' + key[6:]] = (torch.nanmean(lg, dim=2) if red == 'mean_q' else lg).cpu()

    def capture(self, what=('resid', 'attn', 'mlp'), steps=(0,), layers=range(16), reduce='mean', positions=None):
        """register expert captures at the given flow steps. reduce: 'mean' over 50 action positions | 'none' | 'last'.
        keys: s{step}_{resid|attn|mlp|mlph}_l{L}; resid_l16 = final pre-norm residual."""
        steps = set(steps)
        def red(x):
            x = x[:, -50:].float()
            return x.mean(1) if reduce == 'mean' else (x[:, -1] if reduce == 'last' else x)
        def mk(key, clone=True):
            def f(x):
                if self.phase == 'denoise' and self.step in steps: self.cap[f's{self.step}_{key}'] = red(x).cpu()
            return f
        for L in layers:
            lay = self.ex.layers[L]
            if 'resid' in what:
                f = mk(f'resid_l{L}'); self._caps.append(lay.input_layernorm.register_forward_pre_hook(lambda mod, a, f=f: f(a[0])))
            if 'attn' in what:
                f = mk(f'attn_l{L}'); self._caps.append(lay.self_attn.o_proj.register_forward_hook(lambda mod, a, o, f=f: f(o)))
            if 'mlp' in what:
                f = mk(f'mlp_l{L}'); self._caps.append(lay.mlp.register_forward_hook(lambda mod, a, o, f=f: f(o)))
            if 'mlph' in what:
                f = mk(f'mlph_l{L}'); self._caps.append(lay.mlp.down_proj.register_forward_pre_hook(lambda mod, a, f=f: f(a[0])))
        if 'resid' in what:
            f = mk('resid_l16'); self._caps.append(self.ex.norm.register_forward_pre_hook(lambda mod, a, f=f: f(a[0])))

    def capture_prefix(self, layers=range(16), slots=None, what=('resid',)):
        """VLM prefix residual entering VLM layer L (and L=16: final pre-norm) at prefix slots (default all 113)."""
        def mk(key):
            def f(x):
                if self.phase == 'prefix': self.cap[key] = (x.float() if slots is None else x[:, slots].float()).cpu()
            return f
        for L in layers:
            f = mk(f'pre_resid_l{L}'); self._caps.append(self.vlm.layers[L].input_layernorm.register_forward_pre_hook(lambda mod, a, f=f: f(a[0])))
        f = mk('pre_resid_l16'); self._caps.append(self.vlm.norm.register_forward_pre_hook(lambda mod, a, f=f: f(a[0])))

    def clear(self):
        for h in self._caps: h.remove()
        self._caps = []; self.cap = {}; self.rec_probs = None; self.attn_fn = self.vlm_attn_fn = self.resid_fn = None

    # ---- running
    def prefix(self, inp):
        m = self.m
        with torch.no_grad(), torch.autocast(DEVICE.split(':')[0], dtype=torch.bfloat16):
            embs, pm, am = m.embed_prefix(inp['images'], inp['img_masks'], inp['ids'], inp['mk'], state=inp['state'])
            a2 = make_att_2d_masks(pm, am); pos = torch.cumsum(pm, dim=1) - 1
            self.phase = 'prefix'
            try:
                _, kv = self.ve.forward(attention_mask=a2, position_ids=pos, past_key_values=None, inputs_embeds=[embs, None],
                                        use_cache=True, fill_kv_cache=True)
            finally: self.phase = None
        return pm, kv

    def denoise(self, pm, kv, noise, xt_edit=None):
        """xt_edit(step, x) -> None (edit x in place) lets you swap/perturb x_t before a step."""
        m = self.m; xs = []; x = noise.clone(); B = x.shape[0]
        dt = torch.tensor(-1.0 / NSTEP, dtype=torch.float32, device=x.device); time = torch.tensor(1.0, dtype=torch.float32, device=x.device)
        with torch.no_grad(), torch.autocast(DEVICE.split(':')[0], dtype=torch.bfloat16):
            for k in range(NSTEP):                       # identical arithmetic to VLAFlowMatching.sample_actions
                if xt_edit is not None: xt_edit(k, x)
                self.phase, self.step = 'denoise', k
                try: v = m.denoise_step(pm, kv, x, time.expand(B))
                finally: self.phase = None
                xs.append(_to_deg((x - time * v)[:, :, :2].float()))      # per position (B, 50, 2) raw degrees
                x += dt * v; time += dt
        self.step = -1
        fp = _to_deg(x[:, :, :2].float()); xp = np.stack(xs)
        return dict(final=pos_mean(fp), xhat=pos_mean(xp), final_raw=fp.mean(1), xhat_raw=xp.mean(2), final_pos=fp, xhat_pos=xp,
                    actions=x.float())

    def run(self, inp, noise, xt_edit=None):
        pm, kv = self.prefix(inp); return self.denoise(pm, kv, noise, xt_edit)


def _to_deg(t):
    """model coordinates (B, 50, 2) -> degrees."""
    return t.cpu().numpy() * 180


def run_batched(H, inp, noise, bs=64, **kw):
    """H.run over minibatches; concatenates final/xhat."""
    B = len(inp['scs']); out = {}
    for j in range(0, B, bs):
        idx = np.arange(j, min(B, j + bs)); o = H.run(subset(inp, idx), noise[idx[0]:idx[-1] + 1], **kw)
        for k in ('final', 'final_raw', 'final_pos'): out.setdefault(k, []).append(o[k])
        for k in ('xhat', 'xhat_raw', 'xhat_pos'): out.setdefault(k, []).append(o[k])
    return {k: np.concatenate(v, 1 if k.startswith('xhat') else 0) for k, v in out.items()}
