"""Fizgig H3 Tweaks — training-free nudges to what MiniMax H3's blocks write, per step.

One model-patch node (after your LoRAs, before the sampler) with five controls, each off at 0.
Active controls go into a shared list in the model's transformer_options and ONE dispatcher
patch per block applies them all — chaining two Tweaks nodes adds both lists (Block Skip / Token
Route on the same blocks still replace it: ComfyUI allows one replace-patch per block).

Update tweaks re-weight a block's update to the VIDEO tokens (what the block adds, `d`), split
into bands over each frame's token grid (one token = 32x32 image px at 512) or over time:
  Detail          d_high = d - blur(d, 1 token)            (blocks 40-49, steps 4+)
  Local Contrast  d_mid  = blur(d, 1) - blur(d, 3)         (40-49, 4+)
  Composition     d_low  = blur(d, 3)                      (20-49, 1-2)
  Motion          d_move = d - mean over frames of d       (20-49, 3-5)
Each adds strength * band to d. Detail can instead use only the part of d_high that is the same
in every frame ("stable across frames") — sharper without shimmer.
Prompt Strength works inside attention: the text tokens' values are scaled by (1 + strength), so
every video (and audio) token takes proportionally more from the prompt. H3 Turbo runs without
CFG; this is a CFG-free adherence dial.

Text and audio rows are never modified by the update tweaks; everything at 0 = untouched.
Tested 25 Sep 2026 (6-step Turbo @0.75, int8 base, 512x512x22): Detail +0.3 = crisper pores /
freckles / lashes, no crunch; +0.6 punchy; negative smooths (Peter). Default +0.15. A whole-
update gain was tried and dropped (grainy by 0.2, broken by 0.35). Round 2 (22 renders): Local Contrast clean
and gentle; Prompt Strength promising; Motion's calm side works (-0.5 = -76% motion), its lively
side smears (capped at +0.2); Composition reshuffled the layout rather than bolding it.
"""
from __future__ import annotations

import math
import re
from typing import Set

import torch
import torch.nn.functional as F
from comfy_api.latest import io

NUM_BLOCKS = 50
KEY = "fizgig_h3_tweaks"
DETAIL_MODES = ("per frame", "stable across frames")


# ---- parsing / step detection ------------------------------------------------------------------
def _parse_ranges(text: str, lo: int, hi: int, what: str) -> Set[int]:
    """'3-12, 14' -> {3..12, 14}; 'all'; '4+' = 4 to the end (returned with -1 marker)."""
    out: Set[int] = set()
    text = (text or "").strip().lower()
    if not text:
        return out
    if text == "all":
        return set(range(lo, hi + 1))
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if m:
            a, b = sorted((int(m.group(1)), int(m.group(2))))
            out.update(range(a, b + 1)); continue
        m = re.fullmatch(r"(\d+)\s*\+", part)
        if m:
            out.add(int(m.group(1))); out.add(-1); continue
        if part.isdigit():
            out.add(int(part)); continue
        raise ValueError(f"cannot read {what} '{part}' — use numbers, ranges like 40-49, or 4+")
    for v in out:
        if v != -1 and not (lo <= v <= hi):
            raise ValueError(f"{what} {v} is out of range ({lo}-{hi})")
    return out


def _steps(spec: str, n: int) -> Set[int]:
    s = _parse_ranges(spec, 1, 10_000, "step")
    if -1 in s:
        s.discard(-1)
        s = set(range(min(s) if s else 1, n + 1))
    return {v for v in s if 1 <= v <= n}


def _step_index(to) -> tuple:
    """(1-based step, total) from the sampler's sigma bookkeeping (as in Fizgig H3 Block Skip)."""
    sig, ss = to.get("sigmas"), to.get("sample_sigmas")
    if sig is None or ss is None:
        return None, None
    try:
        s = sig.flatten()[0]
        i = int((ss.to(s.device, s.dtype) - s).abs().argmin())
        n = max(int(ss.shape[0]) - 1, 1)
        return min(i + 1, n), n
    except Exception:
        return None, None


# ---- bands ----------------------------------------------------------------------------------------
def _blur_grid(x, sigma=1.0):
    """x [T, h, w, C] -> Gaussian blur over (h, w), reflect padding."""
    T, h, w, C = x.shape
    r = max(1, int(math.ceil(sigma * 3)))
    k = torch.arange(-r, r + 1, device=x.device, dtype=torch.float32)
    k = torch.exp(-(k ** 2) / (2 * sigma ** 2)); k = k / k.sum()
    y = x.float().permute(0, 3, 1, 2).reshape(T * C, 1, h, w)
    if h > 1:
        rh = min(r, h - 1); kh = k[r - rh:r + rh + 1]; kh = kh / kh.sum()
        y = F.conv2d(F.pad(y, (0, 0, rh, rh), mode="reflect"), kh.view(1, 1, -1, 1))
    if w > 1:
        rw = min(r, w - 1); kw = k[r - rw:r + rw + 1]; kw = kw / kw.sum()
        y = F.conv2d(F.pad(y, (rw, rw, 0, 0), mode="reflect"), kw.view(1, 1, 1, -1))
    return y.reshape(T, C, h, w).permute(0, 2, 3, 1)


def band(kind, d4, mode=None):
    """d4 [T, gh, gw, C] fp32 -> the band this tweak scales, same shape."""
    if kind == "detail":
        high = d4 - _blur_grid(d4, 1.0)
        if mode == "stable across frames" and d4.shape[0] > 1:
            high = high.mean(dim=0, keepdim=True).expand_as(high)
        return high
    if kind == "contrast":
        return _blur_grid(d4, 1.0) - _blur_grid(d4, 3.0)
    if kind == "composition":
        return _blur_grid(d4, 3.0)
    if kind == "motion":
        return d4 - d4.mean(dim=0, keepdim=True) if d4.shape[0] > 1 else torch.zeros_like(d4)
    raise ValueError(kind)


# ---- prompt strength: an attention module the block can use instead of its own ---------------
class _PromptWeightedAttention:
    """Same maths as comfy.ldm.minimax.model.Attention.forward, with the text rows' values scaled."""

    def __init__(self, attn, text_len, scale):
        self.attn, self.text_len, self.scale = attn, text_len, scale

    def __call__(self, x, rope_freqs=None, transformer_options={}):
        import comfy.model_management
        import comfy.quant_ops
        from comfy.ldm.modules.attention import optimized_attention
        from comfy.ldm.minimax.model import AttentionTensorContainer
        a = self.attn
        s = x.shape[0]
        q, k, v = a.qkv_proj(x).split(a.heads * a.head_dim, dim=-1)
        v = v.reshape(s, a.heads, a.head_dim).clone()
        v[: self.text_len] *= self.scale
        if rope_freqs is not None:
            q = q.reshape(1, s, a.heads, a.head_dim).contiguous()
            k = k.reshape(1, s, a.heads, a.head_dim).contiguous()
            qw = comfy.model_management.cast_to(a.q_norm.weight, device=x.device)
            kw = comfy.model_management.cast_to(a.k_norm.weight, device=x.device)
            rot = rope_freqs.shape[-3] * 2
            q, k = comfy.quant_ops.ck.rms_rope_split_half(q, k, rope_freqs, qw, kw, epsilon=a.q_norm.eps, rot_dim=rot)
            q, k = q[0], k[0]
        else:
            q = a.q_norm(q.reshape(s, a.heads, a.head_dim))
            k = a.k_norm(k.reshape(s, a.heads, a.head_dim))
        q = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
        k = AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0))
        v = AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))
        out = optimized_attention(q, k, v, a.heads, mask=None, skip_reshape=True,
                                  transformer_options=transformer_options)
        return a.out_proj(out.squeeze(0))


# ---- the dispatcher ------------------------------------------------------------------------------
def make_dispatcher(dm):
    cache = {}

    def dispatch(args, extra):
        original = extra["original_block"]
        to = args.get("transformer_options", {})
        tweaks = to.get(KEY) or []
        layout = args.get("layout")
        step, n = _step_index(to)
        if not tweaks or step is None or layout is None:
            return original(args)
        i = int(to.get("block_index", -1))
        active = []
        for tw in tweaks:
            key = (id(tw), n)
            if key not in cache:
                cache[key] = _steps(tw["steps"], n)
            if i in tw["blocks"] and step in cache[key]:
                active.append(tw)
        if not active:
            return original(args)
        report = any(tw.get("report") for tw in active)
        call = dict(args)
        prompt = [tw for tw in active if tw["kind"] == "prompt"]
        if prompt:
            text_len = next((b - a for a, b, kind in layout.segments if kind == "text"), 0)
            scale = 1.0
            for tw in prompt:
                scale *= 1.0 + tw["strength"]
            if text_len and scale != 1.0:
                call["attention"] = _PromptWeightedAttention(dm.blocks[i].attn, text_len, scale)
        updates = [tw for tw in active if tw["kind"] != "prompt"]
        if not updates:
            out = original(call)
            if report:
                print(f"[Fizgig H3 Tweaks] step {step}/{n} block {i}: prompt x{scale:.2f}", flush=True)
            return out
        va, vb, _ = next(s for s in layout.segments if s[2] == "video")
        _, T, lh, lw, _ = layout.signature
        gh, gw = lh // 2, lw // 2                         # 2x2 patches -> token grid
        if T * gh * gw != vb - va:
            return original(call)
        h_in = args["img"][va:vb].clone()                # blocks update the stream in place
        out = original(call)["img"]
        d = out[va:vb].float() - h_in.float()
        d4 = d.reshape(T, gh, gw, -1)
        new = d4
        for tw in updates:
            new = new + tw["strength"] * band(tw["kind"], d4, tw.get("mode"))
        out[va:vb] = (h_in.float() + new.reshape(d.shape)).to(out.dtype)
        if report:
            print(f"[Fizgig H3 Tweaks] step {step}/{n} block {i}: "
                  + ", ".join(f"{tw['kind']} {tw['strength']:+.2f}" for tw in updates), flush=True)
        return {"img": out}

    return dispatch


def add_tweaks(model, tweaks, node_name="Fizgig H3 Tweaks"):
    """Append the active tweaks to the model's shared list and route their blocks through one
    dispatcher. Chaining two Tweaks nodes adds both lists."""
    dm = getattr(getattr(model, "model", None), "diffusion_model", None)
    if dm is None or type(dm).__name__ != "MiniMaxH3Model":
        raise ValueError(f"{node_name} only knows MiniMax H3 — connect an H3 model.")
    m = model.clone()
    tweaks = [t for t in tweaks if t["blocks"] and t["strength"] != 0]
    if not tweaks:
        return m
    to = m.model_options.setdefault("transformer_options", {})
    to[KEY] = list(to.get(KEY) or []) + tweaks
    existing = to.get("patches_replace", {}).get("dit", {})
    blocks = set()
    for tw in to[KEY]:
        blocks |= set(tw["blocks"])
    foreign = sorted(i for i in blocks if ("double_block", i) in existing
                     and getattr(existing[("double_block", i)], "_fizgig_tweaks", False) is False)
    if foreign:
        print(f"[{node_name}] note: blocks {foreign} carry another node's patch (Block Skip / Token "
              "Route?); the tweaks replace it there.", flush=True)
    patch = make_dispatcher(dm)
    patch._fizgig_tweaks = True
    for i in sorted(blocks):
        m.set_model_patch_replace(patch, "dit", "double_block", i)
    return m


def _tweak(kind, strength, blocks, steps, report, **extra):
    bl = _parse_ranges(blocks, 0, NUM_BLOCKS - 1, "block"); bl.discard(-1)
    return dict(kind=kind, strength=float(strength), blocks=sorted(bl), steps=steps,
                report=bool(report), **extra)


# Where each control acts — fixed at the tested values (25 Sep 2026 live tests).
WHERE = {
    "detail":      ("40-49", "4+"),    # fine structure is written by the deep blocks, late
    "motion":      ("20-49", "3-5"),   # from step 3, so step 2's layout is left alone
    "contrast":    ("40-49", "4+"),
    "composition": ("20-49", "1-2"),   # the first steps set the layout
    "prompt":      ("all", "all"),
}


# ---- the node -----------------------------------------------------------------------------------
class FizgigH3Tweaks(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FizgigH3Tweaks", display_name="Fizgig H3 Tweaks", category="Fizgig",
            search_aliases=["detail boost", "skin detail", "smooth skin", "freeu", "sharpen", "motion",
                            "calm", "local contrast", "composition", "prompt strength", "prompt adherence"],
            description="Training-free nudges for MiniMax H3, each off at 0: fine detail, motion, local "
                        "contrast, composition, prompt strength. A model patch: after your LoRAs, before "
                        "the sampler. Experimental — start small.",
            inputs=[
                io.Model.Input("model", tooltip="The H3 model with any LoRAs applied."),
                io.Float.Input("high_freq_detail", display_name="High Freq Detail", default=0.15,
                               min=-1.0, max=1.0, step=0.05,
                               tooltip="Fine detail from the deep blocks on the late steps. Above 0: crisper "
                                       "pores, freckles, lashes (0.15-0.3 clean; 0.6 adds contrast pop). "
                                       "Below 0: smoother, softer skin. 0 = off."),
                io.Combo.Input("detail_mode", display_name="  ↳ High Freq Detail mode", options=list(DETAIL_MODES),
                               default="stable across frames",
                               tooltip="Applies to High Freq Detail only (does nothing when it is 0). stable across frames: "
                                       "only detail that is the same in every frame — "
                                       "measured +7% shimmer at 0.3 vs +20% per frame. per frame: each "
                                       "frame's own detail (fine for stills)."),
                io.Float.Input("motion", display_name="Motion", default=0.0, min=-1.0, max=0.2, step=0.05,
                               tooltip="Below 0: calmer, steadier clips (-0.5 measured -76% motion, clean). "
                                       "Above 0 smears quickly, so it stops at +0.2. Acts from step 3. 0 = off."),
                io.Float.Input("local_contrast", display_name="Local Contrast", default=0.0, min=-1.0, max=1.0,
                               step=0.05,
                               tooltip="Mid-band punch, coarser than detail. Above 0: more pop; below 0: "
                                       "flatter. Gentle — +0.5 is still subtle. 0 = off."),
                io.Float.Input("composition", display_name="Composition", default=0.0, min=-0.5, max=0.5,
                               step=0.05,
                               tooltip="The broad layout on the first two steps. In testing it reshuffled the "
                                       "scene (like a nearby seed) rather than making it bolder — try it and "
                                       "see. 0 = off."),
                io.Float.Input("prompt_strength", display_name="Prompt Strength", default=0.0, min=-0.5,
                               max=1.0, step=0.05,
                               tooltip="How much every video/audio token takes from the prompt (H3 Turbo has no "
                                       "CFG). Above 0: named things come through more strongly; below 0: "
                                       "looser. 0 = off."),
                io.Boolean.Input("report", default=False, tooltip="Console lines per step and block."),
            ],
            outputs=[io.Model.Output(display_name="model")])

    @classmethod
    def execute(cls, model, high_freq_detail=0.15, detail_mode="stable across frames", motion=0.0,
                local_contrast=0.0, composition=0.0, prompt_strength=0.0, report=False) -> io.NodeOutput:
        t = [
            _tweak("detail", high_freq_detail, *WHERE["detail"], report, mode=detail_mode),
            _tweak("motion", motion, *WHERE["motion"], report),
            _tweak("contrast", local_contrast, *WHERE["contrast"], report),
            _tweak("composition", composition, *WHERE["composition"], report),
            _tweak("prompt", prompt_strength, *WHERE["prompt"], report),
        ]
        return io.NodeOutput(add_tweaks(model, t))


NODES = [FizgigH3Tweaks]
