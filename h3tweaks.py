"""Fizgig H3 Tweaks — training-free nudges to what MiniMax H3's blocks write, per step.

A model patch (after your LoRAs, before the sampler). On the steps and blocks you name, each
block's update to the VIDEO tokens (what the block adds to the stream) is re-weighted:

- detail_gain: the whole update, scaled by (1 + gain). Aimed at the late steps and deep blocks,
  where the fine structure is written (the Fizgig fine map, 24-25 Sep 2026: detail forms on the
  last steps; the deep blocks carry the most and change the most). The FreeU idea — turn up the
  part of the network that carries the look — aimed with a map instead of applied everywhere.
- detail_hf_gain: only the high-frequency part of the update across each frame's token grid
  (update minus its Gaussian blur over neighbouring tokens), scaled by (1 + gain). Sharper
  local structure. One token covers 32x32 image pixels at 512, so this works at that scale;
  texture finer than a token lives inside each token's features and is what detail_gain reaches.

Text and audio rows are never touched. Both gains at 0 = the block unchanged, bit for bit.
Tested 25 Sep 2026 (6-step Turbo @0.75, int8 base, 512x512x22, blocks 40-49, steps 4+, same
seed): detail_hf_gain 0.3 = crisper pores / freckles / lashes with no crunch (the default);
0.6 = clean but punchy (more contrast and saturation). detail_gain 0.1 = subtle and clean, 0.2 =
grainy papery skin, 0.35 = broken. Both sit on a knife edge — go up in small steps.
"""
from __future__ import annotations

import math
import re
from typing import Dict, Set

import torch
import torch.nn.functional as F
from comfy_api.latest import io

NUM_BLOCKS = 50


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


def build_patch(blocks: Set[int], steps_spec: str, gain: float, hf_gain: float, report: bool):
    state = {"n": None, "steps": set(), "last": None, "hits": 0}

    def patch(args, extra):
        original = extra["original_block"]
        to = args.get("transformer_options", {})
        step, n = _step_index(to)
        layout = args.get("layout")
        if step is None or layout is None:
            return original(args)
        if state["n"] != n:
            state.update(n=n, steps=_steps(steps_spec, n), last=None, hits=0)
        if step not in state["steps"]:
            return original(args)
        va, vb, _ = next(s for s in layout.segments if s[2] == "video")
        h_in = args["img"][va:vb].clone()              # blocks update the stream in place
        out = original(args)["img"]
        delta = out[va:vb].float() - h_in.float()
        new = delta * (1.0 + gain)
        if hf_gain:
            _, T, lh, lw, _ = layout.signature
            gh, gw = lh // 2, lw // 2                   # 2x2 patches -> token grid
            if T * gh * gw == delta.shape[0]:
                d4 = delta.reshape(T, gh, gw, -1)
                high = (d4 - _blur_grid(d4)).reshape(delta.shape)
                new = new + hf_gain * high
        out[va:vb] = (h_in.float() + new).to(out.dtype)
        if report:
            if state["last"] != step:
                if state["last"] is not None:
                    print(f"[Fizgig H3 Tweaks] step {state['last']}/{n}: {state['hits']} block(s) tweaked", flush=True)
                state.update(last=step, hits=0)
            state["hits"] += 1
        return {"img": out}

    return patch


class FizgigH3Tweaks(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FizgigH3Tweaks",
            display_name="Fizgig H3 Tweaks",
            category="Fizgig",
            search_aliases=["detail boost", "skin detail", "freeu", "sharpen", "minimax h3 detail"],
            description=(
                "Training-free: turns up (or down) what chosen MiniMax H3 blocks write on chosen "
                "steps. detail_gain scales the whole update of the late, deep blocks where fine "
                "structure forms; detail_hf_gain scales only its high-frequency part. A model patch: "
                "after your LoRAs, before the sampler. Experimental — start small."
            ),
            inputs=[
                io.Model.Input("model", tooltip="The H3 model with any LoRAs applied."),
                io.Float.Input("detail_gain", default=0.0, min=-0.3, max=0.3, step=0.01,
                               tooltip="Scale the chosen blocks' whole update by (1 + this). 0 = off. "
                                       "Tested: 0.1 is subtle and clean, 0.2 already grainy. Below 0: softer."),
                io.Float.Input("detail_hf_gain", default=0.3, min=-1.0, max=1.0, step=0.05,
                               tooltip="Gain on only the high-frequency part of the update (across "
                                       "neighbouring tokens) - the skin lever. Tested: 0.2-0.4 = crisper "
                                       "pores, freckles, lashes; 0.6 adds contrast and saturation pop. 0 = off."),
                io.String.Input("blocks", default="40-49",
                                tooltip="Which blocks (0-49), e.g. '40-49' or '30-49'. The deep blocks "
                                        "carry the fine structure."),
                io.String.Input("steps", default="4+",
                                tooltip="Which sampling steps (from 1), e.g. '4+' = step 4 to the end. "
                                        "Late steps are where detail forms; early steps set composition."),
                io.Boolean.Input("report", default=False, tooltip="One console line per step."),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, detail_gain=0.0, detail_hf_gain=0.3, blocks="40-49", steps="4+",
                report=False) -> io.NodeOutput:
        dm = getattr(getattr(model, "model", None), "diffusion_model", None)
        if dm is None or type(dm).__name__ != "MiniMaxH3Model":
            raise ValueError("Fizgig H3 Tweaks only knows MiniMax H3 — connect an H3 model.")
        bl = _parse_ranges(blocks, 0, NUM_BLOCKS - 1, "block"); bl.discard(-1)
        _parse_ranges(steps, 1, 10_000, "step")
        m = model.clone()
        if bl and (detail_gain or detail_hf_gain):
            patch = build_patch(bl, steps, float(detail_gain), float(detail_hf_gain), bool(report))
            existing = (m.model_options.get("transformer_options", {})
                        .get("patches_replace", {}).get("dit", {}))
            clash = sorted(i for i in bl if ("double_block", i) in existing)
            if clash:
                print(f"[Fizgig H3 Tweaks] note: blocks {clash} were already patched (Block Skip / "
                      "Token Route?); this node replaces those patches there.", flush=True)
            for i in sorted(bl):
                m.set_model_patch_replace(patch, "dit", "double_block", i)
        return io.NodeOutput(m)
