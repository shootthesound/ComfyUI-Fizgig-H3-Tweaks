# Fizgig H3 Tweaks

One training-free node for MiniMax H3 (category **Fizgig**): **Fizgig H3 Tweaks**. A model patch — put it after your LoRAs, before the sampler. Every control is off at 0 except High Freq Detail.

| Control | Default | What it does (tested 25 Sep 2026, 6-step Turbo, int8 base, 512x512x22) |
|---|---|---|
| **High Freq Detail** | 0.15 | Fine detail from the deep blocks on the late steps. Works in both directions: above 0 = crisper pores, freckles, lashes (0.15 is typically clean; higher adds more, and contrast pop by 0.6); below 0 = smoother, softer skin. |
| **↳ High Freq Detail mode** | stable across frames | Sub-control of High Freq Detail only. *stable*: only detail that is the same in every frame (measured +7% shimmer at 0.3); *per frame*: each frame's own (+20% shimmer on clips — fine for stills). |
| **Scene Variation** | 0 | A small re-roll of an almost-right render, like a sub-seed: nudges the layout on the first two steps while keeping the overall look. Small values = small changes. |
| **Prompt Strength** | 0 | How much every video/audio token takes from the prompt (H3 Turbo has no CFG). Above 0: named things come through more strongly; below 0: looser. |
| **report** | off | Console lines per step and block. |

Where each control acts is fixed at the tested blocks/steps (High Freq Detail: blocks 40-49, steps 4+; Scene Variation: blocks 20-49, steps 1-2; Prompt Strength: everywhere). Two Tweaks nodes in a chain add together. Block Skip / Token Route on the same blocks are replaced by it there (ComfyUI allows one block patch).

## Example workflow

`example_workflows/h3_tweaks_text_to_video.json` (also in ComfyUI's Templates browser under this pack): ComfyUI's built-in **MiniMax H3: Text to Video** template flattened — the same model files and settings, every node on one canvas, no subgraph — with the Lightning LoRA on at 8 steps and Fizgig H3 Tweaks between the LoRA and the sampler. Rebuild it from the installed template with `dev/make_example.py`.
