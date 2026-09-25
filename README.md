# Fizgig H3 Tweaks

One training-free node for MiniMax H3 (category **Fizgig**): **Fizgig H3 Tweaks**. A model patch — put it after your LoRAs, before the sampler. Every control is off at 0 except High Freq Detail.

| Control | Default | What it does (tested 25 Sep 2026, 6-step Turbo, int8 base, 512x512x22) |
|---|---|---|
| **High Freq Detail** | 0.15 | Fine detail from the deep blocks on the late steps. Above 0: crisper pores, freckles, lashes (0.15-0.3 clean; 0.6 adds contrast pop). Below 0: smoother, softer skin. |
| **↳ High Freq Detail mode** | stable across frames | Sub-control of High Freq Detail only. *stable*: only detail that is the same in every frame (measured +7% shimmer at 0.3); *per frame*: each frame's own (+20% shimmer on clips — fine for stills). |
| **Motion** | 0 | Below 0: calmer, steadier clips (-0.5 = -76% motion, clean). Above 0 smears quickly, so it stops at +0.2. Acts from step 3. |
| **Local Contrast** | 0 | Mid-band punch, coarser than detail. Gentle — +0.5 is still subtle. Below 0: flatter. |
| **Composition** | 0 | The broad layout on the first two steps. In testing it reshuffled the scene (like a nearby seed) rather than making it bolder — being evaluated. |
| **Prompt Strength** | 0 | How much every video/audio token takes from the prompt (H3 Turbo has no CFG). Above 0: named things come through more strongly; below 0: looser. |
| **report** | off | Console lines per step and block. |

Where each control acts is fixed at the tested blocks/steps (Detail and Local Contrast: blocks 40-49, steps 4+; Motion: 20-49, steps 3-5; Composition: 20-49, steps 1-2; Prompt Strength: everywhere). Two Tweaks nodes in a chain add together. Block Skip / Token Route on the same blocks are replaced by it there (ComfyUI allows one block patch).
