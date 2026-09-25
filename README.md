# Fizgig H3 Tweaks

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20me%20a%20coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/lorasandlenses)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

One node for **MiniMax H3** in ComfyUI that nudges what the model draws while it samples — no training, no extra models. Detail and contrast in one dial (crisper and punchier, or softer and airier), a gentle re-roll of an almost-right render, and a prompt-strength dial for H3's CFG-free Turbo renders.

## How do I install it?

```
cd ComfyUI/custom_nodes
git clone https://github.com/shootthesound/ComfyUI-Fizgig-H3-Tweaks
```

Restart ComfyUI. The node is **Fizgig H3 Tweaks**, under the **Fizgig** category. No extra Python dependencies — it runs on ComfyUI's built-in MiniMax H3 support.

## How do I use it?

It's a model patch. Put it after your LoRAs and before the sampler:

```
Load Diffusion Model → (your LoRAs) → Fizgig H3 Tweaks → BasicGuider / BasicScheduler → SamplerCustomAdvanced
```

Every control except Detail & Contrast starts at 0 (off), so turn on only what you need.

## How do I get crisper skin — or a softer, airier look?

**Detail & Contrast** (default 0.15). It moves texture and local contrast together, in both directions:

- **Above 0 — crisper and punchier.** More pores, freckles and lashes, and deeper shadows. 0.15 is typically clean; higher than 0.15 can work, but can often be overbaked.
- **Below 0 — softer and airier.** Smoother skin and lifted shadows, which can flatter a shot.

Its sub-control, **↳ Detail & Contrast mode**, matters on clips:

- **stable across frames** (default) — only adds detail that is the same in every frame, so fine texture doesn't shimmer.
- **per frame** — each frame's own detail. Fine for stills; on clips it can shimmer.

## How do I re-roll a render that's almost right?

**Scene Variation**. It works like a sub-seed: the scene rearranges a little while the overall look stays the same. Small values give small changes, and the same settings always give the same result. Try ±0.1 to ±0.3.

## How do I make the prompt come through more strongly?

**Prompt Strength**. H3's Turbo renders run without CFG, so there's normally no dial for how closely the model follows your prompt. Above 0, the things you name come through more strongly; below 0 the result is looser. Start around 0.2 and step up gradually.

## Which models and settings does it work with?

MiniMax H3 only (fl2va or ref2va) — the node tells you if another model is connected. Any step count: it reads how many steps your sampler runs and scales to it — Detail & Contrast acts on the last half of the steps, Scene Variation on the first third (at 6 steps: 4-6 and 1-2; at 4 steps: 3-4 and 1-2; at 8 steps: 5-8 and 1-3; at 20 steps: 11-20 and 1-7). With a render split across two samplers, each sampler counts its own steps. It was tested on 6-step Turbo renders; on long renders without a Turbo LoRA Detail & Contrast covers more steps, so a lower value may suit.

## Can I combine it with other nodes?

Yes — it sits alongside LoRAs and any sampler. Two Fizgig H3 Tweaks nodes in a chain add together. Other nodes that replace H3's blocks (block-skipping nodes, for example) take over the blocks they share with it, so use one or the other on the same blocks.

## Is there an example workflow?

Yes: [`example_workflows/h3_tweaks_text_to_video.json`](example_workflows/) — ComfyUI's own **MiniMax H3: Text to Video** template with everything on one canvas (no subgraph), the Lightning LoRA on at 8 steps, and Fizgig H3 Tweaks between the LoRA and the sampler. Load it from ComfyUI's Templates browser (it appears under this pack's name), or drag the JSON onto the canvas. The models are the ones the template uses — see its *Model Links* note in ComfyUI's Templates browser.

## Support

If this tool saves you time or fits into your workflow, consider
[buying me a coffee](https://buymeacoffee.com/lorasandlenses).

Your support helps me keep developing and maintaining these nodes. Members get
early access to new builds before public release.

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20me%20a%20coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/lorasandlenses)

## Author

Peter Neill — [ShootTheSound.com](https://shootthesound.com) / [UltrawideWallpapers.net](https://ultrawidewallpapers.net)

Feedback is welcome — open an issue or reach out.

## License

MIT
