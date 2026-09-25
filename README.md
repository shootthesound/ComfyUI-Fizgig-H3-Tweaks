# Fizgig H3 Tweaks

Training-free nudges for MiniMax H3 (category **Fizgig**). **Fizgig H3 Tweaks** is a model patch: put it after your LoRAs, before the sampler. On the blocks and steps you choose, it scales the fine detail each block writes into the video — text and audio are never touched.

| Input | What it does |
|---|---|
| **High Freq Detail** (default 0.15) | Scales the fine, high-frequency part of the chosen blocks' update. Above 0: crisper pores, freckles, lashes (0.15-0.3 clean; 0.6 adds contrast and saturation pop). Below 0: smoother, softer skin. |
| **blocks** (default `40-49`) | Which blocks. The deep blocks carry the fine structure. |
| **steps** (default `4+`) | Which sampling steps — `4+` = step 4 to the end, where detail forms (early steps set composition). |
| **report** | One console line per step. |

0 leaves the model untouched. Tested on the 6-step Turbo regime (int8 base, Turbo LoRA 0.75, 512×512, 22 frames): see the comparison sheet in the Fizgig research notes. Doesn't combine with Block Skip / Token Route on the same blocks — the later node's patch replaces the earlier one's there.

More tweaks (contrast, composition strength, motion) are planned on the same mechanism.
