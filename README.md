# Fizgig H3 Tweaks

Training-free nudges for MiniMax H3 (category **Fizgig**). **Fizgig H3 Tweaks** is a model patch: put it after your LoRAs, before the sampler. On the blocks and steps you choose, it re-weights what each block writes into the video — text and audio are never touched.

| Input | What it does |
|---|---|
| **detail_hf_gain** (default 0.3) | Boosts only the high-frequency part of the chosen blocks' update — the skin lever: crisper pores, freckles, lashes. 0.2–0.4 is the sweet spot; 0.6 adds contrast and saturation pop. |
| **detail_gain** (default 0) | Scales the chosen blocks' whole update. 0.1 is subtle and clean; 0.2 is already grainy. Below 0 softens. |
| **blocks** (default `40-49`) | Which blocks. The deep blocks carry the fine structure. |
| **steps** (default `4+`) | Which sampling steps — `4+` = step 4 to the end, where detail forms (early steps set composition). |
| **report** | One console line per step. |

Both gains at 0 leave the model untouched. Tested on the 6-step Turbo regime (int8 base, Turbo LoRA 0.75, 512×512, 22 frames): see the comparison sheet in the Fizgig research notes. Doesn't combine with Block Skip / Token Route on the same blocks — the later node's patch replaces the earlier one's there.

More tweaks (contrast, composition strength, motion) are planned on the same mechanism.
