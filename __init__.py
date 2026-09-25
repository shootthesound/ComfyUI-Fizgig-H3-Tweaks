"""ComfyUI-Fizgig-H3-Tweaks — training-free nudges to what MiniMax H3's blocks write, per step.

**Fizgig H3 Tweaks**: a model patch (after LoRAs, before the sampler) that re-weights the video
update of chosen blocks on chosen steps — detail gain and high-frequency gain. See h3tweaks.py.
"""
from comfy_api.latest import ComfyExtension


class FizgigH3TweaksExtension(ComfyExtension):
    async def get_node_list(self):
        from .h3tweaks import FizgigH3Tweaks
        return [FizgigH3Tweaks]


async def comfy_entrypoint() -> FizgigH3TweaksExtension:
    return FizgigH3TweaksExtension()
