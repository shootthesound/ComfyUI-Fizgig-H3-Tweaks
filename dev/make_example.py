"""Build example_workflows/h3_tweaks_text_to_video.json: ComfyUI's built-in "MiniMax H3: Text to
Video" template flattened — the same model files and node definitions, every node on the main
canvas (no subgraph), the template's Lightning LoRA switched on at its 8 steps, and Fizgig H3
Tweaks between the LoRA and the sampler.

    <ComfyUI>/venv/Scripts/python.exe custom_nodes/ComfyUI-Fizgig-H3-Tweaks/dev/make_example.py
"""
import copy, json, os, uuid

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
COMFY = os.path.abspath(os.path.join(PACK, "..", ".."))
SRC = os.path.join(COMFY, "venv", "Lib", "site-packages", "comfyui_workflow_templates_json",
                   "templates", "video_minimax_h3_t2v.json")
OUT = os.path.join(PACK, "example_workflows", "h3_tweaks_text_to_video.json")

tpl = json.load(open(SRC, encoding="utf-8"))
sg = tpl["definitions"]["subgraphs"][0]
T = {n["id"]: n for n in sg["nodes"]} | {n["id"]: n for n in tpl["nodes"]}
top = [n for n in tpl["nodes"] if n["id"] == 140][0]["widgets_values"]
PROMPT, SEED = top[0], top[4]

nodes, links = [], []
nid = [0]
lid = [0]


def clone(src_id, pos, widgets=None, title=None):
    n = copy.deepcopy(T[src_id])
    nid[0] += 1
    n["id"] = nid[0]
    n["pos"] = pos
    for k in ("order",):
        n[k] = nid[0]
    for i in n.get("inputs", []):
        i["link"] = None
    for o in n.get("outputs", []):
        o["links"] = []
    if widgets is not None:
        n["widgets_values"] = widgets
    if title:
        n["title"] = title
    elif "title" in n:
        del n["title"]
    nodes.append(n)
    return n


def link(a, a_slot, b, b_input_name):
    lid[0] += 1
    out = a["outputs"][a_slot]
    tgt = next(k for k, i in enumerate(b["inputs"]) if i["name"] == b_input_name)
    out["links"].append(lid[0])
    b["inputs"][tgt]["link"] = lid[0]
    links.append([lid[0], a["id"], a_slot, b["id"], tgt, out["type"]])


# loaders (column 1)
unet = clone(127, [0, 0], ["minimax_h3_fl2va_pruned_int8_convrot.safetensors", "default"])
lora = clone(134, [0, 150], ["minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", 1],
             title="Lightning LoRA (8 steps) — bypass it for 20 steps without")
clip = clone(128, [0, 300], ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "minimax", "default"])
vae = clone(119, [0, 470], ["minimax_h3_video_vae_fp16.safetensors"])
avae = clone(120, [0, 580], ["minimax_h3_audio_vae_fp32.safetensors"])

# Fizgig H3 Tweaks (column 2, top)
tweaks = {
    "id": None, "type": "FizgigH3Tweaks", "pos": [460, 0], "size": [400, 240], "flags": {}, "order": 0,
    "mode": 0,
    "inputs": [{"localized_name": "model", "name": "model", "type": "MODEL", "link": None}],
    "outputs": [{"localized_name": "model", "name": "model", "type": "MODEL", "links": []}],
    "properties": {"Node name for S&R": "FizgigH3Tweaks"},
    "widgets_values": [0.15, "stable across frames", 0.0, 0.0, False],
}
nid[0] += 1; tweaks["id"] = nid[0]; tweaks["order"] = nid[0]; nodes.append(tweaks)

res = clone(115, [460, 300], ["16:9 (Widescreen)", 0.4, 32])
cond = clone(131, [460, 460])
cond["widgets_values"] = [PROMPT, 832, 480, 124]

# sampling (column 3)
noise = clone(129, [960, 0], [SEED, "fixed"])
guider = clone(126, [960, 150])
ksel = clone(123, [960, 280], ["res_multistep"])
sched = clone(124, [960, 400], ["simple", 8, 1])
sampler = clone(125, [1360, 0])

# decode + save (column 4)
dec = clone(122, [1700, 0])
adec = clone(121, [1700, 120])
vid = clone(130, [1700, 250], [24, 8])
save = clone(92, [2020, 0], ["video/Fizgig_H3_Tweaks", "auto", "auto"])

link(unet, 0, lora, "model")
link(lora, 0, tweaks, "model")
link(tweaks, 0, guider, "model")
link(tweaks, 0, sched, "model")
link(clip, 0, cond, "clip")
link(vae, 0, cond, "vae")
link(res, 0, cond, "width")
link(res, 1, cond, "height")
link(cond, 0, guider, "conditioning")
link(noise, 0, sampler, "noise")
link(guider, 0, sampler, "guider")
link(ksel, 0, sampler, "sampler")
link(sched, 0, sampler, "sigmas")
link(cond, 1, sampler, "latent_image")
link(sampler, 0, dec, "samples")
link(vae, 0, dec, "vae")
link(sampler, 0, adec, "samples")
link(avae, 0, adec, "vae")
link(dec, 0, vid, "images")
link(adec, 0, vid, "audio")
link(vid, 0, save, "video")

note = {
    "id": nid[0] + 1, "type": "MarkdownNote", "pos": [460, -420], "size": [720, 380], "flags": {},
    "order": 0, "mode": 0, "inputs": [], "outputs": [], "title": "Note: Fizgig H3 Tweaks",
    "properties": {}, "color": "#432", "bgcolor": "#653",
    "widgets_values": [
        "## Fizgig H3 Tweaks — text to video\n\n"
        "ComfyUI's built-in **MiniMax H3: Text to Video** template, flattened (no subgraph): the same "
        "model files and settings, with the Lightning LoRA on at 8 steps, and **Fizgig H3 Tweaks** "
        "between the LoRA and the sampler.\n\n"
        "- **High Freq Detail** 0.15 — crisper skin; negative values smooth it.\n"
        "- **Scene Variation** 0 — try ±0.1-0.3 to re-roll an almost-right render.\n"
        "- **Prompt Strength** 0 — raise it if named things don't come through.\n\n"
        "The tweaks were tuned on 6-8 step Turbo renders. They act on fixed step numbers (detail from "
        "step 4, variation on steps 1-2), so on a 20-step render without the LoRA the detail covers most "
        "of the render — use a lower value there.\n\n"
        "Models: see the template's *Model Links* note (Templates → MiniMax H3: Text to Video)."],
}
nodes.append(note)

wf = {
    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, "fizgig-h3-tweaks/t2v")), "revision": 0,
    "last_node_id": note["id"], "last_link_id": lid[0], "nodes": nodes, "links": links,
    "groups": [], "config": {}, "extra": {"ds": {"scale": 0.6, "offset": [300, 600]}}, "version": 0.4,
}

# consistency: every link is referenced at both ends
ids = {n["id"]: n for n in nodes}
for l in links:
    assert l[0] in ids[l[1]]["outputs"][l[2]]["links"], l
    assert ids[l[3]]["inputs"][l[4]]["link"] == l[0], l
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(wf, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("wrote", OUT, len(nodes), "nodes,", len(links), "links")
