"""Where does ICWM look when it predicts an action chunk?

For training samples (query + 5 same-viewpoint context clips, no target action) the policy generates
its action tokens greedily, exactly as at evaluation. The prompt + generated tokens are then run
through the LM once more with eager attention (causal, so the attention rows of the generated
positions equal those during generation) and, for every generated token, the attention mass that
lands on each input segment is summed (mean over heads), per layer.

Segments: system/template | clip i: o^s image, action tokens, o^e image | current observation
(agentview, wrist) | instruction text | previously generated action tokens.

    PYTHONPATH=src python diag_icwm_attention.py --ckpt <pretrained_model> --n 12
"""
import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace/yujin/icwm/lerobot/src")
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.datasets.icwm_context_sampler import ICWMViewpointMatchedDataset  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.icwm.modeling_icwm import ICWMPolicy  # noqa: E402

CTX = ["observation.icwm_context.images_start", "observation.icwm_context.images_end", "observation.icwm_context.actions"]
VISION_START, VISION_END, IM_START, IM_END = 151652, 151653, 151644, 151645


def segment_ids(ids, act_lo, act_hi, n_clips, n_query_imgs, prompt_len):
    """Label every position of the prompt with a segment name."""
    seg = ["system/template"] * len(ids)
    # image blocks in order: clip i start, clip i end (x n_clips), then the query cameras
    blocks, j = [], 0
    while j < prompt_len:
        if ids[j] == VISION_START:
            k = j
            while ids[k] != VISION_END:
                k += 1
            blocks.append((j, k + 1))
            j = k + 1
        else:
            j += 1
    assert len(blocks) == 2 * n_clips + n_query_imgs, f"found {len(blocks)} image blocks"
    for c in range(n_clips):
        s0, s1 = blocks[2 * c]
        e0, e1 = blocks[2 * c + 1]
        for p in range(s0, s1):
            seg[p] = "context o^s (start img)"
        for p in range(e0, e1):
            seg[p] = "context o^e (end img)"
        for p in range(s1, e0):  # the clip's FAST action tokens sit between its two images
            if act_lo <= ids[p] < act_hi:
                seg[p] = "context action tokens"
    names = ["current obs: agentview", "current obs: wrist"]
    for q in range(n_query_imgs):
        a, b = blocks[2 * n_clips + q]
        for p in range(a, b):
            seg[p] = names[q] if q < len(names) else f"current obs {q}"
    # instruction: user-turn text after the last query image, up to the user <|im_end|>
    last_img_end = blocks[-1][1]
    p = last_img_end
    while p < prompt_len and ids[p] != IM_END:
        seg[p] = "instruction"
        p += 1
    for p in range(prompt_len, len(ids)):
        seg[p] = "generated action tokens"
    return seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    raw_cfg = json.loads((Path(args.ckpt) / "config.json").read_text())
    raw_cfg.setdefault("type", "icwm")
    scratch = tempfile.mkdtemp()
    (Path(scratch) / "config.json").write_text(json.dumps(raw_cfg))
    cfg = PreTrainedConfig.from_pretrained(scratch)
    cfg.device = "cuda"
    policy = ICWMPolicy.from_pretrained(args.ckpt, config=cfg).eval()
    pre, _ = make_pre_post_processors(cfg, pretrained_path=args.ckpt)
    act_lo = policy._action_token_str_offset  # noqa: SLF001
    act_hi = act_lo + cfg.action_token_vocab_size
    im_end = policy._im_end_id  # noqa: SLF001

    roots = list(cfg.context_pool_dataset_roots)
    q = [LeRobotDataset(r, root=r, delta_timestamps=resolve_delta_timestamps(cfg, LeRobotDatasetMetadata(r, root=r)),
                        return_uint8=True) for r in roots]
    ds = ICWMViewpointMatchedDataset(q, [LeRobotDataset(r, root=r) for r in roots], horizon=cfg.action_chunk_size,
                                     n_context_clips=cfg.n_context_clips, stride=cfg.context_pool_stride, seed=1)
    cams = [k for k, f in cfg.input_features.items() if f.type.value == "VISUAL"]
    # spread the samples over the 4 viewpoint datasets
    rng = np.random.default_rng(1)
    picks = []
    for k in range(len(roots)):
        lo, hi = int(ds._cum[k]), int(ds._cum[k + 1])  # noqa: SLF001
        picks += rng.integers(lo, hi, size=max(1, args.n // len(roots))).tolist()

    mass = defaultdict(lambda: defaultdict(list))   # segment -> layer -> [mass per generated token]
    size = defaultdict(list)                        # segment -> number of tokens (per sample)
    n_layers = None
    for i in picks:
        it = ds[i]
        raw = {k: it[k].float() / 255.0 for k in cams}
        raw["observation.state"] = it["observation.state"]
        for k in CTX:
            raw[k] = it[k][None]
        raw["task"] = [it["task"]]
        b = pre(raw)
        with torch.no_grad():
            gen = policy.model.generate(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                                        pixel_values=b["pixel_values"], image_grid_thw=b["image_grid_thw"],
                                        max_new_tokens=cfg.max_action_tokens_per_chunk, do_sample=False,
                                        use_cache=True, eos_token_id=im_end)
        prompt_len = b["input_ids"].shape[1]
        full = gen[0].tolist()
        # queries: the generated action tokens (the position that *predicts* token t is t-1, so use the
        # rows of positions prompt_len-1 .. last-1: each of them produced one generated token)
        gen_tokens = full[prompt_len:]
        n_act = sum(act_lo <= t < act_hi for t in gen_tokens)
        if n_act == 0:
            continue
        policy.model.set_attn_implementation({"text_config": "eager"})
        with torch.no_grad():
            out = policy.model(input_ids=gen[:, :-1] if gen_tokens[-1] == im_end else gen,
                               attention_mask=torch.ones_like(gen[:, :-1] if gen_tokens[-1] == im_end else gen),
                               pixel_values=b["pixel_values"], image_grid_thw=b["image_grid_thw"],
                               output_attentions=True, use_cache=False)
        policy.model.set_attn_implementation({"text_config": "sdpa"})
        ids = (gen[0, :-1] if gen_tokens[-1] == im_end else gen[0]).tolist()
        seg = segment_ids(ids, act_lo, act_hi, cfg.n_context_clips, len(cams), prompt_len)
        names = sorted(set(seg))
        idx = {nm: torch.tensor([p for p, s in enumerate(seg) if s == nm], device="cuda") for nm in names}
        for nm in names:
            size[nm].append(len(idx[nm]))
        rows = list(range(prompt_len - 1, len(ids)))  # positions that emitted a generated token
        rows = [r for r in rows if r + 1 < len(full) and act_lo <= full[r + 1] < act_hi]  # only action-token predictions
        n_layers = len(out.attentions)
        for L, att in enumerate(out.attentions):
            a = att[0].float().mean(0)[rows]  # [n_rows, S], averaged over heads
            for nm in names:
                m = a[:, idx[nm]].sum(-1)
                mass[nm][L] += m.tolist()
        del out
        torch.cuda.empty_cache()

    order = ["system/template", "context o^s (start img)", "context action tokens", "context o^e (end img)",
             "current obs: agentview", "current obs: wrist", "instruction", "generated action tokens"]
    order = [o for o in order if o in mass] + [o for o in mass if o not in order]
    bands = [(0, n_layers // 3), (n_layers // 3, 2 * n_layers // 3), (2 * n_layers // 3, n_layers)]
    lines = [f"checkpoint: {args.ckpt}",
             f"samples used: {len(size['instruction'])}, layers: {n_layers}",
             "",
             "attention mass of generated action tokens (mean over heads, tokens, samples)",
             f"{'segment':28s} {'tokens':>7s} {'all layers':>10s} {'layers 0-' + str(bands[0][1] - 1):>11s} "
             f"{'mid':>7s} {'last third':>10s} {'per-token x1e3':>15s}"]
    for nm in order:
        allv = np.mean([np.mean(mass[nm][L]) for L in range(n_layers)])
        bv = [np.mean([np.mean(mass[nm][L]) for L in range(*bd)]) for bd in bands]
        ntok = np.mean(size[nm])
        lines.append(f"{nm:28s} {ntok:7.0f} {allv:10.3f} {bv[0]:11.3f} {bv[1]:7.3f} {bv[2]:10.3f} {1e3 * allv / max(ntok, 1):15.2f}")
    lines += ["", "per layer (mass):", "layer " + " ".join(f"{o[:12]:>12s}" for o in order)]
    for L in range(n_layers):
        lines.append(f"{L:5d} " + " ".join(f"{np.mean(mass[o][L]):12.3f}" for o in order))
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
