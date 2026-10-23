"""Does the trained ICWM checkpoint produce decodable action chunks on its *own training data*?

Separates "the model can't generate valid FAST sequences yet" from "the eval inputs differ from
training". For random training samples (query + same-viewpoint context clips, built exactly as in
training) it
  1. runs the training forward (with labels) -> teacher-forced token accuracy (sanity check vs wandb),
  2. runs the eval-style generate() on the same inputs without the action -> free-running tokens,
and compares the generated action tokens with the ground-truth target tokens.

    PYTHONPATH=src python diag_icwm_decode.py --ckpt <.../checkpoints/020000/pretrained_model> --n 40
"""
import argparse
import json
import sys
import tempfile
from collections import Counter
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw_cfg = json.loads((Path(args.ckpt) / "config.json").read_text())
    raw_cfg.setdefault("type", "icwm")
    scratch = tempfile.mkdtemp()
    (Path(scratch) / "config.json").write_text(json.dumps(raw_cfg))
    cfg = PreTrainedConfig.from_pretrained(scratch)
    cfg.device = "cuda"
    policy = ICWMPolicy.from_pretrained(args.ckpt, config=cfg).eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=args.ckpt)
    model_offset, vocab = policy._action_token_str_offset, cfg.action_token_vocab_size  # noqa: SLF001
    seq_step = next(st for st in pre.steps if type(st).__name__ == "ICWMSequenceBuilderStep")
    offset = int(seq_step.action_token_ids[0])  # the id the training targets actually use for <|action_token_0|>
    print(f"action-token offset: processor/training {offset} vs model decode path {model_offset}", flush=True)

    roots = list(cfg.context_pool_dataset_roots)
    q = []
    for r in roots:
        meta = LeRobotDatasetMetadata(r, root=r)
        q.append(LeRobotDataset(r, root=r, delta_timestamps=resolve_delta_timestamps(cfg, meta), return_uint8=True))
    pools = [LeRobotDataset(r, root=r) for r in roots]
    ds = ICWMViewpointMatchedDataset(q, pools, horizon=cfg.action_chunk_size, n_context_clips=cfg.n_context_clips,
                                     stride=cfg.context_pool_stride, seed=args.seed)
    cams = [k for k, f in cfg.input_features.items() if f.type.value == "VISUAL"]
    rng = np.random.default_rng(args.seed)

    stats = Counter()
    tf_hit = tf_tot = 0
    lens_gen, lens_gt, examples = [], [], []
    for i in rng.integers(0, len(ds), size=args.n).tolist():
        item = ds[i]
        raw = {k: item[k].float() / 255.0 for k in cams}
        raw["observation.state"] = item["observation.state"]
        for k in CTX:
            raw[k] = item[k][None]
        raw["task"] = [item["task"]]

        # (1) training-style forward with the target -> labels + teacher-forced accuracy
        with_action = dict(raw)
        with_action["action"] = item["action"][None]
        b = pre(with_action)
        labels = b["labels"]
        with torch.no_grad():
            out = policy.model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], pixel_values=b["pixel_values"],
                               image_grid_thw=b["image_grid_thw"], labels=labels, use_cache=False)
        lab = labels[:, 1:]
        pred = out.logits[:, :-1].argmax(-1)
        sup = lab != -100
        tf_hit += int(((pred == lab) & sup).sum())
        tf_tot += int(sup.sum())
        # target action tokens only (the labels now also supervise the closing <|im_end|>)
        gt = [int(t) - offset for t in labels[0][labels[0] != -100].tolist() if offset <= int(t) < offset + vocab]

        # (2) eval-style free-running generation on the same inputs (no action)
        b2 = pre(dict(raw))
        with torch.no_grad():
            gen = policy.model.generate(input_ids=b2["input_ids"], attention_mask=b2["attention_mask"],
                                        pixel_values=b2["pixel_values"], image_grid_thw=b2["image_grid_thw"],
                                        max_new_tokens=cfg.max_action_tokens_per_chunk, do_sample=False, use_cache=True)
        new = gen[0, b2["input_ids"].shape[1]:].tolist()
        act = [t - offset for t in new if offset <= t < offset + vocab]
        non_action = [t for t in new if not (offset <= t < offset + vocab)]

        fast = policy._fast_tokenizer  # noqa: SLF001
        need = cfg.action_chunk_size * cfg.output_features["action"].shape[0]

        def n_coeffs(ids):
            try:
                return len(fast.bpe_tokenizer.decode(ids))
            except Exception:
                return -1

        n_coeff = n_coeffs(act)
        # what the eval path sees today: wrong offset (model_offset)
        act_bug = [t - model_offset for t in new if t >= model_offset]
        stats["decodable_with_old_offset"] += n_coeffs(act_bug) == need
        # stop rule: shortest prefix of the generated action tokens that decodes to exactly `need` coefficients
        prefix = next((act[:k] for k in range(1, len(act) + 1) if n_coeffs(act[:k]) == need), None)
        stats["decodable_with_prefix_stop"] += prefix is not None
        stats["prefix_exact_match"] += prefix == gt
        first_non_action = next((j for j, t in enumerate(new) if not (offset <= t < offset + vocab)), len(new))
        stats["starts_with_action_tokens"] += first_non_action > 0
        stats["emits_im_end"] += 151645 in new
        stats["n"] += 1
        stats["decodable"] += n_coeff == need
        stats["exact_match"] += act == gt
        stats["same_length"] += len(act) == len(gt)
        stats["first_token_right"] += bool(act) and bool(gt) and act[0] == gt[0]
        stats["hit_max_new_tokens"] += len(new) >= cfg.max_action_tokens_per_chunk
        lens_gen.append(len(act))
        lens_gt.append(len(gt))
        if len(examples) < 5:
            examples.append({"gt": gt, "gen": act, "non_action_tokens": non_action[:8], "coeffs": n_coeff, "need": need})

    n = stats["n"]
    print(f"\nsamples: {n}   (checkpoint {args.ckpt})")
    print(f"teacher-forced token accuracy here: {tf_hit / max(tf_tot, 1):.3f}  (wandb token_acc ~0.43 at the end)")
    for k in ("decodable_with_old_offset", "decodable", "decodable_with_prefix_stop", "exact_match", "prefix_exact_match",
              "same_length", "first_token_right", "starts_with_action_tokens", "emits_im_end", "hit_max_new_tokens"):
        print(f"{k:20s}: {stats[k]}/{n}")
    print(f"action tokens per chunk: generated mean {np.mean(lens_gen):.1f} (min {min(lens_gen)}, max {max(lens_gen)}),"
          f" ground truth mean {np.mean(lens_gt):.1f} (min {min(lens_gt)}, max {max(lens_gt)})")
    for e in examples:
        print(json.dumps(e))


if __name__ == "__main__":
    main()
