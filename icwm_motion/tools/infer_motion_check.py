"""Load a motion-reasoning checkpoint, run predict_action_chunk on a few dataset frames (with their
context clips), and report the reasoning text, parsed-vs-target motion error and the action chunk.
    python infer_motion_check.py <pretrained_model dir> [n_samples]
"""
import sys

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.icwm.modeling_icwm import ICWMPolicy
from lerobot.policies.icwm.motion_reasoning import parse_motion
from lerobot.policies.factory import make_pre_post_processors

ckpt = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
cfg = PreTrainedConfig.from_pretrained(ckpt)
cfg.device = "cuda"
policy = ICWMPolicy.from_pretrained(ckpt, config=cfg).eval()
pre, post = make_pre_post_processors(cfg, pretrained_path=ckpt)

# the training dataset (viewpoint-matched queries + their context clips + calibration sidecars)
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
from lerobot.datasets.factory import make_dataset

root = cfg.context_pool_dataset_roots[0]
tcfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id=root, root=root, video_backend="pyav"), policy=cfg)
ds = make_dataset(tcfg)
gen = torch.Generator().manual_seed(0)
idx = torch.randint(len(ds), (n,), generator=gen).tolist()
for i in idx:
    item = ds[i]
    batch = {k: (v[None] if isinstance(v, torch.Tensor) else [v]) for k, v in item.items()}
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor) and v.dtype == torch.uint8 and k.startswith("observation.images"):
            batch[k] = v.float() / 255
    gt_action = batch.pop("action")
    # targets for comparison: the training-time processor computes them from the calibration
    tb = pre({**batch, "action": gt_action})
    target = tb["motion_targets"][0]
    ib = pre(batch)
    with torch.no_grad():
        chunk = policy.predict_action_chunk(ib)
    text = policy._last_motion_text[0]
    parsed = parse_motion(text)
    err = None if parsed is None else (parsed - target).abs().max().item()
    act = post(chunk)
    print(f"[{i}] reasoning={text!r}\n     target={[[round(v, 2) for v in r] for r in target.tolist()]}"
          f"\n     max|err|={err}  action[0]={[round(v, 3) for v in act[0, 0].tolist()]}"
          f"  gt[0]={[round(v, 3) for v in gt_action[0, 0].tolist()]}", flush=True)
print("INFER_OK")
