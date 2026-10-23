"""Which tokens of the assistant turn does the training loss actually supervise?"""
import json, sys, tempfile
from pathlib import Path
import torch
sys.path.insert(0, "/workspace/yujin/icwm/lerobot/src")
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.icwm_context_sampler import ICWMViewpointMatchedDataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors

ckpt = sys.argv[1]
raw_cfg = json.loads((Path(ckpt) / "config.json").read_text()); raw_cfg.setdefault("type", "icwm")
d = tempfile.mkdtemp(); (Path(d) / "config.json").write_text(json.dumps(raw_cfg))
cfg = PreTrainedConfig.from_pretrained(d); cfg.device = "cpu"
pre, _ = make_pre_post_processors(cfg, pretrained_path=ckpt, preprocessor_overrides={"device_processor": {"device": "cpu"}})
step = next(s for s in pre.steps if type(s).__name__ == "ICWMSequenceBuilderStep")
tok = step.processor.tokenizer
roots = list(cfg.context_pool_dataset_roots)
q = [LeRobotDataset(r, root=r, delta_timestamps=resolve_delta_timestamps(cfg, LeRobotDatasetMetadata(r, root=r)), return_uint8=True) for r in roots]
ds = ICWMViewpointMatchedDataset(q, [LeRobotDataset(r, root=r) for r in roots], horizon=5, n_context_clips=5, stride=20)
cams = [k for k, f in cfg.input_features.items() if f.type.value == "VISUAL"]
for i in (0, 20000, 40000):
    it = ds[i]
    raw = {k: it[k].float() / 255 for k in cams}
    raw["observation.state"] = it["observation.state"]
    for k in ("observation.icwm_context.images_start", "observation.icwm_context.images_end", "observation.icwm_context.actions"):
        raw[k] = it[k][None]
    raw["task"] = [it["task"]]
    raw["action"] = it["action"][None]
    b = pre(raw)
    ids, lab = b["input_ids"][0].tolist(), b["labels"][0].tolist()
    # the assistant turn starts after the last "<|im_start|>assistant\n"
    start = max(j for j in range(len(ids) - 1) if tok.decode([ids[j]]) == "<|im_start|>" and tok.decode([ids[j + 1]]).startswith("assistant")) + 3
    turn = ids[start:]
    is_act = [int(step.action_token_ids[0]) <= t <= int(step.action_token_ids[-1]) for t in turn]
    supervised = [l != -100 for l in lab[start:]]
    print(f"sample {i}: assistant turn = {len(turn)} tokens -> {tok.convert_ids_to_tokens(turn)[:3]} ... {tok.convert_ids_to_tokens(turn)[-3:]}")
    print(f"   action tokens in the turn: {sum(is_act)}, supervised by the loss: {sum(supervised)}; "
          f"first supervised index in turn: {supervised.index(True) if any(supervised) else None}; "
          f"<|im_end|> supervised: {any(s and t == 151645 for s, t in zip(supervised, turn))}")
