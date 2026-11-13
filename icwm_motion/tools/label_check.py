import sys, torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
from lerobot.datasets.factory import make_dataset
from transformers import AutoTokenizer
ckpt=sys.argv[1]; cfg=PreTrainedConfig.from_pretrained(ckpt); cfg.device="cpu"
pre,_=make_pre_post_processors(cfg, pretrained_path=ckpt, preprocessor_overrides={"device_processor":{"device":"cpu"}})
root=cfg.context_pool_dataset_roots[0]
ds=make_dataset(TrainPipelineConfig(dataset=DatasetConfig(repo_id=root,root=root,video_backend="pyav"),policy=cfg))
item=ds[12345]; b={k:(v[None] if isinstance(v,torch.Tensor) else [v]) for k,v in item.items()}
for k in list(b):
    if k.startswith("observation.images") and b[k].dtype==torch.uint8: b[k]=b[k].float()/255
out=pre(b)
tok=AutoTokenizer.from_pretrained(cfg.backbone_name_or_path)
lab=out["labels"][0]; sup=lab!=-100
ids=lab[sup].tolist()
print("supervised:", repr(tok.decode([i for i in ids if i < len(tok)])), "| action tokens:", sum(i>=len(tok) for i in ids), "| n_sup", len(ids))
print("reasoning_end", out.get("reasoning_end"), "first sup pos", int(sup.nonzero()[0]), "seq", lab.shape[0])
print("first action pos", int(((out["input_ids"][0]>=len(tok)+0)&sup).nonzero()[0]) if any(i>=len(tok) for i in ids) else None)
