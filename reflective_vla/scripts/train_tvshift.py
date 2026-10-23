"""Reflective VLA training on the tvshift data (4 angles x 2 tasks), delta-EEF actions.

    torchrun --nproc_per_node=8 scripts/train_tvshift.py --out outputs/tvshift_K8

Init is lerobot/pi05_base (local copy). The whole policy is one FSDP2 group: the pi0.5 code
calls decoder-layer weights directly rather than through ``layer.forward`` so per-layer
sharding hooks would never fire; one group unshards everything at the wrapper's forward.
fp32 sharded master weights, bf16 compute, fp32 gradient reduction.

Resumable: every --save-every steps a sharded DCP checkpoint (model+optimizer+step) is
written to <out>/ckpt/last and a bf16 safetensors for inference to <out>/export/step_N.
"""
import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, get_model_state_dict, StateDictOptions
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.profiler import record_function
from torch.utils.data import DataLoader, DistributedSampler

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

from reflective_vla import ReflectiveVLAConfig, ReflectiveVLAPolicy
from reflective_vla.dtype_utils import add_input_casts
from reflective_vla.data.triplet_dataset import CAM_KEYS, TripletDataset


class Wrapper(torch.nn.Module):
    """Gives FSDP2 a root whose ``__call__`` runs the full training forward."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, batch):
        return self.policy(batch)


def freeze_unused(policy):
    """Parameters that never receive a gradient (LM heads, the last VLM layer's post-attention
    path and the final norm: only that layer's K/V are read by the action expert). Left in the
    optimizer they never get state, which breaks resuming from a distributed checkpoint."""
    last = len(policy.model.paligemma_with_expert.paligemma.model.language_model.layers) - 1
    n_frozen = 0
    for n, p in policy.named_parameters():
        in_last = f"paligemma.model.language_model.layers.{last}." in n and any(
            x in n for x in ("self_attn.o_proj", "mlp.", "post_attention_layernorm"))
        if (n.endswith("paligemma.lm_head.weight") or "gemma_expert.lm_head" in n
                or (n.endswith("language_model.norm.weight") and "paligemma" in n) or in_last):
            p.requires_grad = False
            n_frozen += 1
    return n_frozen


def load_paligemma_init(policy, path):
    """pi0.5 "from PaliGemma" init: the VLM gets google/paligemma-3b-pt-224, everything else
    (action expert, action/time projections) keeps its random init. PaliGemma's vocabulary has 64
    extra rows (257216 vs 257152): only the first 257152 are kept, as openpi does; lm_head is tied
    to the embeddings."""
    import glob
    from safetensors.torch import load_file

    sub = policy.model.paligemma_with_expert.paligemma
    own = sub.state_dict()
    sd = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        for k, v in load_file(f).items():
            k = (k.replace("language_model.model.", "model.language_model.")
                  .replace("vision_tower.", "model.vision_tower.")
                  .replace("multi_modal_projector.", "model.multi_modal_projector.")
                  .replace("model.model.", "model."))
            sd[k] = v
    emb = "model.language_model.embed_tokens.weight"
    sd[emb] = sd[emb][: own[emb].shape[0]]
    if "lm_head.weight" in own:
        sd["lm_head.weight"] = sd[emb]
    missing = [k for k in own if k not in sd]
    unexpected = [k for k in sd if k not in own]
    bad = [k for k in own if k in sd and sd[k].shape != own[k].shape]
    assert not (missing or unexpected or bad), (missing[:5], unexpected[:5], bad[:5])
    sub.load_state_dict({k: v.to(own[k].dtype) for k, v in sd.items()}, strict=True)
    return len(sd)


def make_config(args):
    img = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
    cfg = ReflectiveVLAConfig(
        input_features={**{k: img for k in CAM_KEYS},
                        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        context_frames=args.K, chunk_size=10, n_action_steps=10, context_gap_max=15,
        optimizer_lr=args.lr, scheduler_warmup_steps=args.warmup, scheduler_decay_steps=args.steps,
        image_token_pool=args.image_token_pool, gradient_checkpointing=True, device="cuda", push_to_hub=False, dtype="bfloat16",
    )
    return cfg


def lr_at(step, args):
    if step < args.warmup:
        return (step + 1) / args.warmup
    p = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.final_lr_frac + (1 - args.final_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def to_device(batch, dev):
    out = {}
    for k, v in batch.items():
        if not torch.is_tensor(v):
            continue
        v = v.to(dev, non_blocking=True)
        if v.dtype == torch.uint8:
            v = v.float() / 255.0
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="/workspace/yujin/data/reflective_tvshift_cache")
    ap.add_argument("--init", default="/workspace/yujin/checkpoints/pi05_base")
    ap.add_argument("--init-paligemma", default="", help="init the VLM from this PaliGemma checkpoint dir "
                    "(e.g. google/paligemma-3b-pt-224) instead of --init; the action expert starts random")
    ap.add_argument("--tokenizer", default="/workspace/yujin/checkpoints/paligemma_tokenizer", help="local copy of google/paligemma-3b-pt-224 (gated)")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=2, help="per GPU micro-batch")
    ap.add_argument("--accum", type=int, default=2, help="micro-batches per optimizer step (effective = bs*accum*GPUs)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--final-lr-frac", type=float, default=0.1)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb-project", default="reflective-tvshift")
    ap.add_argument("--smoke", action="store_true", help="few steps, no saving")
    ap.add_argument("--image-token-pool", type=int, default=1, help="average-pool image tokens by this factor (1 = paper)")
    ap.add_argument("--debug-nograd", action="store_true")
    ap.add_argument("--anomaly", action="store_true", help="autograd anomaly detection (slow)")
    args = ap.parse_args()

    if args.anomaly:
        torch.autograd.set_detect_anomaly(True)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    torch.manual_seed(args.seed + rank)
    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)

    cfg = make_config(args)
    if args.init_paligemma:
        policy = ReflectiveVLAPolicy(cfg)
        policy.float()
        n = load_paligemma_init(policy, args.init_paligemma)
        if rank == 0:
            print(f"init: PaliGemma VLM from {args.init_paligemma} ({n} tensors), action expert random", flush=True)
    else:
        policy = ReflectiveVLAPolicy.from_pretrained(args.init, config=cfg, strict=False)
    policy.train()
    policy.float()
    freeze_unused(policy)
    add_input_casts(policy)
    wrapper = Wrapper(policy)
    fully_shard(wrapper, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32))

    groups = policy.get_optim_params()
    groups = [{**g, "params": [p for p in g["params"] if p.requires_grad]} for g in groups]
    for g in groups:
        g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(groups, betas=cfg.optimizer_betas, weight_decay=cfg.optimizer_weight_decay, eps=1e-8)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ds = TripletDataset(args.cache, K=args.K, C=cfg.chunk_size, gap_max=cfg.context_gap_max,
                        tokenizer=tok, max_len=cfg.tokenizer_max_length, train=True)
    if rank == 0:
        (out / "stats.json").write_text(json.dumps({k: {a: b.tolist() for a, b in v.items()}
                                                    for k, v in ds.stats.items()}))
        cfg.save_pretrained(out / "config")
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
    dl = DataLoader(ds, batch_size=args.bs, sampler=sampler, num_workers=args.workers, drop_last=True,
                    pin_memory=True, persistent_workers=args.workers > 0, prefetch_factor=2 if args.workers else None)

    step, epoch = 0, 0
    last = out / "ckpt" / "last"
    if last.exists() and not args.smoke:
        sd, osd = get_state_dict(wrapper, opt)
        state = {"model": sd, "optim": osd, "extra": {"step": torch.zeros(1)}}
        dcp.load(state, checkpoint_id=str(last))
        set_state_dict(wrapper, opt, model_state_dict=state["model"], optim_state_dict=state["optim"])
        step = int(state["extra"]["step"].item())
        epoch = step * args.bs * args.accum * world // len(ds)
        if rank == 0:
            print(f"resumed from step {step}", flush=True)

    run = None
    if rank == 0 and not args.smoke:
        try:
            import wandb
            run = wandb.init(project=args.wandb_project, name=out.name, config=vars(args), resume="allow",
                             id=(out / "wandb_id").read_text() if (out / "wandb_id").exists() else None)
            (out / "wandb_id").write_text(run.id)
        except Exception as e:  # logging must never block training
            print("wandb disabled:", e, flush=True)

    total = 6 if args.smoke else args.steps

    def batches(epoch):
        while True:  # endless stream; reshuffles every epoch
            sampler.set_epoch(epoch)
            for b in dl:
                yield b
            epoch += 1

    stream = batches(epoch)
    t0, tl = time.time(), 0.0
    t_start = step
    tim = dict.fromkeys(["data_s", "forward_s", "backward_s", "optimizer_s", "step_s"], 0.0)
    t_step = time.perf_counter()

    prof, prof_n = None, 0
    if os.environ.get("REFLECTIVE_PROFILE") and rank == 0:
        from torch.profiler import ProfilerActivity, profile, schedule
        prof_dir = out / "profile"
        prof_dir.mkdir(parents=True, exist_ok=True)

        def _write(p):
            (prof_dir / "cuda_top.txt").write_text(p.key_averages().table(sort_by="cuda_time_total", row_limit=40))
            (prof_dir / "cpu_top.txt").write_text(p.key_averages().table(sort_by="cpu_time_total", row_limit=40))
            print(f"profiler tables written to {prof_dir}", flush=True)

        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                       schedule=schedule(wait=2, warmup=1, active=3, repeat=1), on_trace_ready=_write)
        prof.start()
    while step < total:
        f = lr_at(step, args)
        for i, g in enumerate(opt.param_groups):
            frozen = i == 1 and step < cfg.resolved_vlm_freeze_steps  # VLM held frozen at the start
            g["lr"] = 0.0 if frozen else g["base_lr"] * f
        sync = torch.cuda.synchronize
        tm = time.perf_counter
        for m in range(args.accum):
            # Keep the gathered (unsharded) parameters between micro-batches of one optimizer step and
            # reshard only after the last one: otherwise FSDP re-runs the full-model all-gather before
            # every micro-batch (~1.1 s each on hilo2). Gradients are still reduce-scattered per
            # micro-batch, so the update is unchanged; cost is one bf16 copy of the model (~7 GB).
            wrapper.set_reshard_after_backward(m == args.accum - 1)
            sync(); t_a = tm()
            with record_function("data"):
                batch = to_device(next(stream), dev)
            sync(); t_b = tm()
            with record_function("forward"):
                loss, info = wrapper(batch)
            sync(); t_c = tm()
            with record_function("backward"):
                (loss / args.accum).backward()
            sync(); t_d = tm()
            tl += loss.item() / args.accum
            tim["data_s"] += t_b - t_a
            tim["forward_s"] += t_c - t_b
            tim["backward_s"] += t_d - t_c
        if args.debug_nograd and step == 0 and rank == 0:
            print("NOGRAD:", [n for n, p in wrapper.named_parameters() if p.requires_grad and p.grad is None], flush=True)
        t_e = tm()
        with record_function("optimizer"):
            gn = torch.nn.utils.clip_grad_norm_(wrapper.parameters(), cfg.optimizer_grad_clip_norm)
            opt.step()
            opt.zero_grad(set_to_none=True)
        sync()
        tim["optimizer_s"] += tm() - t_e
        tim["step_s"] += tm() - t_step
        t_step = tm()
        step += 1
        if prof is not None:
            prof.step()
            prof_n += 1
            if prof_n >= 7:  # wait 2 + warmup 1 + active 3 (+1): tables are written after step 6
                prof.stop()
                prof = None
        if step % args.log_every == 0 or args.smoke:
            n_win = 1 if args.smoke else args.log_every
            avg, tl = tl / n_win, 0.0
            if rank == 0:
                dt = (time.time() - t0) / (step - t_start)
                mem = torch.cuda.max_memory_allocated() / 2**30
                times = {k: v / n_win for k, v in tim.items()}
                print(f"step {step}/{total} loss {avg:.4f} gn {float(gn):.2f} lr {opt.param_groups[0]['lr']:.2e} "
                      f"mem {mem:.1f}G {dt:.1f}s/step | " + " ".join(f"{k} {v:.2f}" for k, v in times.items()), flush=True)
                if run:
                    run.log({"loss": avg, "grad_norm": float(gn), "lr": opt.param_groups[0]["lr"],
                             **{f"time/{k}": v for k, v in times.items()},
                             "system/gpu_mem_max_gb": mem,
                             "system/gpu_mem_reserved_gb": torch.cuda.memory_reserved() / 2**30}, step=step)
            tim = dict.fromkeys(tim, 0.0)
        if step % args.save_every == 0 and not args.smoke:
            save(wrapper, opt, step, out, rank)
    if not args.smoke and step % args.save_every != 0:
        save(wrapper, opt, step, out, rank)
    dist.barrier()
    dist.destroy_process_group()


def save(wrapper, opt, step, out, rank):
    sd, osd = get_state_dict(wrapper, opt)
    tmp = out / "ckpt" / "tmp"
    dcp.save({"model": sd, "optim": osd, "extra": {"step": torch.tensor([step])}}, checkpoint_id=str(tmp))
    dist.barrier()
    if rank == 0:
        last = out / "ckpt" / "last"
        if last.exists():
            shutil.rmtree(last)
        tmp.rename(last)
    full = get_model_state_dict(wrapper, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if rank == 0:
        from safetensors.torch import save_file
        d = out / "export" / f"step_{step}"
        d.mkdir(parents=True, exist_ok=True)
        save_file({k.removeprefix("policy."): v.to(torch.bfloat16).contiguous() for k, v in full.items()},
                  str(d / "model.safetensors"))
        # keep only the newest two exports; each is ~7GB
        exports = sorted((out / "export").glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
        for old in exports[:-2]:
            shutil.rmtree(old)
        print(f"saved step {step}", flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
