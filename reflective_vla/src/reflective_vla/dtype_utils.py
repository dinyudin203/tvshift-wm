"""dtype helpers shared by training and evaluation."""
import torch


def add_input_casts(module):
    """Cast floating inputs of Linear/Conv layers to the weight dtype.

    The pi0.5/reflective code feeds fp32 activations (noisy actions, time embeddings, images)
    into layers whose weights are bf16 (FSDP bf16 compute in training, a bf16 model at
    inference); without this the first such layer raises a dtype mismatch.
    """
    def pre(m, args):
        w = m.weight.dtype
        return tuple(a.to(w) if torch.is_tensor(a) and a.is_floating_point() and a.dtype != w else a for a in args)
    for m in module.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv3d)):
            m.register_forward_pre_hook(pre)
