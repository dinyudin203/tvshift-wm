"""Reflective VLA -- in-context observation-action-consequence conditioning for dual-system VLAs.

Reproduction of *Reflective VLA: In-Context Action Consequences Make VLAs Generalize*
(arXiv:2606.25215) on top of the LeRobot pi0.5 backbone.
"""

from .buffer import Triplet, TripletBuffer
from .configuration_reflective_vla import PAPER_TRAINING_PRESETS, ReflectiveVLAConfig
from .masking import (
    build_attention_mask,
    build_position_ids,
    build_prefix_attention_mask,
    build_prefix_layout,
    suffix_query_frames,
)
from .modeling_reflective_vla import ReflectiveVLAPolicy, ReflectiveVLAPytorch
from .processor_reflective_vla import make_reflective_vla_pre_post_processors
from .tokenization import TripletProjectors

__all__ = [
    "PAPER_TRAINING_PRESETS",
    "ReflectiveVLAConfig",
    "ReflectiveVLAPolicy",
    "ReflectiveVLAPytorch",
    "Triplet",
    "TripletBuffer",
    "TripletProjectors",
    "build_attention_mask",
    "build_position_ids",
    "build_prefix_attention_mask",
    "build_prefix_layout",
    "make_reflective_vla_pre_post_processors",
    "suffix_query_frames",
]
