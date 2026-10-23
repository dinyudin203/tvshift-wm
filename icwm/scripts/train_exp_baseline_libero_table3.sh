#!/usr/bin/env bash
# EXP baseline: MV + ground-truth camera azimuth given as text (e.g. appended to the
# instruction as "... Camera azimuth: 90 degrees."), still without in-context conditioning
# (n_context_clips=0). This is the "oracle psi as text" ablation the paper compares ICWM
# against.
#
# NOT EXECUTED (see icwm/README.md).
#
# Implementation status: this baseline is NOT implemented as a separate policy/processor in
# this deliverable -- it only needs a one-line change to
# lerobot/policies/icwm/processor_icwm.py::ICWMSequenceBuilderStep.complementary_data (append
# the ground-truth azimuth string, e.g. from a dataset/env metadata field, to `cleaned_task`
# before it is added to the message content), gated behind a new `--policy.append_azimuth_text`
# style flag on ICWMConfig. Left as a follow-up: the current LIBERO-Spatial LeRobot datasets
# under /workspace/yujin/data/libero_spatial_rot/lerobot_v30/ encode the azimuth only in the
# directory name (libero_spatial_theta030_...), not as a per-sample column, so wiring this up
# properly means either (a) reading the azimuth from the dataset repo_id/root path at
# preprocessing time, or (b) adding an explicit azimuth column when rendering the cross-view
# data (scripts/render_libero_cross_view.sh).

set -euo pipefail
echo "EXP baseline is not yet implemented as a runnable script -- see the TODO/comment block"
echo "at the top of this file for what remains, and icwm/README.md for full context."
exit 1
