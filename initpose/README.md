# Robot initial-pose perturbation data (LIBERO)

Generates LIBERO demos whose robot starts from a perturbed joint pose, with the *action labels* changed accordingly
(LIBERO-Plus style: not only the images differ).

- `initpose.py` - condition `HOME_Q + mag * u` (`u`: unit direction in joint space from a seed; `mag` in rad, signed;
  0 = default, GT replayed as is), GT reference, merge-step choice, warped approach + feed-forward / feedback action
  correction, contact and success checks.
- `make_shard.py` - roll out the GT demos of one task from the perturbed start, drop no-ops, write episode shards
  (retry until a clean, successful rollout; `--max-track-mm` rejects large post-merge tracking error).
- `run_stage_a*.sh` - generate the shards for the 6 training (suite, task, magnitude) conditions
  (Spatial 1,3 default; Goal 0,7 +0.20 rad; Object 0,1 -0.30 rad), plus the retry scripts.
- `run_stage_b.sh` - pack each condition into its own LeRobot v3.0 dataset (`{name}_no_noops_lerobot`).

Method: the grip-site offset (dp, w0) to the GT trajectory at the start decays to zero with a smoothstep by the merge
step (10 steps before the first grasp / new contact); action = GT action + feed-forward (-ds*offset) + kp * feedback,
clipped to [-1, 1]; the actions actually sent are the labels. Rollouts that fail or touch something new on the way in are rejected.

External dependencies (not in this repo): `replay.py` / `embodiments.py` (libero-embodiment), `render_shard.py` and
`pack_lerobot.py` (FastWAM `libero_rot`), LIBERO + robosuite 1.4 + MuJoCo; paths are hard-coded to the hilo server layout.
