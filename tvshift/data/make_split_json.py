"""Build a trivial train/seen_eval/unseen split_json for one packed LeRobot dataset.
All episodes go to train, except a small tail held out as seen_eval (diagnostic only).
unseen_eval_episodes stays empty: this baseline has no held-out (task,theta) split at
the training-data level (the "unseen pair" test happens later at eval time by combining
task/theta values across the 4 packed sets).
"""
import json
import sys

from lerobot.datasets.lerobot_dataset import LeRobotDataset

root, repo_id, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
ds = LeRobotDataset(repo_id=repo_id, root=root)
eps = sorted(int(row["episode_index"]) for row in ds.meta.episodes)
n_holdout = max(1, len(eps) // 10)
holdout = eps[-n_holdout:]
train = eps[:-n_holdout]
json.dump({
    "train_episodes": train,
    "seen_eval_episodes": holdout,
    "unseen_eval_episodes": [],
}, open(out_path, "w"))
print(f"{repo_id}: {len(eps)} episodes -> train={len(train)} seen_eval={len(holdout)} -> {out_path}")
