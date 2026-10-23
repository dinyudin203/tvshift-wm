"""Self-exploration context pool for ICWM training (context_source="probe").

For every seen (task, camera angle) scene of the tvshift protocol, run the exact probing routine the
evaluation uses (`run_icwm_eval_tvshift.probe`: 20 task-agnostic moves of 5 steps from the episode's
start pose, gripper open) from each of LIBERO's 50 init states, `--repeats` times with different seeds.
Training then draws the query's context clips from the probes of its own scene, so train and test
context come from the same distribution.

Output per scene, in --out:
    {dataset}__task{task_index}__start.npy  uint8 [N, 512, 512, 3]  o^s (vertically flipped, as the datasets)
    {dataset}__task{task_index}__end.npy    uint8 [N, 512, 512, 3]  o^e
    {dataset}__task{task_index}__actions.npy float32 [N, 5, 7]
    {dataset}__task{task_index}__meta.json  (prompt, scene, N, init states, repeats); written last = done marker

    python gen_icwm_probe_pool.py --out /workspace/yujin/data/icwm_probe_pool --tasks 0,1 --gpu 0
"""
import argparse
import json
from pathlib import Path

import numpy as np

import run_icwm_eval_tvshift as ev

DATASET = {0: "libero_spatial_theta000", 45: "libero_goal_theta045", 345: "libero_goal_theta345",
           270: "libero_object_theta270"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="0,1,2,3,4,5,6,7", help="tvshift task ids (= ev.TASKS rows)")
    ap.add_argument("--inits", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--data-root", default="/workspace/yujin/data/libero_tvshift/lerobot_v30")
    ap.add_argument("--env-python", default="/opt/conda/envs/lingbot-va/bin/python")
    ap.add_argument("--libero-root", default="/workspace/lerobot-libero")
    ap.add_argument("--adapt-repo", default="/workspace/yujin/lingbot-va-adapt")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    envc = ev.EnvClient(args, args.gpu)
    for tid in map(int, args.tasks.split(",")):
        _, suite, idx, theta = ev.TASKS[tid]
        ds = f"{DATASET[theta]}_no_noops_lerobot"
        tasks = [json.loads(line)["task"] for line in open(f"{args.data_root}/{ds}/meta/tasks.jsonl")]
        n = args.inits * args.repeats * ev.N_PROBES
        starts = ends = acts = None
        k, prompt = 0, None
        for init in range(args.inits):
            for rep in range(args.repeats):
                r = envc.call(cmd="reset", suite=suite, idx=idx, theta=theta, episode=init, full=True)
                prompt = r["prompt"]
                clips = ev.probe(envc, r["obs"], np.random.default_rng((tid, init, rep)))
                if starts is None:
                    tag = f"{ds}__task{tasks.index(prompt)}"
                    if (out / f"{tag}__meta.json").exists():
                        break
                    shape = clips[0][0].shape
                    starts = np.lib.format.open_memmap(out / f"{tag}__start.npy", "w+", np.uint8, (n, *shape))
                    ends = np.lib.format.open_memmap(out / f"{tag}__end.npy", "w+", np.uint8, (n, *shape))
                    acts = np.zeros((n, ev.PROBE_STEPS, 7), np.float32)
                for o_s, a, o_e in clips:
                    starts[k], ends[k], acts[k] = o_s, o_e, a
                    k += 1
            else:
                print(f"t{tid} init {init}: {k}/{n} clips", flush=True)
                continue
            break  # scene already done
        if starts is None or k < n:
            print(f"t{tid}: already done, skipped", flush=True)
            continue
        starts.flush(); ends.flush()
        np.save(out / f"{tag}__actions.npy", acts)
        (out / f"{tag}__meta.json").write_text(json.dumps(dict(
            prompt=prompt, tvshift_task=tid, suite=suite, libero_idx=idx, theta=theta, dataset=ds,
            task_index=tasks.index(prompt), n_clips=n, inits=args.inits, repeats=args.repeats,
            n_probes=ev.N_PROBES, probe_steps=ev.PROBE_STEPS)))
        print(f"t{tid}: wrote {tag} ({n} clips)", flush=True)
    envc.close()


if __name__ == "__main__":
    main()
