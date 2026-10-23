#!/usr/bin/env bash
# Runs ON hilo2, after run_mj332_queue_hilo2.sh has recorded the first five models' videos.
# Context ablation of the three ICWM variants on the tvshift protocol (seen 8 + novel 24 combos,
# 10 episodes, MuJoCo 3.3.2):
#   blank : 5 context clips that carry no information (black frames, arm still, gripper open)
#   empty : no context clip at all in the prompt (n_valid = 0)
# (the history model sees n_valid = 0 at the start of every episode; v2 never saw it in training)
set -uo pipefail
exec 7>/tmp/ctx_ablation.lock
flock -n 7 || { echo "ablation already running"; exit 0; }
log() { echo "[$(date -u '+%F %T')] ctx-ablation: $*"; }
I=/workspace/yujin/icwm; IPY=$I/conda_env312/bin/python
until grep -q "wam plans" /workspace/yujin/mj332_queue.log 2>/dev/null; do sleep 600; done
for mode in blank empty; do
    flag=--no-context; [ $mode = empty ] && flag=--empty-context
    for m in v2:icwm_libero_tvshift_v2 history:icwm_libero_tvshift_hist sametask:icwm_libero_tvshift_sametask; do
        name=${m%%:*}; run=${m#*:}
        out=$I/evaluate_results/ctx${mode}_${name}_step20000_mj332
        mkdir -p $out
        for i in 1 2 3; do
            [ -f $out/summary.json ] && break
            log "$(basename $out): attempt $i"
            (cd $I/lerobot && env HF_HOME=/root/.cache/huggingface TOKENIZERS_PARALLELISM=false PYTHONPATH=src \
                $IPY run_icwm_eval_tvshift.py --ckpt $I/outputs/train/$run/checkpoints/020000/pretrained_model \
                --out $out --gpus 0,1,2,3 --episodes 10 $flag >> $out/eval.log 2>&1)
        done
        log "$(basename $out): $(tail -3 $out/eval.log | tr '\n' ' ')"
    done
done
log "ALL DONE"
