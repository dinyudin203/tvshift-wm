#!/usr/bin/env bash
# ICWM with Reflective-VLA style context (context_source=history) on tvshift:
#   1. train on napali2 (8 GPU, 20000 steps, = ICWM v2 except the context source); fresh start /
#      resume from checkpoints/last after a crash, up to 20 attempts. Disk is tight (22GB per
#      checkpoint), so only the 2 newest step checkpoints are kept.
#   2. evaluate step 20000 on hilo2 on the tvshift protocol (seen 8 + novel 24 combos, 10 episodes)
#      with --history-context (context = last 5 executed chunks of the episode, no probing).
# Idempotent (flock).
set -uo pipefail
exec 9>/workspace/yujin/.icwm_hist.lock
flock -n 9 || { echo "icwm hist already running"; exit 0; }
L=/workspace/yujin/icwm; R=/workspace/yujin/icwm
RUN=icwm_libero_tvshift_hist
OUT=$L/outputs/train/$RUN
FINAL=$OUT/checkpoints/020000
EVAL=evaluate_results/hist_step20000
log() { echo "[$(date -u '+%F %T')] icwm-hist: $*"; }

prune() {  # keep the 2 newest numbered checkpoints
    while true; do
        ls -d $OUT/checkpoints/0* 2>/dev/null | sort | head -n -2 | xargs -r rm -rf
        sleep 300
    done
}
prune & PRUNE=$!
trap 'kill $PRUNE 2>/dev/null' EXIT
i=0
while [ ! -f "$FINAL/pretrained_model/model.safetensors" ]; do
    while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -gt 2000 ]; do sleep 30; done
    [ $i -ge 20 ] && { log "giving up after $i attempts"; exit 1; }
    i=$((i+1))
    if [ -f "$OUT/checkpoints/last/pretrained_model/train_config.json" ]; then
        log "resume attempt $i"; bash $L/scripts/resume_icwm_libero_tvshift_hist_8gpu.sh >> $L/logs/icwm_hist_$(date +%s).log 2>&1
    else
        log "fresh start attempt $i"; rm -rf "$OUT"; bash $L/scripts/train_icwm_libero_tvshift_hist_8gpu.sh >> $L/logs/icwm_hist_$(date +%s).log 2>&1
    fi
    sleep 60
done
log "training DONE"

CK=outputs/train/$RUN/checkpoints/020000/pretrained_model
sync_code_and_ckpt() {
    local f
    for f in datasets/factory.py datasets/icwm_context_sampler.py datasets/icwm_frame_cache.py \
             policies/icwm/configuration_icwm.py policies/icwm/modeling_icwm.py policies/icwm/processor_icwm.py; do
        rsync -a -e ssh $L/lerobot/src/lerobot/$f hilo2:$R/lerobot/src/lerobot/$f || return 1
    done
    rsync -a -e ssh $L/lerobot/run_icwm_eval_tvshift.py hilo2:$R/lerobot/ || return 1
    ssh hilo2 "mkdir -p $R/$CK" && rsync -a --partial -e ssh $L/$CK/ hilo2:$R/$CK/
}
until sync_code_and_ckpt; do log "sync failed, retrying"; sleep 120; done
until ssh hilo2 "test -f $R/$EVAL/summary.json"; do
    if [ "$(ssh hilo2 "ps -eo cmd | grep -c '[r]un_icwm_eval_tvshift.py --history-context'")" = 0 ]; then
        log "launching eval on hilo2"
        ssh hilo2 "mkdir -p $R/$EVAL && cd $R/lerobot && (nohup setsid env HF_HOME=/root/.cache/huggingface TOKENIZERS_PARALLELISM=false PYTHONPATH=src \
          $R/conda_env312/bin/python run_icwm_eval_tvshift.py --history-context --ckpt $R/$CK --out $R/$EVAL \
          --gpus \$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,) --episodes 10 >> $R/$EVAL/eval.log 2>&1 &)"
    fi
    sleep 120
done
mkdir -p $L/$EVAL; rsync -a -e ssh hilo2:$R/$EVAL/ $L/$EVAL/
log "DONE: $(tail -3 $L/$EVAL/eval.log | tr '\n' ' ')"
