#!/usr/bin/env bash
# ICWM tvshift evaluation on hilo2 at several training checkpoints (default 10000 15000 20000).
#
# One sub-driver per step, each: wait until that checkpoint is fully written -> ship code and the
# checkpoint's pretrained_model to hilo2 -> run the seen + novel-pair eval with active probing
# (10 episodes per combo, all 4 GPUs) -> pull the results back to
# icwm/evaluate_results/tvshift_step<STEP>. At most MAX_CONCURRENT evaluations run on hilo2 at the
# same time (each takes ~15 GB per GPU). Idempotent and resumable: re-running skips finished steps
# and finished combos, and a per-step lock prevents duplicates.
set -uo pipefail

L=/workspace/yujin/icwm
R=/workspace/yujin/icwm                   # same path on hilo2
EPISODES=${EPISODES:-10}
MAX_CONCURRENT=${MAX_CONCURRENT:-2}
RUN=${ICWM_RUN:-icwm_libero_tvshift}          # training run under icwm/outputs/train/
PREFIX=${EVAL_PREFIX:-tvshift}                # results go to icwm/evaluate_results/<PREFIX>_step<STEP>
if [ $# -gt 0 ]; then STEPS=("$@"); else STEPS=(10000 15000 20000); fi
log() { echo "[$(date -u '+%F %T')] icwm-eval: $*"; }

sync_code_and_ckpt() {  # $1 = checkpoint dir relative to $L
    local f
    for f in datasets/factory.py datasets/icwm_context_sampler.py datasets/icwm_frame_cache.py \
             policies/icwm/configuration_icwm.py policies/icwm/modeling_icwm.py policies/icwm/processor_icwm.py; do
        rsync -a -e ssh $L/lerobot/src/lerobot/$f hilo2:$R/lerobot/src/lerobot/$f || return 1
    done
    rsync -a -e ssh $L/lerobot/run_icwm_eval_tvshift.py hilo2:$R/lerobot/ || return 1
    rsync -a -e ssh /workspace/yujin/reflectiveVLA/src/reflective_vla/eval/env_worker.py \
        hilo2:/workspace/yujin/reflectiveVLA/src/reflective_vla/eval/ || return 1
    ssh hilo2 "mkdir -p $R/$1" && rsync -a --partial -e ssh $L/$1/ hilo2:$R/$1/
}

eval_one() {
    local STEP=$1
    exec 5>/workspace/yujin/.icwm_eval_${PREFIX}_${STEP}.lock
    flock -n 5 || { log "step $STEP: driver already running"; return 0; }
    local CK=outputs/train/${RUN}/checkpoints/$(printf %06d "$STEP")/pretrained_model
    local OUT=evaluate_results/${PREFIX}_step${STEP}
    local TAG=${PREFIX}_${STEP}
    local W=$L/$CK/model.safetensors

    if [ -f "$L/$OUT/summary.json" ]; then log "step $STEP: already done"; return 0; fi
    log "step $STEP: waiting for $CK"
    local last=-1 n=0 s
    while [ "$n" -lt 3 ]; do
        s=$(stat -c %s "$W" 2>/dev/null || echo 0)
        if [ "$s" -eq "$last" ] && [ "$s" -gt 0 ]; then n=$((n+1)); else n=0; fi
        last=$s; sleep 120
    done

    log "step $STEP: syncing code + checkpoint to hilo2"
    until sync_code_and_ckpt "$CK"; do
        log "step $STEP: sync failed, retrying"; sleep 60
    done

    # remote launcher: its command line carries no pattern the concurrency check below matches
    ssh hilo2 "mkdir -p $R/$OUT && cat > /tmp/launch_icwm_eval_${TAG}.sh <<EOF
#!/usr/bin/env bash
cd $R/lerobot
GPUS=\\\$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)
exec env HF_HOME=/root/.cache/huggingface TOKENIZERS_PARALLELISM=false PYTHONPATH=src $R/conda_env312/bin/python run_icwm_eval_tvshift.py --ckpt $R/$CK --out $R/$OUT --gpus \\\$GPUS --episodes $EPISODES
EOF
chmod +x /tmp/launch_icwm_eval_${TAG}.sh"

    until ssh hilo2 "test -f $R/$OUT/summary.json"; do
        local mine running
        mine=$(ssh hilo2 "ps -eo cmd | grep -c '[r]un_icwm_eval_tvshift.py --ckpt.*/${PREFIX}_step${STEP} '")
        if [ "$mine" = 0 ]; then
            running=$(ssh hilo2 "ps -eo cmd | grep -c '[r]un_icwm_eval_tvshift.py --ckpt'")
            if [ "$running" -lt "$MAX_CONCURRENT" ]; then
                log "step $STEP: launching on hilo2 ($running other eval(s) running)"
                ssh hilo2 "(nohup setsid flock -n /tmp/icwm_eval_${TAG}.lock /tmp/launch_icwm_eval_${TAG}.sh >> $R/$OUT/eval.log 2>&1 &)"
            else
                log "step $STEP: waiting, $running evaluation(s) already on hilo2"
            fi
        fi
        sleep 300
    done
    mkdir -p $L/$OUT
    rsync -a -e ssh hilo2:$R/$OUT/ $L/$OUT/
    log "step $STEP DONE: $(tail -3 $L/$OUT/eval.log | tr '\n' ' ')"
}

for s in "${STEPS[@]}"; do
    eval_one "$s" &
    sleep 5
done
wait
log "all requested steps finished"
