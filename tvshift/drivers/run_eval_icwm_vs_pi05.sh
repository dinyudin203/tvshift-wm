#!/usr/bin/env bash
# ICWM v2 vs pi0.5 base K=1 on napali2 (hilo2 unreachable), while run D is paused:
#   1. wait for D (icwm_libero_tvshift_mixdrop) to write its 015000 checkpoint, then stop D and the
#      lock-waiter that would restart the mixed driver
#   2. both models evaluated at the same time on all 8 GPUs (one worker per GPU per model), MuJoCo 3.3.2
#      (fastwam env) like the training data: seen pair (8 combos) + novel pair (24) and the novel-angle
#      sweep (8 tasks x 11 untrained angles), 10 episodes each
#   3. restart run_icwm_mixed.sh: D resumes from 015000, then C
# Idempotent (flock); finished combos are skipped on a re-run.
set -uo pipefail
exec 8>/workspace/yujin/.eval_icwm_vs_pi05.lock
flock -n 8 || { echo "eval already running"; exit 0; }
log() { echo "[$(date -u '+%F %T')] eval-vs: $*"; }
CK=/workspace/yujin/icwm/outputs/train/icwm_libero_tvshift_mixdrop/checkpoints
ENVARGS="--env-python /workspace/envs/fastwam/bin/python --libero-root /workspace/yujin/hyunwoo/LIBERO --adapt-repo /workspace/yujin/lingbot-va-adapt"
GPUS=0,1,2,3,4,5,6,7
export ENV_WORKER_PYTHONPATH=""  # fastwam already has MuJoCo 3.3.2; the default wheel dir only exists on hilo2

if [ ! -f /workspace/yujin/.d_paused ]; then
    until [ "$(readlink $CK/last)" = 015000 ]; do sleep 60; done
    sleep 120  # let the save finish
    log "D wrote 015000: pausing D"
    pkill -f "until flock -n /workspace/yujin/.icwm_mixed.lock"
    pkill -f "^/workspace/yujin/icwm/.venv_gpu/bin/python -u run_icwm_train.py"
    pkill -f "^/workspace/yujin/icwm/.venv_gpu/bin/python /workspace/yujin/icwm/.venv_gpu/bin/accelerate"
    touch /workspace/yujin/.d_paused
fi
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -gt 2000 ]; do sleep 30; done
log "GPUs free"

icwm() {  # $1 = out dir, rest = env
    local out=$1; shift
    mkdir -p $out
    for i in 1 2 3; do
        [ -f $out/summary.json ] && break
        log "icwm $(basename $out): attempt $i"
        (cd /workspace/yujin/icwm/lerobot && env "$@" HF_HOME=/workspace/yujin/icwm/.hf_home TOKENIZERS_PARALLELISM=false PYTHONPATH=src \
            /workspace/yujin/icwm/.venv_gpu/bin/python run_icwm_eval_tvshift.py \
            --ckpt /workspace/yujin/icwm/outputs/train/icwm_libero_tvshift_v2/checkpoints/020000/pretrained_model \
            --out $out --gpus $GPUS --episodes 10 $ENVARGS >> $out/eval.log 2>&1)
    done
    log "icwm $(basename $out): $(tail -3 $out/eval.log | tr -s ' ' | tr '\n' ' ')"
}
pi05() {
    local out=$1; shift
    mkdir -p $out
    for i in 1 2 3; do
        [ -f $out/summary.json ] && break
        log "pi05 $(basename $out): attempt $i"
        (cd /workspace/yujin/reflectiveVLA && env "$@" TOKENIZERS_PARALLELISM=false ./conda_env/bin/python scripts/eval_tvshift.py \
            --run-dir outputs/tvshift_pi05base_K1 --out $out --gpus $GPUS --episodes 10 $ENVARGS >> $out/eval.log 2>&1)
    done
    log "pi05 $(basename $out): $(tail -3 $out/eval.log | tr -s ' ' | tr '\n' ' ')"
}
I=/workspace/yujin/icwm/evaluate_results; P=/workspace/yujin/reflectiveVLA/evaluate_results
( icwm $I/napali_v2_step20000 X=1; icwm $I/napali_v2_step20000_sweep TVSHIFT_SWEEP=1 TVSHIFT_SWEEP_TASKS=0,1,2,3,4,5,6,7 ) &
( pi05 $P/napali_pi05base_K1 X=1; pi05 $P/napali_pi05base_K1_sweep TVSHIFT_SWEEP=1 ) &
wait
log "evals DONE: resuming D"
rm -f /workspace/yujin/.d_paused
cd /workspace/yujin && setsid bash run_icwm_mixed.sh >> /workspace/yujin/icwm/logs/icwm_mixed_driver.log 2>&1 < /dev/null 8>&- &
log "ALL DONE"
