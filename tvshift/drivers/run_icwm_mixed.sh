#!/usr/bin/env bash
# ICWM with both contexts (context_source=mixed: "Exploration:" 5 pool clips + "History:" 5 own-episode clips):
#   run D (icwm_libero_tvshift_mixdrop, context dropout 0.3/0.3) then run C (icwm_libero_tvshift_mixed, no
#   dropout, the control), each on napali2 (8 GPU, 20000 steps, = the hist run except the context source);
#   resume after a crash, up to 20 attempts, keep the 2 newest step checkpoints.
# Training only (hilo2 is unreachable). Evaluate later with
#   run_icwm_eval_tvshift.py --mixed-context --mixed-keep both|explore|history --ckpt .../020000/pretrained_model
# Idempotent (flock).
set -uo pipefail
exec 9>/workspace/yujin/.icwm_mixed.lock
flock -n 9 || { echo "icwm mixed already running"; exit 0; }
L=/workspace/yujin/icwm; R=/workspace/yujin/icwm
log() { echo "[$(date -u '+%F %T')] icwm-mixed: $*"; }

train() {  # $1 = run name suffix (mixdrop | mixed)
    local RUN=icwm_libero_tvshift_$1
    local OUT=$L/outputs/train/$RUN
    local FINAL=$OUT/checkpoints/020000
    ( while true; do ls -d $OUT/checkpoints/0* 2>/dev/null | sort | head -n -2 | xargs -r rm -rf; sleep 300; done ) 9>&- &
    local PRUNE=$!
    local i=0
    while [ ! -f "$FINAL/pretrained_model/model.safetensors" ]; do
        while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -gt 2000 ]; do sleep 30; done
        [ $i -ge 20 ] && { log "$1: giving up after $i attempts"; kill $PRUNE; return 1; }
        i=$((i+1))
        if [ -f "$OUT/checkpoints/last/pretrained_model/train_config.json" ]; then
            log "$1: resume attempt $i"; bash $L/scripts/resume_icwm_libero_tvshift_$1_8gpu.sh >> $L/logs/icwm_$1_$(date +%s).log 2>&1
        else
            log "$1: fresh start attempt $i"; rm -rf "$OUT"; bash $L/scripts/train_icwm_libero_tvshift_$1_8gpu.sh >> $L/logs/icwm_$1_$(date +%s).log 2>&1
        fi
        sleep 60
    done
    kill $PRUNE
    log "$1: training DONE"
}

train mixdrop || exit 1
train mixed || exit 1
log "ALL DONE"
