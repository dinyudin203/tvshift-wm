#!/usr/bin/env bash
# Run Reflective VLA tvshift training after ICWM tvshift has finished. Idempotent/resumable
# (train_tvshift.py resumes from ckpt/last); flock prevents duplicates.
set -uo pipefail
exec 7>/workspace/yujin/.reflective_chain.lock
flock -n 7 || { echo "reflective chain already running"; exit 0; }
ICWM_FINAL=/workspace/yujin/icwm/outputs/train/icwm_libero_tvshift/checkpoints/$(printf %06d "${ICWM_STEPS:-20000}")
R=/workspace/yujin/reflectiveVLA
STEPS=${STEPS:-5000}
FINAL=$R/outputs/tvshift_K8/export/step_${STEPS}
log() { echo "[$(date -u '+%F %T')] reflective: $*"; }
log "waiting for ICWM final checkpoint $ICWM_FINAL"
while [ ! -d "$ICWM_FINAL" ]; do sleep 120; done
sleep 120   # let ICWM's last save and process exit settle
i=0
while [ ! -d "$FINAL" ]; do
    while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -gt 2000 ]; do sleep 30; done
    [ $i -ge 20 ] && { log "giving up after $i restarts"; exit 1; }
    i=$((i+1)); log "launch attempt $i"
    ( cd $R && STEPS=$STEPS bash scripts/run_tvshift.sh >> outputs_tvshift_K8_$(date +%s).log 2>&1 )
    sleep 60
done
log "DONE"
