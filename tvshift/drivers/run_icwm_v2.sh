#!/usr/bin/env bash
# ICWM tvshift v2 (label/decoding fixes) on napali2: fresh start, resumes from checkpoints/last after
# a crash or reboot, up to 20 attempts. Idempotent (flock).
set -uo pipefail
exec 9>/workspace/yujin/.icwm_v2.lock
flock -n 9 || { echo "icwm v2 already running"; exit 0; }
I=/workspace/yujin/icwm
OUT=$I/outputs/train/icwm_libero_tvshift_v2
FINAL=$OUT/checkpoints/020000
log() { echo "[$(date -u '+%F %T')] icwm-v2: $*"; }
i=0
while [ ! -d "$FINAL" ]; do
    while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -gt 2000 ]; do sleep 30; done
    [ $i -ge 20 ] && { log "giving up after $i attempts"; exit 1; }
    i=$((i+1))
    if [ -f "$OUT/checkpoints/last/pretrained_model/train_config.json" ]; then
        log "resume attempt $i"; bash $I/scripts/resume_icwm_libero_tvshift_v2_8gpu.sh >> $I/logs/icwm_v2_$(date +%s).log 2>&1
    else
        log "fresh start attempt $i"; rm -rf "$OUT"; bash $I/scripts/train_icwm_libero_tvshift_v2_8gpu.sh >> $I/logs/icwm_v2_$(date +%s).log 2>&1
    fi
    sleep 60
done
log "DONE"
