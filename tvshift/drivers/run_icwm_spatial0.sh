#!/usr/bin/env bash
# ICWM on LIBERO-Spatial only (all 10 tasks, 0°) on napali2: builds the frame cache once, then
# fresh start / resume from checkpoints/last after a crash or reboot, up to 20 attempts. Idempotent (flock).
set -uo pipefail
exec 9>/workspace/yujin/.icwm_spatial0.lock
flock -n 9 || { echo "icwm spatial0 already running"; exit 0; }
I=/workspace/yujin/icwm
OUT=$I/outputs/train/icwm_libero_spatial0
FINAL=$OUT/checkpoints/${STEPS:-20000}
CACHE=/workspace/yujin/data/icwm_spatial0_cache
log() { echo "[$(date -u '+%F %T')] icwm-spatial0: $*"; }
until [ -f $CACHE/DONE ]; do
    log "building frame cache"
    (cd $I/lerobot && $I/.venv_gpu/bin/python build_icwm_cache.py --out $CACHE \
        --root /workspace/yujin/data/libero_spatial_rot/lerobot_v30 --names libero_spatial_theta000_no_noops_lerobot) || sleep 60
done
i=0
while [ ! -d "$FINAL" ]; do
    while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -gt 2000 ]; do sleep 30; done
    [ $i -ge 20 ] && { log "giving up after $i attempts"; exit 1; }
    i=$((i+1))
    if [ -f "$OUT/checkpoints/last/pretrained_model/train_config.json" ]; then
        log "resume attempt $i"; bash $I/scripts/resume_icwm_libero_spatial0_8gpu.sh >> $I/logs/icwm_spatial0_$(date +%s).log 2>&1
    else
        log "fresh start attempt $i"; rm -rf "$OUT"; bash $I/scripts/train_icwm_libero_spatial0_8gpu.sh >> $I/logs/icwm_spatial0_$(date +%s).log 2>&1
    fi
    sleep 60
done
log "DONE"
