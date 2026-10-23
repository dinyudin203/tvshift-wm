#!/usr/bin/env bash
# ICWM v2 settings on LIBERO-Spatial x 8 azimuths on napali2: builds the frame cache once (8 datasets
# in parallel, ~770GB), then fresh start / resume from checkpoints/last after a crash or reboot, up to
# 20 attempts. Disk is tight (22GB per checkpoint), so only the 2 newest step checkpoints are kept.
# Idempotent (flock).
set -uo pipefail
exec 9>/workspace/yujin/.icwm_spatial8_v2.lock
flock -n 9 || { echo "icwm spatial8_v2 already running"; exit 0; }
I=/workspace/yujin/icwm
OUT=$I/outputs/train/icwm_libero_spatial8_v2
FINAL=$OUT/checkpoints/0${STEPS:-20000}
CACHE=/workspace/yujin/data/icwm_spatial8_cache
ROOT=/workspace/yujin/data/libero_spatial_rot/lerobot_v30
NAMES="030 060 090 120 240 270 300 330"
log() { echo "[$(date -u '+%F %T')] icwm-spatial8_v2: $*"; }
until [ -f $CACHE/DONE ]; do
    log "building frame cache"
    mkdir -p $CACHE
    for a in $NAMES; do
        n=libero_spatial_theta${a}_no_noops_lerobot
        (cd $I/lerobot && OMP_NUM_THREADS=2 $I/.venv_gpu/bin/python build_icwm_cache.py --out $CACHE \
            --root $ROOT --only $n > $I/logs/build_cache_spatial8_$a.log 2>&1) &
    done
    wait
    ok=1
    for a in $NAMES; do [ -f $CACHE/libero_spatial_theta${a}_no_noops_lerobot__meta.json ] || ok=0; done
    if [ $ok = 1 ]; then echo ok > $CACHE/DONE; log "cache built"; else log "cache build incomplete, retrying"; sleep 60; fi
done
prune() {  # keep the 2 newest numbered checkpoints
    while true; do
        ls -d $OUT/checkpoints/0* 2>/dev/null | sort | head -n -2 | xargs -r rm -rf
        sleep 300
    done
}
prune & PRUNE=$!
trap 'kill $PRUNE 2>/dev/null' EXIT
i=0
while [ ! -d "$FINAL" ]; do
    while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -gt 2000 ]; do sleep 30; done
    [ $i -ge 20 ] && { log "giving up after $i attempts"; exit 1; }
    i=$((i+1))
    if [ -f "$OUT/checkpoints/last/pretrained_model/train_config.json" ]; then
        log "resume attempt $i"; bash $I/scripts/resume_icwm_libero_spatial8_v2_8gpu.sh >> $I/logs/icwm_spatial8_v2_$(date +%s).log 2>&1
    else
        log "fresh start attempt $i"; rm -rf "$OUT"; bash $I/scripts/train_icwm_libero_spatial8_v2_8gpu.sh >> $I/logs/icwm_spatial8_v2_$(date +%s).log 2>&1
    fi
    sleep 60
done
log "DONE"
