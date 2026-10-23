#!/usr/bin/env bash
# Runs ON niihau1 (1x A100 80GB). Re-evaluates ICWM v2 step 20000 on everything executed in the
# libero_object scene, with the probing fix (probe box relative to the EE start pose; before, the box
# was fixed at table height and the arm left the camera view in the floor scene):
#   1. seen/novel   : tasks 6,7 (obj0, obj1) x 4 trained angles            -> tvshift_v2_step20000_probefix_obj
#   2. wrong ctx    : same 8 combos, probing with the camera at the farthest -> wrongctx_v2_step20000_probefix_obj
#   3. ctx swap     : obj0 @270, 3 context variants                         -> ctxswap_v2_step20000_obj0_270_probefix
#   4. angle sweep  : tasks 6,7 x 11 extra angles                           -> angle_sweep_obj_v2_step20000_probefix
# 10 episodes each, NW worker processes sharing the one GPU. Resumable (finished combos are skipped).
set -uo pipefail
exec 9>/workspace/yujin/.icwm_objfix.lock
flock -n 9 || { echo "already running"; exit 0; }
R=/workspace/yujin/icwm
CK=$R/outputs/train/icwm_libero_tvshift_v2/checkpoints/020000/pretrained_model
NW=${NW:-7}
GPUS=$(yes 0 | head -n $NW | paste -sd,)
export HF_HOME=$R/.hf_eval LIBERO_CONFIG_PATH=$R/libero_cfg TOKENIZERS_PARALLELISM=false PYTHONPATH=src
PY=$R/conda_env312/bin/python
log() { echo "[$(date -u '+%F %T')] icwm-objfix: $*"; }
cd $R/lerobot

run() {  # $1 = out dir name, rest = extra args; retried until summary.json exists
    local out=$R/evaluate_results/$1; shift
    mkdir -p $out
    for i in 1 2 3 4 5; do
        [ -f $out/summary.json ] && break
        log "$(basename $out): attempt $i"
        $PY run_icwm_eval_tvshift.py --ckpt $CK --out $out --gpus $GPUS --episodes 10 "$@" >> $out/eval.log 2>&1
    done
    log "$(basename $out): $(tail -3 $out/eval.log | tr '\n' ' ')"
}

TVSHIFT_TASKS=6,7 run tvshift_v2_step20000_probefix_obj
TVSHIFT_TASKS=6,7 run wrongctx_v2_step20000_probefix_obj --wrong-ctx-angle
run ctxswap_v2_step20000_obj0_270_probefix --ctx-swap
TVSHIFT_SWEEP=1 TVSHIFT_SWEEP_TASKS=6,7 run angle_sweep_obj_v2_step20000_probefix
log "ALL DONE"
