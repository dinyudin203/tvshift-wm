#!/usr/bin/env bash
# ICWM v2 (step 20000) context ablation on hilo2: the full tvshift protocol (seen 8 + novel 24 combos,
# 10 episodes each) with blank context clips (black frames, motionless arm) and no probing.
# Resumable (finished combos are skipped).
set -uo pipefail
exec 7>/workspace/yujin/.icwm_noctx.lock
flock -n 7 || { echo "already running"; exit 0; }
L=/workspace/yujin/icwm; R=/workspace/yujin/icwm
CK=$R/outputs/train/icwm_libero_tvshift_v2/checkpoints/020000/pretrained_model
OUT=evaluate_results/noctx_v2_step20000
log() { echo "[$(date -u '+%F %T')] noctx: $*"; }
until rsync -a -e ssh $L/lerobot/run_icwm_eval_tvshift.py hilo2:$R/lerobot/; do sleep 60; done
until ssh hilo2 "test -f $R/$OUT/summary.json"; do
    if [ "$(ssh hilo2 "ps -eo cmd | grep -c '[r]un_icwm_eval_tvshift.py --no-context'")" = 0 ]; then
        log "launching on hilo2"
        ssh hilo2 "mkdir -p $R/$OUT && cd $R/lerobot && (nohup setsid env HF_HOME=/root/.cache/huggingface TOKENIZERS_PARALLELISM=false PYTHONPATH=src \
          $R/conda_env312/bin/python run_icwm_eval_tvshift.py --no-context --ckpt $CK --out $R/$OUT \
          --gpus \$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,) --episodes 10 >> $R/$OUT/eval.log 2>&1 &)"
    fi
    sleep 120
done
mkdir -p $L/$OUT; rsync -a -e ssh hilo2:$R/$OUT/ $L/$OUT/
log "DONE: $(tail -3 $L/$OUT/eval.log | tr '\n' ' ')"
