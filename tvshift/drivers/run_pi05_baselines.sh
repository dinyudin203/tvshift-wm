#!/usr/bin/env bash
# Plain pi0.5 baselines on tvshift, same code/data/hyperparameters as Reflective-VLA tvshift_K8
# (FSDP2 on napali2's 8 GPUs, 5000 steps, effective batch 32 = 1 x accum 4 x 8, lr 1e-4, warmup 500,
# chunk 10) except K=1 (no context = reactive pi0.5):
#   1. pi05_base init  -> outputs/tvshift_pi05base_K1  (pi0.5 finetune)
#   2. PaliGemma init  -> outputs/tvshift_pi05pali_K1  (pi0.5 trained from the VLM; action expert random)
# Each run is evaluated on hilo2 (seen 8 + novel 24 combos, 10 episodes) while the next one trains.
# Resumable (train_tvshift.py resumes from <out>/ckpt/last), flock-protected.
set -uo pipefail
exec 9>/workspace/yujin/.pi05_baselines.lock
flock -n 9 || { echo "pi05 baselines already running"; exit 0; }
L=/workspace/yujin/reflectiveVLA; R=/workspace/yujin/reflectiveVLA   # same path on hilo2
STEPS=5000
log() { echo "[$(date -u '+%F %T')] pi05: $*"; }

train() {  # $1 = run name, rest = extra train args
    local run=$1; shift
    local i=0
    while [ ! -f "$L/outputs/$run/export/step_${STEPS}/model.safetensors" ]; do
        while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -gt 2000 ]; do sleep 30; done
        [ $i -ge 20 ] && { log "$run: giving up after $i attempts"; return 1; }
        i=$((i+1)); log "$run: train attempt $i"
        (cd $L && OUT=outputs/$run K=1 STEPS=$STEPS BS=1 ACCUM=4 WARMUP=500 MASTER_PORT=29581 \
            bash scripts/run_tvshift.sh "$@" >> $L/outputs_${run}_$(date +%s).log 2>&1)
        sleep 60
    done
    log "$run: training DONE"
}

evaluate() {  # $1 = run name; runs on hilo2, returns when the summary is back
    local run=$1 out=evaluate_results/$1
    until rsync -a -e ssh --exclude __pycache__ $L/src $L/scripts hilo2:$R/ \
       && ssh hilo2 "mkdir -p $R/outputs/$run/export" \
       && rsync -a --partial -e ssh $L/outputs/$run/config $L/outputs/$run/stats.json hilo2:$R/outputs/$run/ \
       && rsync -a --partial -e ssh $L/outputs/$run/export/step_${STEPS} hilo2:$R/outputs/$run/export/; do
        log "$run: sync failed, retrying"; sleep 60
    done
    until ssh hilo2 "test -f $R/$out/summary.json"; do
        if [ "$(ssh hilo2 "ps -eo cmd | grep -c '[e]val_tvshift.py --run-dir outputs/$run'")" = 0 ]; then
            log "$run: launching eval on hilo2"
            ssh hilo2 "cd $R && mkdir -p $out && GPUS=\$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,) && \
              (nohup setsid env TOKENIZERS_PARALLELISM=false ./conda_env/bin/python scripts/eval_tvshift.py \
               --run-dir outputs/$run --out $out --gpus \$GPUS --episodes 10 >> $out/eval.log 2>&1 &)"
        fi
        sleep 300
    done
    mkdir -p $L/$out; rsync -a -e ssh hilo2:$R/$out/ $L/$out/
    log "$run: eval DONE: $(tail -3 $L/$out/eval.log | tr '\n' ' ')"
}

train tvshift_pi05base_K1 || exit 1
evaluate tvshift_pi05base_K1 &
train tvshift_pi05pali_K1 --init-paligemma /workspace/yujin/checkpoints/paligemma-3b-pt-224 || exit 1
evaluate tvshift_pi05pali_K1
wait
log "ALL DONE"
