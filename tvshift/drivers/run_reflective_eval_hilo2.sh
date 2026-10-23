#!/usr/bin/env bash
# After Reflective VLA tvshift training finishes: ship the final export to hilo2, run the
# seen + novel-pair eval there, pull the results back. Idempotent and resumable.
set -uo pipefail
exec 6>/workspace/yujin/.reflective_eval.lock
flock -n 6 || { echo "reflective eval driver already running"; exit 0; }

L=/workspace/yujin/reflectiveVLA
RUN=outputs/tvshift_K8
STEPS=${STEPS:-5000}
W=$L/$RUN/export/step_${STEPS}/model.safetensors
R=/workspace/yujin/reflectiveVLA            # same path on hilo2
EPISODES=${EPISODES:-10}
log() { echo "[$(date -u '+%F %T')] reflective-eval: $*"; }

log "waiting for $RUN/export/step_${STEPS}"
last=-1; n=0
while [ "$n" -lt 3 ]; do
    s=$(stat -c %s "$W" 2>/dev/null || echo 0)
    if [ "$s" -eq "$last" ] && [ "$s" -gt 0 ]; then n=$((n+1)); else n=0; fi
    last=$s; sleep 120
done

log "syncing code + export to hilo2"
until rsync -a -e ssh --exclude __pycache__ $L/src $L/scripts hilo2:$R/ \
   && ssh hilo2 "mkdir -p $R/$RUN/export" \
   && rsync -a --partial -e ssh $L/$RUN/config $L/$RUN/stats.json hilo2:$R/$RUN/ \
   && rsync -a --partial -e ssh $L/$RUN/export/step_${STEPS} hilo2:$R/$RUN/export/; do
    log "sync failed, retrying"; sleep 60
done

OUT=evaluate_results/tvshift_K8
until ssh hilo2 "test -f $R/$OUT/summary.json"; do
    ssh hilo2 "cd $R && mkdir -p $OUT && GPUS=\$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,) && \
      (nohup setsid flock -n /tmp/reflective_eval.lock env TOKENIZERS_PARALLELISM=false ./conda_env/bin/python scripts/eval_tvshift.py \
       --run-dir $RUN --out $OUT --gpus \$GPUS --episodes $EPISODES >> $OUT/eval.log 2>&1 &)"
    sleep 300
done
mkdir -p $L/$OUT
rsync -a -e ssh hilo2:$R/$OUT/ $L/$OUT/
log "DONE: $(tail -3 $L/$OUT/eval.log | tr '\n' ' ')"
