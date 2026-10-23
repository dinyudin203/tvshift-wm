#!/usr/bin/env bash
# After an angle_sweep2 run (wam|icwm) finishes, rerun it while some of its 55 combos have no summary
# (e.g. combos that died with CUDA OOM while other jobs shared the hilo2 GPUs). Finished combos are
# skipped by the eval tools, so only the missing ones run again. Up to 3 retries.
set -uo pipefail
WHICH=${1:?wam|icwm}
log() { echo "[$(date -u '+%F %T')] sweep2-retry-$WHICH: $*"; }
if [ "$WHICH" = wam ]; then R=/workspace/yujin/lingbot-va-tvshift-eval; L=/workspace/yujin/lingbot-va; OUT=evaluate_results/angle_sweep2_wan22init
else R=/workspace/yujin/icwm; L=$R; OUT=evaluate_results/angle_sweep2_v2_step20000; fi
for i in 1 2 3; do
    until [ -f $L/$OUT/summary.json ] && ! pgrep -f "run_angle_sweep2_hilo2.sh [$WHICH]" >/dev/null; do sleep 300; done
    n=$(ssh hilo2 "ls $R/$OUT/sweep_*/*_summary.json 2>/dev/null | wc -l")
    log "attempt $i: $n/55 combos have a summary"
    [ "$n" -ge 55 ] && { log "complete"; exit 0; }
    ssh hilo2 "rm -f $R/$OUT/summary.json"; rm -f $L/$OUT/summary.json
    bash /workspace/yujin/run_angle_sweep2_hilo2.sh $WHICH
done
log "still incomplete after 3 retries"
