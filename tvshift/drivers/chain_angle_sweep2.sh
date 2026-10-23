#!/usr/bin/env bash
# phase 2 of the angle sweep (remaining 5 trained tasks) starts once phase 1 (3 tasks) of the same model is done
WHICH=${1:?wam|icwm}
if [ "$WHICH" = wam ]; then PH1=/workspace/yujin/lingbot-va/evaluate_results/angle_sweep_wan22init; else PH1=/workspace/yujin/icwm/evaluate_results/angle_sweep_v2_step20000; fi
while [ ! -f $PH1/summary.json ]; do sleep 300; done
exec /workspace/yujin/run_angle_sweep2_hilo2.sh $WHICH
