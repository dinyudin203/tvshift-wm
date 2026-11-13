#!/usr/bin/env bash
# H100 GPUs 4-7: after the running molmoact 5cm training (started by queue_motion_h100.sh) finishes, train ki 5cm.
# The 10cm runs moved to napali2 (queue_motion10_napali.sh there).
#   cd /dataset/yujin/vla/icwm_reason && nohup setsid bash queue_motion5_h100.sh >> logs/queue_h100.log 2>&1 < /dev/null &
set -uo pipefail
R=/dataset/yujin/vla/icwm_reason
cd $R
say() { echo "$(date '+%m-%d %H:%M:%S') $*"; }
echo $$ > queue5_h100.pid
say "waiting for molmoact 5cm training to finish"
while pgrep -f "[r]un_icwm_train.py.*motion_reasoning_mode=molmoact" >/dev/null; do sleep 120; done
M=$R/runs/icwm_pair5_ctx5_motionmolmoact_5cm_chunk_step_b64_s12000
[ -f $M/checkpoints/012000/pretrained_model/model.safetensors ] && say "molmoact 5cm done" || say "WARNING: molmoact 5cm ended without step 12000 checkpoint"
for spec in ki:5; do
  m=${spec%%:*}; c=${spec##*:}
  OUT=$R/runs/icwm_pair5_ctx5_motion${m}_${c}cm_chunk_step_b64_s12000
  if [ -f $OUT/checkpoints/012000/pretrained_model/model.safetensors ]; then say "skip $m ${c}cm (done)"; continue; fi
  if [ -d $OUT ]; then mv $OUT ${OUT}.partial_$(date +%s); fi
  say "train $m ${c}cm -> $OUT"
  MODE=$m CM=$c OUT=$OUT PORT=29720 bash run_motion_h100.sh > logs/train_h100_${m}_${c}cm.log 2>&1
  say "finished $m ${c}cm exit=$?"
done
say "QUEUE5_DONE"
