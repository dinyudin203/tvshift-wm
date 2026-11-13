#!/usr/bin/env bash
# napali2: after the pair5 pluckerearly chain (scripts/pair5early_after_fine.sh: train + eval) is done, train the two
# 10cm pair5 motion-reasoning runs on GPUs 0-7 (rank batch 8, global 64, 12000 steps). The 5cm runs are on H100.
#   cd /dataset/yujin/vla/icwm_reason && nohup setsid bash queue_motion10_napali.sh >> logs/queue10.log 2>&1 < /dev/null &
set -uo pipefail
R=/dataset/yujin/vla/icwm_reason
cd $R
say() { echo "$(date "+%m-%d %H:%M:%S") $*"; }
echo $$ > queue10.pid
say "waiting for pair5early_after_fine.sh (train + eval) to finish"
while pgrep -f "[p]air5early_after_fine.sh" >/dev/null || pgrep -f "[n]api_train_eval.sh" >/dev/null \
   || pgrep -f "[n]apali_train_eval.sh" >/dev/null || pgrep -f "[r]un_icwm_train.py" >/dev/null || pgrep -f "[e]valuate.py" >/dev/null; do
  sleep 120
done
say "napali2 free; waiting 5 min for GPUs to drain"; sleep 300
if pgrep -f "[r]un_icwm_train.py|[e]valuate.py" >/dev/null; then say "another job started on napali2 meanwhile; aborting"; exit 1; fi
for spec in molmoact:10 ki:10; do
  m=${spec%%:*}; c=${spec##*:}
  OUT=$R/runs/icwm_pair5_ctx5_motion${m}_${c}cm_chunk_step_b64_s12000
  if [ -f $OUT/checkpoints/012000/pretrained_model/model.safetensors ]; then say "skip $m ${c}cm (done)"; continue; fi
  if [ -d $OUT ]; then mv $OUT ${OUT}.partial_$(date +%s); fi
  say "train $m ${c}cm -> $OUT"
  MODE=$m CM=$c OUT=$OUT PORT=29720 bash run_motion_napali.sh > logs/train_${m}_${c}cm.log 2>&1
  say "finished $m ${c}cm exit=$?"
done
say "QUEUE10_DONE"
