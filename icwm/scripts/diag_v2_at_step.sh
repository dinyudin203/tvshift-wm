#!/usr/bin/env bash
# Wait for an ICWM v2 checkpoint, ship it to hilo2 and run the decode diagnostic there
# (training samples -> free-running generation -> how many chunks decode / match the target).
set -uo pipefail
STEP=${1:-5000}
I=/workspace/yujin/icwm
CK=outputs/train/icwm_libero_tvshift_v2/checkpoints/$(printf %06d $STEP)/pretrained_model
OUT=$I/logs/diag_v2_step${STEP}.txt
log() { echo "[$(date -u '+%F %T')] diag: $*"; }
log "waiting for $CK"
last=-1; n=0
while [ "$n" -lt 3 ]; do
    s=$(stat -c %s "$I/$CK/model.safetensors" 2>/dev/null || echo 0)
    if [ "$s" -eq "$last" ] && [ "$s" -gt 0 ]; then n=$((n+1)); else n=0; fi
    last=$s; sleep 60
done
log "syncing code + checkpoint to hilo2"
for f in datasets/factory.py datasets/icwm_context_sampler.py datasets/icwm_frame_cache.py \
         policies/icwm/configuration_icwm.py policies/icwm/modeling_icwm.py policies/icwm/processor_icwm.py; do
    rsync -a $I/lerobot/src/lerobot/$f hilo2:$I/lerobot/src/lerobot/$f
done
rsync -a $I/lerobot/diag_icwm_decode.py $I/lerobot/diag_icwm_labels.py hilo2:$I/lerobot/
ssh hilo2 "mkdir -p $I/$CK" && rsync -a --partial $I/$CK/ hilo2:$I/$CK/
log "running the diagnostic on hilo2 GPU 0"
ssh hilo2 "cd $I/lerobot && CUDA_VISIBLE_DEVICES=0 HF_HOME=/root/.cache/huggingface TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONPATH=src \
  $I/.venv_gpu/bin/python diag_icwm_decode.py --ckpt $I/$CK --n 40 2>&1" \
  | grep -v "Warning\|warn\|WARNING\|^Tokens\|Error decoding\|processor_kwargs\|use_fast\|Loading weights" > $OUT
log "done -> $OUT"
tail -20 $OUT
