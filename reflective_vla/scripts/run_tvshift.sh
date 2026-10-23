#!/usr/bin/env bash
# Reflective VLA on tvshift (8 GPU). Resumes automatically from <out>/ckpt/last.
#   STEPS=20000 BS=4 K=8 bash scripts/run_tvshift.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
PY=./conda_env/bin/python
K="${K:-8}"; STEPS="${STEPS:-5000}"; BS="${BS:-1}"; ACCUM="${ACCUM:-4}"; WARMUP="${WARMUP:-500}"
OUT="${OUT:-outputs/tvshift_K${K}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# one process per GPU on a shared CPU: the default (one intra-op thread per core in every rank)
# oversubscribes the machine; see the ICWM profiling notes
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3} MKL_NUM_THREADS=${OMP_NUM_THREADS:-3}
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "${HOME}/.netrc" ]; then
    WANDB_API_KEY="$(awk '/machine api.wandb.ai/{f=1} f&&/password/{print $2; exit}' "${HOME}/.netrc")"; export WANDB_API_KEY
fi
exec ./conda_env/bin/torchrun --nproc_per_node="${NGPU:-8}" --master_port="${MASTER_PORT:-29571}" \
    scripts/train_tvshift.py --out "${OUT}" --K "${K}" --steps "${STEPS}" --bs "${BS}" --accum "${ACCUM}" --warmup "${WARMUP}" --save-every "${SAVE_EVERY:-500}" "$@"
