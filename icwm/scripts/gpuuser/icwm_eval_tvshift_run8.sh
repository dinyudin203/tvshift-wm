#!/usr/bin/env bash
# Eval an ICWM tvshift run (checkpoint 006000): main seen/novel (320 eps), then the 8-task unseen-angle sweep
# (8 tasks x 11 angles = 88 combos, 880 eps). Each runs alone on all GPUs at 4 workers/GPU (a worker peaks at
# 11-15GB), retries at 2/GPU. Resumable (finished combos are skipped).
#   ./icwm_eval_tvshift_run8.sh <run dir name under vla/runs> <tag> ["extra eval flags"]  -> eval/<tag>_s6000, eval/sweep_<tag>_s6000
RUN=$1; TAG=$2; EXTRA=${3:-}
export LIBERO_CONFIG_PATH=/dataset/yujin/libero_cfg HF_HOME=/dataset/yujin/hf_home TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
E=/dataset/yujin/eval
CKPT=/dataset/yujin/vla/runs/$RUN/checkpoints/006000/pretrained_model
ICWM_DIR=/home/gpuuser/yujin/repos/icwm/icwm/lerobot
export ICWM_SRC=$ICWM_DIR/src ICWM_ENV_WORKER=/home/gpuuser/yujin/repos/reflective-va/reflectiveVLA/src/reflective_vla/eval/env_worker.py
ts() { date +%m-%d\ %H:%M; }
until [ -f $CKPT/model.safetensors ]; do sleep 60; done
G8=0,1,2,3,4,5,6,7
icwm() {  # out_prefix [sweep]
  for gpus in $G8,$G8,$G8,$G8 $G8,$G8 $G8,$G8; do
    until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 30; done
    echo "[$(ts)] $1$TAG start (gpus x$(( $(echo $gpus | tr ',' '\n' | wc -l) / 8 ))/GPU)"
    ( cd $ICWM_DIR && TVSHIFT_SWEEP=${2:-} TVSHIFT_SWEEP_TASKS=0,1,2,3,4,5,6,7 PYTHONPATH=$ICWM_SRC /dataset/yujin/envs/icwm/bin/python \
        run_icwm_eval_tvshift.py $EXTRA --gpus $gpus --episodes 10 --ckpt $CKPT --out $E/$1${TAG}_s6000 \
        --env-python /dataset/yujin/envs/tv/bin/python --libero-root /home/gpuuser/yujin/repos/LIBERO \
        --adapt-repo /home/gpuuser/yujin/repos/lingbot-va-adapt >> $E/$1${TAG}_parent.log 2>&1 ) && return 0
    echo "[$(ts)] $1$TAG incomplete, relaunching"
  done
}
icwm ""
icwm sweep_ 1
echo "[$(ts)] $TAG all done"
command grep -E "seen *:|novel *:" $E/${TAG}_parent.log | tail -2; command grep -E "sweep *:" $E/sweep_${TAG}_parent.log | tail -1
