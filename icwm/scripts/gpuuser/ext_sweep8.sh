#!/usr/bin/env bash
# Extend the unseen-angle sweep from 3 tasks (0,2,6) to all 8 tvshift tasks: 11 angles x 8 tasks = 88 combos x 10 eps.
# Writes into each model's existing sweep_* dir; the 33 finished combos are skipped (their summary jsons exist) and
# summary.json is rebuilt over all 88 (the 3-task one is kept as summary_3task.json). One model at a time, all GPUs.
#   ./ext_sweep8.sh icwm:<run dir>:<sweep dir> ... pi05:<pi05_finetune|pi05_pretrain> ... wam
export LIBERO_CONFIG_PATH=/dataset/yujin/libero_cfg HF_HOME=/dataset/yujin/hf_home TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export TVSHIFT_SWEEP=1 TVSHIFT_SWEEP_TASKS=0,1,2,3,4,5,6,7
E=/dataset/yujin/eval; CK=/dataset/yujin/vla/runs
ICWM_DIR=/home/gpuuser/yujin/repos/icwm/icwm/lerobot
export ICWM_SRC=$ICWM_DIR/src ICWM_ENV_WORKER=/home/gpuuser/yujin/repos/reflective-va/reflectiveVLA/src/reflective_vla/eval/env_worker.py
G8=0,1,2,3,4,5,6,7
ts() { date +%m-%d\ %H:%M; }
keep3() { [ -f $1/summary.json ] && [ ! -f $1/summary_3task.json ] && cp $1/summary.json $1/summary_3task.json; }
for m in "$@"; do
  until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 30; done
  IFS=: read kind a b x <<< "$m"   # icwm:<run>:<sweep dir>[:<extra eval flag>]
  echo "[$(ts)] sweep8 $m"
  for gpus in $G8,$G8,$G8,$G8 $G8,$G8 $G8,$G8; do   # 4 workers/GPU, retries at 2/GPU (OOM-safe)
    case $kind in
      icwm) keep3 $E/$b
            ( cd $ICWM_DIR && PYTHONPATH=$ICWM_SRC /dataset/yujin/envs/icwm/bin/python run_icwm_eval_tvshift.py $x --gpus $gpus --episodes 10 \
                --ckpt $CK/$a/checkpoints/006000/pretrained_model --out $E/$b --env-python /dataset/yujin/envs/tv/bin/python \
                --libero-root /home/gpuuser/yujin/repos/LIBERO --adapt-repo /home/gpuuser/yujin/repos/lingbot-va-adapt >> $E/${b}_sweep8.log 2>&1 ) ;;
      pi05) keep3 $E/sweep_${a}_s6000
            ( cd $E && /dataset/yujin/envs/lerobot/bin/python -I run_pi05_eval_tvshift.py --gpus $gpus --episodes 10 \
                --ckpt $CK/${a}_b128_s6000/checkpoints/006000/pretrained_model --out $E/sweep_${a}_s6000 >> $E/sweep_${a}_sweep8.log 2>&1 ) ;;
      wam)  keep3 $E/sweep_wamidm_s6000   # WAM: server+client per slot, 2 slots/GPU as in the original sweep
            ( cd /home/gpuuser/yujin/repos/wam-idm && source $E/wam_env.sh && EVAL_PORT_OFFSET=50 /dataset/yujin/envs/tv/bin/python tools/tvshift_mot_eval.py \
                --ckpt /dataset/yujin/wamidm/runs/libero_tvshift_mot_idm_b128/checkpoints/checkpoint_step_6000 \
                --out $E/sweep_wamidm_s6000 --gpus $G8,$G8 --render-gpu own --episodes 10 --variant idm \
                --server-python $SERVER_PYTHON --client-python $CLIENT_PYTHON --libero-root $LIBERO_ROOT >> $E/sweep_wamidm_sweep8.log 2>&1 ) ;;
    esac && { echo "[$(ts)] sweep8 $m done"; break; }
    echo "[$(ts)] sweep8 $m incomplete, relaunching"
    until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 30; done
  done
done
echo "[$(ts)] sweep8 batch done: $*"
