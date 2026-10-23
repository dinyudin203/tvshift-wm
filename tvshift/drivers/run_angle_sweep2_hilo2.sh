#!/usr/bin/env bash
# Angle sweep on hilo2: WAM-IDM (Wan2.2 init, step 5000) and ICWM v2 (step 20000), one task per suite
# (spatial idx1, goal idx0, object idx0) x 11 angles outside the trained set, 10 episodes each (33 combos/model).
# Usage: run_angle_sweep_hilo2.sh wam|icwm. Resumable (finished combos skipped), flock-protected.
set -uo pipefail
WHICH=${1:?wam|icwm}
exec 7>/workspace/yujin/.angle_sweep2_${WHICH}.lock
flock -n 7 || { echo "already running"; exit 0; }
log() { echo "[$(date -u '+%F %T')] sweep-$WHICH: $*"; }
if [ "$WHICH" = wam ]; then
    L=/workspace/yujin/lingbot-va; R=/workspace/yujin/lingbot-va-tvshift-eval
    CK=train_out/libero_tvshift_mot_idm/checkpoints/checkpoint_step_5000; OUT=evaluate_results/angle_sweep2_wan22init
    until rsync -a --partial -e ssh --exclude __pycache__ $L/wan_va $L/evaluation $L/tools $L/script hilo2:$R/ \
       && ssh hilo2 "mkdir -p $R/$CK" && rsync -a --partial --inplace --exclude training_state.pt -e ssh $L/$CK/ hilo2:$R/$CK/; do log "sync failed"; sleep 60; done
    LAUNCH="cd $R && mkdir -p $OUT && TVSHIFT_SWEEP=1 TVSHIFT_SWEEP_TASKS=1,3,4,5,7 /opt/conda/envs/lingbot-va/bin/python tools/tvshift_mot_eval.py --ckpt $CK --variant idm --out $OUT --gpus 0,1,2,3 --episodes 10 --server-python /opt/conda/envs/lingbot-va/bin/python --client-python /opt/conda/envs/lingbot-va/bin/python --libero-root /workspace/lerobot-libero"
    PAT='[t]vshift_mot_eval.py --ckpt.*angle_sweep2'
else
    L=/workspace/yujin/icwm; R=$L
    CK=outputs/train/icwm_libero_tvshift_v2/checkpoints/020000/pretrained_model; OUT=evaluate_results/angle_sweep2_v2_step20000
    until rsync -a -e ssh $L/lerobot/run_icwm_eval_tvshift.py hilo2:$R/lerobot/; do sleep 60; done
    LAUNCH="mkdir -p $R/$OUT && cd $R/lerobot && env HF_HOME=/root/.cache/huggingface TOKENIZERS_PARALLELISM=false PYTHONPATH=src TVSHIFT_SWEEP=1 TVSHIFT_SWEEP_TASKS=1,3,4,5,7 $R/conda_env312/bin/python run_icwm_eval_tvshift.py --ckpt $R/$CK --out $R/$OUT --gpus 0,1,2,3 --episodes 10"
    PAT='[r]un_icwm_eval_tvshift.py --ckpt.*angle_sweep2'
fi
until ssh hilo2 "test -f $R/$OUT/summary.json"; do
    if [ "$(ssh hilo2 "ps -eo cmd | grep -c '$PAT'")" = 0 ]; then
        log "launching"; ssh hilo2 "mkdir -p $R/$OUT; (nohup setsid bash -c '$LAUNCH' >> $R/$OUT/eval.log 2>&1 &)"
    fi
    sleep 300
done
mkdir -p $L/$OUT; rsync -a -e ssh --exclude _logs hilo2:$R/$OUT/ $L/$OUT/
log "DONE: $(tail -3 $L/$OUT/eval.log | tr '\n' ' ')"
