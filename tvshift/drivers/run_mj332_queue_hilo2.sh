#!/usr/bin/env bash
# Runs ON hilo2. Everything below renders the eval env with MuJoCo 3.3.2 (now the default of all three
# eval tools), like the tvshift training data; MuJoCo 3.8.1 rendered the libero_object scene much darker.
#   1. re-evaluate the libero_object combos (t6, t7: seen + novel, 10 episodes) of ICWM v2 / history /
#      same-task, Reflective-VLA and WAM-IDM, plus the t6/t7 angle sweep of ICWM v2 and WAM-IDM
#   2. inference videos, episode 0 of 16 combos (8 seen + 8 novel) per model; WAM-IDM also gets its
#      decoded per-chunk plan; pi0.5 finetune / PaliGemma and ICWM probe once their evals exist
# One job at a time on all 4 GPUs; each job is resumable and retried up to 3 times.
set -uo pipefail
exec 8>/tmp/mj332_queue.lock
flock -n 8 || { echo "queue already running"; exit 0; }
log() { echo "[$(date -u '+%F %T')] mj332: $*"; }
I=/workspace/yujin/icwm; F=/workspace/yujin/reflectiveVLA; W=/workspace/yujin/lingbot-va-tvshift-eval
IPY=$I/conda_env312/bin/python; WPY=/opt/conda/envs/lingbot-va/bin/python
G=0,1,2,3
OBJ=seen_t6_libero_object_idx0_theta270,seen_t7_libero_object_idx1_theta270
for th in 000 045 345; do OBJ=$OBJ,novel_t6_libero_object_idx0_theta$th; done
for th in 000 045 345; do OBJ=$OBJ,novel_t7_libero_object_idx1_theta$th; done
VID=seen_t0_libero_spatial_idx1_theta000,seen_t1_libero_spatial_idx3_theta000,seen_t2_libero_goal_idx0_theta045
VID=$VID,seen_t3_libero_goal_idx7_theta045,seen_t4_libero_goal_idx5_theta345,seen_t5_libero_goal_idx2_theta345
VID=$VID,seen_t6_libero_object_idx0_theta270,seen_t7_libero_object_idx1_theta270
VID=$VID,novel_t0_libero_spatial_idx1_theta045,novel_t1_libero_spatial_idx3_theta270,novel_t2_libero_goal_idx0_theta000
VID=$VID,novel_t3_libero_goal_idx7_theta270,novel_t4_libero_goal_idx5_theta000,novel_t5_libero_goal_idx2_theta045
VID=$VID,novel_t6_libero_object_idx0_theta000,novel_t7_libero_object_idx1_theta345
VDIR=$I/evaluate_results/inference_videos_mj332

job() {  # $1 = out dir (summary.json = done), $2 = working dir, rest = command
    local out=$1 cwd=$2; shift 2
    mkdir -p $out
    for i in 1 2 3; do
        [ -f $out/summary.json ] && break
        log "$(basename $out): attempt $i"
        (cd $cwd && "$@" >> $out/eval.log 2>&1)
    done
    log "$(basename $out): $(tail -3 $out/eval.log | tr '\n' ' ')"
}
icwm() {  # $1 = ckpt run name, $2 = out name, rest = extra args
    local ck=$I/outputs/train/$1/checkpoints/020000/pretrained_model out=$I/evaluate_results/$2; shift 2
    job $out $I/lerobot env HF_HOME=/root/.cache/huggingface TOKENIZERS_PARALLELISM=false PYTHONPATH=src \
        $IPY run_icwm_eval_tvshift.py --ckpt $ck --out $out --gpus $G "$@"
}
refl() {  # $1 = run dir, $2 = out name, rest = extra args
    local run=$1 out=$F/evaluate_results/$2; shift 2
    job $out $F env TOKENIZERS_PARALLELISM=false ./conda_env/bin/python scripts/eval_tvshift.py \
        --run-dir outputs/$run --out $out --gpus $G "$@"
}
wam() {  # $1 = out name, rest = extra args
    local out=$W/evaluate_results/$1; shift
    job $out $W $WPY tools/tvshift_mot_eval.py --ckpt train_out/libero_tvshift_mot_idm/checkpoints/checkpoint_step_5000 \
        --variant idm --out $out --gpus $G --server-python $WPY --client-python $WPY --libero-root /workspace/lerobot-libero "$@"
}

# ---- 1. libero_object re-evaluation with MuJoCo 3.3.2
TVSHIFT_ONLY=$OBJ icwm icwm_libero_tvshift_v2 tvshift_v2_step20000_obj_mj332 --episodes 10
TVSHIFT_ONLY=$OBJ icwm icwm_libero_tvshift_hist hist_step20000_obj_mj332 --episodes 10 --history-context
TVSHIFT_ONLY=$OBJ icwm icwm_libero_tvshift_sametask sametask_step20000_obj_mj332 --episodes 10
TVSHIFT_ONLY=$OBJ refl tvshift_K8 tvshift_K8_obj_mj332 --episodes 10
TVSHIFT_ONLY=$OBJ wam tvshift_mot_idm_obj_mj332 --episodes 10
TVSHIFT_SWEEP=1 TVSHIFT_SWEEP_TASKS=6,7 icwm icwm_libero_tvshift_v2 angle_sweep_obj_v2_step20000_mj332 --episodes 10
TVSHIFT_SWEEP=1 TVSHIFT_SWEEP_TASKS=6,7 wam angle_sweep_obj_wam_mj332 --episodes 10

# ---- 2. inference videos (episode 0)
TVSHIFT_ONLY=$VID ICWM_SAVE_VIDEO=$VDIR/icwm_v2 icwm icwm_libero_tvshift_v2 videorun_icwm_v2 --episodes 1
TVSHIFT_ONLY=$VID ICWM_SAVE_VIDEO=$VDIR/icwm_history icwm icwm_libero_tvshift_hist videorun_icwm_history --episodes 1 --history-context
TVSHIFT_ONLY=$VID ICWM_SAVE_VIDEO=$VDIR/icwm_sametask icwm icwm_libero_tvshift_sametask videorun_icwm_sametask --episodes 1
TVSHIFT_ONLY=$VID REFL_SAVE_VIDEO=$VDIR/reflective_vla refl tvshift_K8 videorun_reflective --episodes 1
TVSHIFT_ONLY=$VID wam videorun_wam_idm --episodes 1
(cd /workspace/yujin/wamplan/wan_va && CUDA_VISIBLE_DEVICES=0 $WPY /workspace/yujin/wamplan/wam_plan_hilo2.py \
    $W/evaluate_results/videorun_wam_idm $VDIR/wam_idm_with_plan >> $VDIR/wam_plan.log 2>&1)
log "wam plans: $(tail -1 $VDIR/wam_plan.log)"

# ---- models still training: wait for their regular eval to finish on hilo2, then record videos
for run in tvshift_pi05base_K1 tvshift_pi05pali_K1; do
    until [ -f $F/evaluate_results/$run/summary.json ]; do sleep 600; done
    TVSHIFT_ONLY=$VID REFL_SAVE_VIDEO=$VDIR/${run#tvshift_} refl $run videorun_$run --episodes 1
done
until [ -f $I/evaluate_results/probe_step20000/summary.json ]; do sleep 600; done
TVSHIFT_ONLY=$VID ICWM_SAVE_VIDEO=$VDIR/icwm_probe icwm icwm_libero_tvshift_probe videorun_icwm_probe --episodes 1
log "ALL DONE"
