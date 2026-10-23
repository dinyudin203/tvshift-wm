# tvshift-wm

LIBERO **tvshift** 실험 코드 모음. 학습 때 task 마다 카메라 각도가 하나로 고정된 데이터로 학습한 뒤,
(1) 학습한 task–각도 쌍, (2) 학습 때 보지 못한 task–각도 조합, (3) 학습하지 않은 각도에서 평가해서
in-context 정보(world model)가 viewpoint shift 에 얼마나 도움이 되는지 비교한다.

| 모델 | 위치 | 설명 |
|---|---|---|
| **ICWM** | `icwm/` | In-Context World Modeling (arXiv 2606.26025). Qwen2.5-VL-3B + FAST action token, 상호작용 clip `(o^s, a, o^e)` N 개를 prompt 에 넣는다. LeRobot fork 위의 `icwm` policy |
| **Reflective-VLA** | `reflective_vla/` | Reflective VLA (arXiv 2606.25215). pi0.5 위에 observation–action–consequence triplet K 개를 넣는다 |
| **pi0.5 baseline** | `reflective_vla/` | 같은 코드에서 `K=1` (context 없음 = reactive pi0.5). `pi05_base` 또는 PaliGemma 초기화 |
| **tvshift** | `tvshift/` | 다각도 LIBERO 데이터 렌더링, 평가 환경의 카메라 회전, 실험 드라이버 |

## 디렉토리 구조

```
icwm/                         LeRobot fork (PYTHONPATH=src 로 사용) + ICWM
  src/lerobot/policies/icwm/        ICWMConfig / ICWMPolicy / sequence builder (processor)
  src/lerobot/datasets/icwm_*.py    context clip sampler (pool / history / sametask_past / probe / mixed), frame cache
  run_icwm_train.py                 lerobot_train 래퍼 (cuDNN off, Qwen2.5-VL conv3d 이슈)
  run_icwm_eval_tvshift.py          tvshift 평가 (active probing / --history-context / --mixed-context)
  build_icwm_cache.py               학습 프레임 디코드 캐시
  gen_icwm_probe_pool.py            학습용 probing clip pool 생성
  diag_icwm_*.py                    디코딩 / attention / label 진단
  scripts/train_*.sh, resume_*.sh   8 GPU 학습 launch 스크립트 (run 별)
  NOTES.md                          ICWM 구현 노트
reflective_vla/
  src/reflective_vla/               Reflective VLA policy (pi0.5 MoT 기반)
  scripts/train_tvshift.py          FSDP2 학습 (K=8 Reflective, K=1 pi0.5 baseline)
  scripts/run_tvshift.sh            torchrun launcher (K, STEPS, BS, ACCUM, WARMUP 환경변수)
  scripts/eval_tvshift.py           tvshift 평가
  src/reflective_vla/eval/env_worker.py   LIBERO 시뮬레이터 subprocess (모든 모델의 평가가 공유)
  third_party/lerobot/              LeRobot 0.6.2 (소스만)
tvshift/
  data/                             다각도 agentview 렌더링 + LeRobot v3.0 패킹 (render_shard.py, pack_lerobot.py)
  online_adapt/env/conditions.py    평가 환경 카메라 회전 (apply_camera_condition) -- env_worker 의 --adapt-repo
  drivers/                          학습 -> 평가 자동화 드라이버 (재시작 가능, flock)
docs/IMPLEMENTATION_NOTES_tvshift.txt   세 모델 구현/세팅 상세 노트
```

## tvshift 프로토콜

학습 데이터: 4 개 LeRobot v3.0 데이터셋 (각 2 task, 512x512 agentview + wrist, state 8, delta-EEF action 7), MuJoCo 3.3.2 로 렌더.

| 데이터셋 | task | 각도 |
|---|---|---|
| `libero_spatial_theta000` | spatial idx 1, 3 | 0° |
| `libero_goal_theta045` | goal idx 0, 7 | 45° |
| `libero_goal_theta345` | goal idx 5, 2 | 345° |
| `libero_object_theta270` | object idx 0, 1 | 270° |

평가 (각 조합 10 episode, horizon 800 step):
- **seen pair**: 8 task x 학습 각도 (80 ep)
- **novel pair**: 8 task x 다른 학습 각도 3 개 (240 ep)
- **novel angle**: 8 task x 학습하지 않은 각도 11 개 (30, 60, 90, 120, 135, 225, 255, 285, 300, 315, 330°) (880 ep), `TVSHIFT_SWEEP=1`

## 실행 예

경로는 원래 서버 기준 절대경로(`/workspace/yujin/...`)로 되어 있다. 다른 환경에서는 스크립트의 경로를 바꿔야 한다.

```bash
# ICWM 학습 (8 GPU)
bash icwm/scripts/train_icwm_libero_tvshift_v2_8gpu.sh

# ICWM 평가: seen + novel pair / novel angle
cd icwm && PYTHONPATH=src python run_icwm_eval_tvshift.py --ckpt <run>/checkpoints/020000/pretrained_model \
    --out <out> --gpus 0,1,2,3 --episodes 10 \
    --env-python <mujoco 3.3.2 env python> --libero-root <LIBERO> --adapt-repo ../tvshift
TVSHIFT_SWEEP=1 TVSHIFT_SWEEP_TASKS=0,1,2,3,4,5,6,7 PYTHONPATH=src python run_icwm_eval_tvshift.py ...

# pi0.5 base K=1 baseline 학습 / 평가
cd reflective_vla && OUT=outputs/tvshift_pi05base_K1 K=1 STEPS=5000 BS=1 ACCUM=4 WARMUP=500 bash scripts/run_tvshift.sh
python scripts/eval_tvshift.py --run-dir outputs/tvshift_pi05base_K1 --out <out> --gpus 0,1,2,3 --episodes 10 \
    --env-python <mujoco 3.3.2 env python> --libero-root <LIBERO> --adapt-repo ../tvshift
```

## License

`icwm/src/lerobot` 와 `reflective_vla/third_party/lerobot` 은 Hugging Face LeRobot (Apache License 2.0,
`reflective_vla/third_party/lerobot/LICENSE`) 을 기반으로 수정한 것이다.
