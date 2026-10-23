# libero_rot — LIBERO-Spatial 다각도 agentview 데이터셋 + Fast-WAM(joint) 학습

lingbot-va 가 0/30도에서만 잘 안 되던 문제를 보고, agentview 카메라를
**0, 30, 60, 90, 120, 240, 270, 300, 330도**에서 렌더링한 LIBERO-Spatial
데이터셋을 만들고 Fast-WAM joint 를 학습시키기 위한 파이프라인.

## 카메라 회전 규약

`robot0_link0`(로봇 베이스)의 z축을 중심으로 agentview 를 theta 도 회전.
수식은 `LIBERO-JY/xyg_scripts/only_rotate_dataset.py` 의
`rotate_camera_based_on_robot_base` 와 동일하다 (LIBERO 에서 base_pos=원점,
base_quat=단위쿼터니언이라 두 구현이 수치적으로 정확히 일치하는 것을 확인함).

같은 수식이 세 군데에 있고, 서로 일치하는지 검증했다:

| 위치 | 용도 |
|---|---|
| `LIBERO-JY/xyg_scripts/only_rotate_dataset.py` | 원본 |
| `libero_rot/render_shard.py: rotated_camera_pose` | 데이터 생성 |
| `FastWAM/experiments/libero/libero_utils.py: _rotate_agentview_sim` | 평가 환경 |

## 이미지 방향 규약 (중요)

- LIBERO 원본 obs 는 상하 반전 상태. `obs[...][::-1]` == `sim.render(...)[::-1]` 이다 (검증함).
- openvla `modified_libero_rlds` 및 그로부터 파생된 **FastWAM 공개 데이터셋은
  `obs[::-1, ::-1]`**, 즉 180도 회전(좌우 미러) 규약을 쓴다.
- **이 파이프라인이 만드는 데이터는 `[::-1]`** (시각적으로 올바른, 미러 없는 방향).

따라서 평가 시 `FASTWAM_LIBERO_IMAGE_FLIP=v` 를 반드시 켜야 한다.
기본값 `vh` 는 FastWAM 공개 체크포인트/데이터용이다.

## 파이프라인

### Stage A — 렌더링 (`render_shard.py`, `run_stage_a.sh`)

hdf5 의 `states` 를 `sim.set_state_from_flattened` 로 그대로 되돌리며 재렌더링한다
(액션 리플레이가 아니라서 드리프트 없음).

- wrist 카메라는 각도와 무관하므로 demo 당 1번만 렌더해 9개 데이터셋이 공유
- no-op 프레임 제거: openvla `modified_libero_rlds` 와 동일한 판정
- 마지막 state 로 `check_success()` 실패한 demo 는 제외
- 출력: 에피소드당 all-intra h264 mp4 (theta별 + wrist) + `frames/*.npz` + `_done/*.json`

```bash
cd /workspace/yujin/libero_rot && ./run_stage_a.sh      # 30 워커
```

### Stage B — LeRobot v3.0 패킹 (`pack_lerobot.py`, `run_stage_b.sh`)

theta 별로 하나씩, 총 9개의 LeRobot v3.0 데이터셋을 만든다. 스키마는
`yuanty/LIBERO-fastwam` 의 `lerobot_v30/libero_spatial_no_noops_lerobot` 와 동일:

- features: `observation.images.image`, `observation.images.wrist_image` (512x512 h264),
  `observation.state`(8), `observation.states.{ee,joint,gripper}_state`, `action`(7)
- fps 20, chunked parquet/video, `meta/{info.json,tasks.parquet,episodes/*,stats.json}`
- 에피소드 mp4 는 ffmpeg concat demuxer 로 무손실 이어붙이고 from/to_timestamp 를 기록

```bash
cd /workspace/yujin/libero_rot && ./run_stage_b.sh      # theta 9개 병렬
```

FastWAM 자체 리더(`fastwam.datasets.lerobot3.LeRobotDataset`)로 로드 검증 완료.

## 학습 (Fast-WAM joint)

```bash
cd /workspace/yujin/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH=$(pwd)/checkpoints
bash scripts/train_zero1.sh 8 task=libero_joint_2cam224_rot9_1e-4
```

- `configs/data/libero_spatial_rot9_lerobot_v30.yaml` — 9개 각도 데이터셋
- `configs/task/libero_joint_2cam224_rot9_1e-4.yaml` — joint 레시피
- A6000 48GB 8장이므로 FastWAM 기본 `batch_size: 16`(H20 96GB 기준)은 줄여야 할 수 있다.

## 평가

```bash
export FASTWAM_LIBERO_IMAGE_FLIP=v          # 필수
export FASTWAM_LIBERO_CAMERA_THETA=60       # 평가할 각도
python experiments/libero/run_libero_manager.py task=libero_joint_2cam224_rot9_1e-4 ...
```

주의: `_rotate_agentview` 는 `env_num=1` 에서만 동작한다 (SubprocVectorEnv 미지원).

## 알려진 제약

이 컨테이너에는 NVIDIA EGL/GL 라이브러리가 없어 MuJoCo offscreen 이 소프트웨어
렌더링(llvmpipe)으로 돈다. 프로세스당 약 4~5 renders/s 이고 프로세스 병렬은
거의 선형으로 스케일해서, 30 워커로 전체 생성에 1.5~2시간 정도 걸린다.
