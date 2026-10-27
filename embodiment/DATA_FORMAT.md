# Embodiment-shift LIBERO 데이터 포맷 (arm + gripper 버전)

> 2026-10-26: 팔+그리퍼 변형 데이터 `/workspace/yujin/data/libero_embodiment_gripper/lerobot_v30/` (코드 `embodiment_swap_v2/`, `run_gripper_shards.sh` → `chain_pack.sh`).
> IIWA+Robotiq85 spatial 2,6 / UR5e+OnRobot RG2 (`OnRobotRG2Gripper`) goal 1,7 / Panda+PandaGripper object 7,8, task당 성공 첫 45개(N=전 task 최소 성공 수), 각 90 ep (10,836 / 8,553 / 13,694 frame).
> 포맷·feature 스키마는 아래 Robotiq 버전과 동일 (info.json features 일치 확인). Panda+Panda는 플래그 없이 기본값(v1과 동일).
> angle 변형: `/workspace/yujin/data/libero_embodiment_gripper_angle/lerobot_v30/` (`run_angle_shards.sh`, `render_emb_shard.py --theta`): IIWA+Robotiq 0° / UR5e+RG2 45° / Panda+Panda 270°, 같은 task·N=45. agentview만 libero_rot 규약대로 회전 (wrist·state·action 동일).

## (이하 원본: Robotiq 버전)

다른 서버에서 같은 데이터를 만들거나, 다른 모델(ICWM 등)용으로 변환할 때 맞춰야 하는 규약 정리.
기준 데이터: `/workspace/yujin/data/libero_embodiment_robotiq/lerobot_v30/`

## 1. 데이터셋 구성

| 데이터셋 (폴더명) | 로봇 | suite | task idx | 에피소드 | frame |
|---|---|---|---|---|---|
| `IIWA_Robotiq85_libero_spatial_no_noops_lerobot` | IIWA | libero_spatial | 6, 2 | 90 (task당 45) | 10,836 |
| `Panda_Robotiq85_libero_object_no_noops_lerobot` | Panda | libero_object | 7, 8 | 90 | 13,622 |
| `UR5e_Robotiq85_libero_goal_no_noops_lerobot` | UR5e | libero_goal | 7, 1 | 90 | 8,577 |

- 그리퍼는 전부 Robotiq 2F-85 (Menagerie 4-bar, `Robotiq85GripperMenagerieNoLinkCol`). 팔만 다름.
- task는 Panda 데모(`libero_raw/{suite}/{task.name}_demo.hdf5`)에서 task당 성공한 처음 45개 데모를 사용.

## 2. 파일 형식: LeRobot v3.0

```
<dataset>/meta/{info.json, stats.json, tasks.jsonl, tasks.parquet, episodes/chunk-000/file-000.parquet}
<dataset>/data/chunk-000/file-000.parquet
<dataset>/videos/observation.images.image/chunk-000/file-000.mp4
<dataset>/videos/observation.images.wrist_image/chunk-000/file-000.mp4
```
- fps 20, `robot_type: franka` (로봇 이름과 무관하게 고정), chunk 1000, data 100MB, video 200MB 단위로 분할.
- 영상: 512x512, h264, yuv420p. 이미 위아래가 바로 선 상태(렌더 후 `[::-1]` 적용)라 추가 flip 불필요.
- 소스 레이아웃/스키마는 `FastWAM/libero_rot/pack_lerobot.py`(yuanty/LIBERO-fastwam lerobot_v30와 동일)를 그대로 따름.

## 3. feature 정의

| key | shape | 의미 |
|---|---|---|
| `observation.images.image` | 512x512x3 video | agentview (기본 카메라, 로봇마다 동일 위치) |
| `observation.images.wrist_image` | 512x512x3 video | eye-in-hand. **그립 site 기준 Panda wrist cam과 같은 상대 pose**로 배치 (`Swap.align_wrist_cam`) |
| `observation.state` | 8 | **Panda 규약**: grip-site 위치(3) + 축-각도(3) + 손가락 2. 아래 참고 |
| `observation.states.ee_state` | 6 | `observation.state[:6]` |
| `observation.states.gripper_state` | 2 | `observation.state[6:]` |
| `observation.states.joint_state` | 7 | 그 로봇의 실제 팔 관절각. UR5e(6축)는 0으로 padding |
| `action` | 7 | OSC delta (x,y,z,roll,pitch,yaw) ∈[-1,1] + gripper(-1 열기, +1 닫기) |
| `timestamp, frame_index, episode_index, index, task_index` | | LeRobot 표준 |

### state 상세 (모델에 들어가는 값)
- 위치: `gripper0_grip_site`의 world 좌표.
- 회전: **Panda 손이 가졌을 자세**의 axis-angle (`site_xmat @ R_site_hand`). 즉 팔이 달라도 같은 의미의 6D ee pose.
- 손가락 2차원: Panda의 `(+w/2, -w/2)` 형태. Robotiq는 패드 표면 간격 기준으로 환산(열림 ≈ ±0.0427).
- 모델 입력으로는 `observation.state`(8-d)만 써도 로봇 간 형식이 동일. `joint_state`는 로봇마다 의미/차원이 다르니 cross-embodiment 모델에는 넣지 말 것.

### action 상세
- 원본 Panda action을 그대로 쓰지 않고 **새 로봇에서 실제로 실행한 action**을 기록.
- 매 step Panda가 노리던 절대 목표(위치 = grip 위치 + scaled delta, 자세 = 회전된 grip 자세)를 새 로봇의 현재 grip pose 기준 delta로 다시 표현한 뒤 clip(-1,1).
- gripper 차원은 원본 Panda 값(±1) 그대로.

## 4. 시간 정렬 규칙 (중요)
- 한 step의 기록 순서: **obs(렌더, state) 기록 → action 실행**. 즉 `state[t]`, 이미지`[t]`는 `action[t]` 실행 전 값 (eval 환경과 동일).
- no-op action(`render_shard.is_noop`)은 제거. 제거된 step은 영상/state/action 모두에서 같이 빠짐.
- 성공한 롤아웃만 저장 (300 중 282개 성공).

## 5. 시뮬레이션 설정 (데이터 재현 시 필수)
- 코드: `/workspace/yujin/embodiment_swap/` — `render_emb_shard.py`(생성), `swap_core.py`, `robotiq_orig.py`(그리퍼 등록), `env_worker_emb.py`(eval 환경).
- 환경: robosuite 1.4.1(로컬 패치됨), mujoco 3.3.2, `MUJOCO_GL=egl`, LIBERO(`hyunwoo/LIBERO`).
- 그리퍼 XML: `grippers/robotiq_2f85_menagerie_orig/..._localfinger_nolinkcol_robosuite.xml` (마운트 quat 수정판).
- 생성 플래그: `--gripper Robotiq85GripperMenagerieNoLinkCol --menagerie_opt --tip_comp --grip_speed 0.01 --modes goal,goal_fb,open`
  - `--menagerie_opt`: impratio=10, elliptic cone
  - `--tip_comp`: 4-bar 손끝 전진량(최대 13.3mm)만큼 목표 보정
  - `--grip_speed 0.01`: 그리퍼 명령 slew (Panda와 맞춤. 기본 0.2는 즉시 닫힘)
- eval은 동일 설정을 환경변수로 전달: `EMB_GRIPPER`, `EMB_MENAGERIE_OPT=1`, `EMB_GRIP_SPEED=0.01`. 학습/eval 설정이 어긋나면 안 됨.
- 초기화: 물체는 데모 첫 상태에서 복사, 팔은 Panda grip pose로 IK. 시작 시 그리퍼는 열린 상태(-1)로 설정.

## 6. 학습 쪽 (ICWM)
- context 샘플링: 데이터셋(=embodiment) 단위로 pool을 묶음 (`context_match_viewpoint=true`), context 5 clips.
- 이미지는 학습 시 224로 리사이즈 (eval과 동일: `cv2.INTER_AREA`).
- frame cache: `build_icwm_cache.py` 로 `/workspace/yujin/data/icwm_embrq_cache` 생성.

## 7. 모델별 변환 — 확인 필요
- 이 문서는 ICWM이 읽는 LeRobot v3.0 형식만 다룸. **다른 모델 6개**(어떤 모델인지, 각자의 입력 포맷: 이미지 해상도, state/action 정규화, action chunk, 언어 입력 등)는 아직 파악 못 함 → 목록을 받으면 모델별로 항목 추가.
