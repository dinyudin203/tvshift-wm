# icwm_motion — pixel 축 motion reasoning ICWM (pair5)

ICWM (Qwen2.5-VL-3B + FAST action token, context clip 5 개) 이 action 을 내기 전에
"이 카메라에서 EEF 를 이미지 +x(오른쪽) / +y(아래) / +z(카메라에서 멀어지는 방향) 로 X cm 움직이려면
base frame 에서 delta EEF 가 얼마인가" 를 텍스트로 먼저 답하게 하는 실험 코드.
정답은 카메라 회전에만 의존하므로 (viewpoint 마다 상수), 모델은 context clip 과 현재 이미지에서 viewpoint 를 읽어야 한다.

## 모드

| 이름 | 설정 | 방법 |
|---|---|---|
| MolmoAct 식 | `motion_reasoning_mode=molmoact` | assistant turn = reasoning 텍스트 → FAST action token. 전부 CE (action loss 가 VLM 으로 들어감). 추론 시 생성된 FAST token 을 decode |
| Knowledge insulation (LAP 식) | `motion_reasoning_mode=ki` | VLM 은 같은 CE (reasoning + FAST, `ki_vlm_fast_loss=true`). 별도 flow-matching action expert (width 1024, 8 층) 가 VLM hidden state 를 `.detach()` 로 cross-attend → flow loss 는 VLM 으로 가지 않음. expert 는 prefix + reasoning 까지만 보고 FAST token 은 보지 않음. 추론 시 reasoning 생성 → prompt+reasoning forward → Euler 10 step |

loss: `CE_action + w·CE_reason (+ flow, ki)`. 두 CE 는 각자 토큰 수로 평균 (reasoning ~50 토큰이 action ~15 토큰을 묻지 않게).

## Reasoning 정답 (`policies/icwm/motion_reasoning.py`)

1. calibration 의 `camera_to_world` (OpenGL) → OpenCV 축 (`diag(1,-1,-1)`) → `R_baseᵀ ·` 로 base frame
2. 행 = 이미지 +x/+y/+z 방향, `× X/100` m
3. 축별 gain (0.01092, 0.01245, 0.01072 m / action unit / step, 학습 데이터 least squares) 으로 action 단위
4. `motion_reasoning_units`: `chunk_step` (chunk 5 step 에 나눠 매 step 값, 기본) / `context_step` (context clip horizon 으로 나눔, horizon = chunk 이면 동일) / `total` / `cm`
5. 텍스트: `x:+0.92,+0.01,-0.10;y:+0.56,+0.13,-0.73;z:-0.69,-0.16,-0.59;` (부호 항상, 소수 2 자리, 회전·gripper 제외)

10cm 에서는 값이 ~1.55 까지 나와 실제 action 범위 (|a| ≲ 0.94) 를 넘는다 (방향·크기 의미만).

## 구성

```
overlay/src/lerobot/        icwm/src + icwm_plucker/overlay 위에 덮어쓰는 파일 (학습에 쓴 그대로, 전체 파일)
  policies/icwm/motion_reasoning.py   정답 계산 / format / parse
  policies/icwm/action_expert.py      KI flow-matching expert
  policies/icwm/{configuration,modeling,processor}_icwm.py   motion_reasoning_* / ki_* / action_expert_* 설정, 학습·추론 경로
  datasets/factory.py, datasets/icwm_plucker_dataset.py      motion_reasoning 일 때도 calibration sidecar 를 붙임
motion_reasoning_vs_plucker.patch   icwm_plucker 코드 대비 diff (리뷰용)
scripts/
  run_motion_napali.sh / run_motion_h100.sh   MODE=molmoact|ki CM=5|10 UNITS=chunk_step 학습 launch
  queue_motion5_h100.sh / queue_motion10_napali.sh   run 순차 실행 큐
tools/
  infer_motion_check.py   checkpoint 로 reasoning 생성 → 정답과 비교
  label_check.py          sequence builder 의 label / reasoning_end 확인
  eef_pixel_arrows.py     관측 이미지에 +x/+y/+z 1/2/3/5/10cm 이동의 pixel 화살표 시각화
```

코드 구성: `icwm/src` 복사 → `icwm_plucker/overlay/src` 덮어쓰기 → `icwm_motion/overlay/src` 덮어쓰기.

## 학습 설정 (pair5)

데이터: Spatial 0,2 @0° / Goal 0,1 @15° / Object 0,1 @30° / Long 1,0 @330° / Long 2,8 @345°
(`tvshift-libero-pair5-lerobot`, calibration `calibration_pair5`).
context 5 clip (같은 viewpoint pool, `normalize_context_actions`), front 카메라만, plucker 입력 없음 (`plucker_frame=base` 는 정답 계산용),
batch 64, 12000 step, lr 5e-5 (expert 1e-4), warmup 2000. run: molmoact/ki × 5cm/10cm, `chunk_step`.

주의: pair5 는 viewpoint 와 task/scene 이 겹쳐 있어서 학습 중 reasoning 정확도는 viewpoint 이해의 증거로 약하다.
