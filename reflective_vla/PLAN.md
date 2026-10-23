# Reflective VLA 구현 계획

논문: **Reflective VLA: In-Context Action Consequences Make VLAs Generalize**
(Qing Lian, Kent Yu, Lei Zhang, arXiv:2606.25215)
프로젝트 페이지: https://lianqing11.github.io/reflective-vla-page/ → **"Code Coming Soon" (코드 미공개)** → from-scratch 구현 필요

---

## 1. 논문 핵심 요약

### 문제의식
기존 VLA는 **reactive**하다: `A_t ~ π(A_t | L, O_t)`. 현재 관측 하나가 action에 필요한 상태를 전부 담고 있다고 가정한다.
그런데 camera-to-robot geometry, robot calibration, actuation bias 같은 **embodiment-specific latent factor `z`** 는
단일 프레임에서 식별 불가능하다 → 학습 환경에 overfit, 배포 시 일반화 실패.

### 제안
결정을 **(observation, action, consequence) triplet context** 에 조건화한다.

```
Eq.2  π(A_t | L, O_t, H) = ∫ π(A_t | L, O_t, z) · P(z | H) dz
Eq.3  P(z | H) ∝ P(z) · Π_i P_env(O'_i | O_i, A_i, z)      ← consequence가 z의 likelihood를 만든다
Eq.4  A_t ~ π_θ(A_t | L, H, O_t),   H = {(O_i, A_i, O'_i)}_{i=1}^{K-1}
```

핵심은 **O'_i (action 실행 후 관측)**. O, A만 주면 "action→observation" 매핑이 여전히 미식별 상태다.
Ablation이 이걸 정확히 보여준다: `O only` 72.3%, `O,A` 73.8%, `O,A,O'` **77.8%** (reactive 73.1%).

### 아키텍처
- **MoT (Mixture-of-Transformers) dual-system**: pretrained VLM prefix + flow-matching action expert suffix,
  **모든 layer에서 attention 공유** → π0.5와 동일한 골격.
- VLM backbone: **PaliGemma-3B** 또는 Qwen3-VL-2B.
- 모든 관측 모달리티(3rd-person image, wrist image, proprio)를 **VLM 토큰 시퀀스 하나로 라우팅**.
  proprio / 과거 action chunk는 2-layer MLP projector로 토큰 공간에 사영. 과거 action chunk = **8 토큰** (`g_A`).
- Action expert hidden dim 1024, flow matching.

### 토큰 레이아웃 (논문 Appendix A.3)
```
<|im_start|>user {instruction}
<|frame_start|> <3rd OBS_i> <Wrist OBS_i> <PROPRIO_i>
                <|action_start|><ACTION_i><|action_end|>
                <3rd OBS'_i+C> <Wrist OBS'_i+C> <PROPRIO_i+C> <|frame_end|>
<|frame_sep|>
[... K-1개 triplet 반복 ...]
<|frame_start|> <3rd OBS_t> <Wrist OBS_t> <PROPRIO_t>
```
- language instruction은 **triplet마다 재삽입**, temporal index `τ_i` 포함.
- consequence O'_i 는 O_i 와 **동일한 모달리티 전부** (양쪽 카메라 + proprio).
- 마지막 블록(현재 프레임)은 obs만, action/consequence 없음.

### Block-causal mask (Eq.6) — 논문의 핵심 트릭
target frame `t_k`의 query slot `Ã_{t_k}` 가 보는 것:
```
V_{t_k} = { T_{t_j} : j < k }  ∪  { (L, τ_{t_k}, O_{t_k}) }
```
명시적으로 **마스킹되는 것**:
- 자기 자신의 prefix action token `A_{t_k}`  ← 정답 유출 차단
- 자기 자신의 consequence `O'_{t_k}`        ← 미래 유출 차단
- 이후의 모든 triplet `{T_{t_j} : j > k}`
- 형제 query slot `{Ã_{t_j}}_{j≠k}`         ← 서로 못 봄

효과: K개 프레임을 **한 번의 forward로 전부 supervise** (대각선 계단 패턴).
```
L_train = Σ_{k=1..K} L_act(A_{t_k}; V_{t_k})     (base VLA와 동일한 flow-matching loss)
```

### Inference
- 완료된 triplet의 **rolling FIFO buffer** (chunk 실행 후 새 triplet append, 가득 차면 oldest evict).
- 과거 triplet의 VLM-side KV는 **한 번만 인코딩하고 캐시 재사용** → step당 비용 거의 상수.
- K=8이 K=1 대비 프롬프트 8배인데 latency는 **1.43배** (178ms vs 124ms). K=4는 1.19배.

### 실험 설정
| 항목 | LIBERO | SimplerEnv-Bridge | Real |
|---|---|---|---|
| K | 8 | 4 | 8 |
| C (action chunk) | 10 | 10 | 30 |
| peak LR (expert/projector) | 1e-4 | 1e-4 | 1e-4 |
| VLM LR multiplier | 0.1× | 0.1× | 0.1× |
| batch/GPU | 4 | 6 | — |
| steps | reactive 90k / **reflective 50k** | reactive 80k / reflective 160k | — |
| GPU | 8×H20 | 8×H20 | — |
| optimizer | AdamW β=(0.9,0.95), wd 0.01, grad clip 1.0, bf16 | | |

- **frame sampling(train)**: 인접 context frame 간격 = `C` + random gap `~U[0,15]` env step (SimplerEnv는 `U[0,5]`).
  (고정 stride shortcut 방지 목적 — Limitations에 명시)
- Table 4 전문: LIBERO reactive 90k/32/K=1, LIBERO Reflective 50k/4/K=8, LIBERO-Plus·Hard reactive 50k/32/K=1,
  LIBERO-Plus·Hard Reflective 90k/4/K=8, SimplerEnv reactive 80k/32/K=1, SimplerEnv Reflective 160k/6/K=4.
- **frame sampling(eval)**: gap = 0, 결정론적 고정 stride.
- VLM은 초기 구간 **frozen**, 이후 linear warm-up + cosine decay.
- action은 sim에서 **absolute end-effector control** 로 변환 (replay 기반). 단 real-world는 **delta EEF**(C=30).
- Limitations: K=8은 **학습 메모리 한계** 때문이며, 과거 action chunk로부터 정답을 외삽하는 shortcut이 존재하고
  **frame-boundary token + stride randomization** 으로 부분 완화한다고 명시 → 경계 토큰은 기능 요소다.
- LIBERO eval: task당 50 rollout, seed 42, horizon 800 (Long은 900).

### 결과 (2026-09-02 원문 verbatim 대조 완료)

**Table 1 — in-distribution**
| Method | Spatial | Object | Goal | Long | **LIBERO Avg** | Spoon | Carrot | Cube | Eggplant | **Simpler Avg** |
|---|---|---|---|---|---|---|---|---|---|---|
| π0.5 (published) | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 | – | – | – | – | – |
| MemoryVLA | 98.4 | 98.4 | 96.4 | 93.4 | 96.5 | 75.0 | 75.0 | 37.5 | 100.0 | 71.9 |
| Reactive π0.5† (K=1, matched) | 97.5 | 98.2 | 97.8 | 94.0 | 96.9 | 91.7 | 79.2 | 70.8 | 50.0 | 72.9 |
| **Reflective VLA** | 98.4 | 99.0 | 98.2 | 94.6 | **97.6** | 95.8 | 83.3 | 79.2 | 54.2 | **78.2** |

**Table 2 — 섭동 하 robustness**
| Method | Camera | Robot | Lang | Light | Bg | Noise | Layout | **Plus Avg** | Camera† | Rob.Calib† | **Hard Avg** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| OpenVLA-OFT | 92.8 | 30.3 | 85.8 | 94.9 | 93.9 | 89.3 | 77.6 | 80.7 | 72.2 | 43.1 | 57.7 |
| MemoryVLA | 93.1 | 42.1 | 84.2 | 92.1 | 92.7 | 89.1 | 77.5 | 81.5 | 72.1 | 49.2 | 60.7 |
| Reactive π0.5† | 90.0 | 50.0 | 92.9 | 92.0 | 85.8 | 90.2 | 75.0 | 82.3 | 74.0 | 55.2 | 64.6 |
| **Reflective VLA** | 95.7 | 72.9 | 92.2 | 92.1 | 92.7 | 95.4 | 73.0 | **87.7** | 76.3 | 61.3 | **68.8** |

Δ = **+5.4pp** (LIBERO-Plus), **+4.2pp** (Hard). 가장 큰 개별 이득은 **Robot 50.0→72.9 (+22.9pp)**.
Lang(-0.7)과 Layout(-2.0)은 오히려 소폭 하락 — consequence 컨텍스트가 만능이 아님을 보여준다.

**Table 3a — context 구성 (K=8 고정)** ← 논문의 핵심 주장
| Context | Camera | Camera† | Rob.Calib† | Avg |
|---|---|---|---|---|
| Reactive (no history) | 90.0 | 74.0 | 55.2 | 73.1 |
| `O` only | 88.8 | 73.2 | 55.0 | **72.3** (reactive보다 나쁨) |
| `O, A` | 89.1 | 74.9 | 57.4 | 73.8 |
| **`O, A, O'`** | 95.7 | 76.3 | 61.3 | **77.8** |

**Table 3b — context 길이 (full `O,A,O'`)**
| K | Camera | Camera† | Rob.Calib† | Avg |
|---|---|---|---|---|
| 1 | 90.0 | 74.0 | 55.2 | 73.1 |
| 2 | 93.8 | 74.6 | 57.4 | 75.3 |
| 4 | 96.1 | 75.7 | 58.4 | 76.7 |
| 8 | 95.7 | 76.3 | 61.3 | 77.8 |

**Latency (Fig.4)**: K=1 124ms → K=4 1.19× → K=8 178ms (1.43×), 프롬프트는 8배.
K=8 상한 이유: "longer contexts exceed our GPU memory budget during block-causal training".

**Real-world (Agilex Piper, N=25/조건, Wilson 95% CI)**
| Task | | Reactive π0.5† | Reflective VLA |
|---|---|---|---|
| Box | seen | 88 [70.0, 95.8] | 92 [75.0, 97.8] |
| Box | **held-out** | 32 [17.2, 51.6] | **76 [56.6, 88.5]** (+44pp) |
| Bowl | seen | 68 [48.4, 82.8] | 72 [52.4, 85.7] |
| Bowl | **held-out** | 16 [6.4, 34.7] | **64 [44.5, 79.8]** (+48pp) |

held-out CI가 겹치지 않아 95% 수준에서 유의. seen은 겹침 → in-distribution 격차는 작다.
데모는 10개 카메라 배치에서 500개 수집, 평가는 seen 5 / held-out 5.

**LIBERO-Plus-Hard 사양 (Appendix B)**
- Camera†: 3인칭 azimuth `[-75°,75°]`, elevation `[0°,15°]`, distance scale `[1.0,2.0]`,
  endpoint rotation `[-10°,10°]`. 손목 카메라는 FOV `[60°,90°]` + {identity, h-flip, v-flip, 180° rot} 중 하나.
  **모든 뷰를 동시에 섭동**시켜 어떤 단일 뷰도 nominal calibration을 보존하지 않게 한다.
- Rob.Calib†: 에피소드 시작 시 고정 bias 샘플 → **모든 absolute EEF command에 적용**.
  축별 translation `[-10,10]`mm, axis-angle rotation ≤`3°`, gripper는 변경 없음.
  replay 후 **성공한 궤적만 수집**.
- 병합 학습셋 43,546 trajectories.

---

## 2. 구현 베이스 선택

### 결론: **LeRobot의 `pi05` 정책을 베이스로, 별도 플러그인 패키지로 구현**

근거:
1. LeRobot `src/lerobot/policies/pi05/` 는 논문 골격과 **정확히 일치**한다 —
   `PaliGemmaWithExpertModel` (PaliGemma prefix + Gemma-300m action expert, layer마다 attention 공유),
   flow-matching, `embed_prefix`/`embed_suffix`, 2D attention mask (`make_att_2d_masks`) 구조.
   우리가 바꿔야 할 건 **prefix 구성 + attention mask + suffix 개수** 세 곳뿐이다.
2. `make_att_2d_masks` 가 이미 cumsum 기반 **block-causal** 마스크다 (docstring에 "causal attention between blocks" 명시).
   triplet 블록 경계만 세팅하면 prefix 쪽 마스크는 그대로 재사용된다.
3. LeRobot에 **LIBERO / LIBERO-Plus env 통합이 이미 있다** (`src/lerobot/envs/libero.py`, `docs/source/libero_plus.mdx`,
   `docker/Dockerfile.benchmark.libero_plus`). 평가 파이프라인을 새로 짤 필요가 없다.
4. `get_policy_class` 가 **convention-based 플러그인 해석**이다 (`@PreTrainedConfig.register_subclass` 로 등록된
   서드파티 정책을 그대로 로드). → **lerobot 본체를 fork/패치할 필요 없음.**
5. `pi05/memory.py` (MEM) 에 이미 inference용 observation history FIFO 큐 (`sample_observation_history`,
   `_stack_inference_memory`, `_memory_queues`) 가 있어 triplet buffer 구현의 참고/재사용이 가능하다.
6. 로컬에 이미 LeRobot v3.0 포맷 LIBERO 데이터(`/workspace/yujin/data/libero_exp02/lerobot_v30/`)가 있다.

대안 검토:
- **openpi** (Physical Intelligence 원본): JAX 중심, 커스텀 attention mask 삽입이 더 번거로움. 기각.
- **DynamicVLA/OpenVLA**: discrete action token + autoregressive → flow-matching action expert가 없어 논문 구조와 불일치. 기각.

### 백본
`lerobot/pi05_base` 체크포인트 (PaliGemma-3B). Qwen3-VL-2B 변형은 v2 이후 옵션.

---

## 3. 리포 구조 (현재 상태)

```
/workspace/yujin/reflectiveVLA/
├── PLAN.md                                   # 이 문서
├── README.md
├── pyproject.toml                            # reflective-vla 패키지 (lerobot 의존)
├── third_party/lerobot/                      # lerobot 0.6.2, editable install, 본체 무수정
├── src/reflective_vla/
│   ├── __init__.py
│   ├── masking.py                    [완료]  # 토큰 레이아웃 + block-causal mask (Eq.6) + position ids
│   ├── tokenization.py               [완료]  # g_A / proprio / τ / boundary projector + prefix 조립
│   ├── configuration_reflective_vla.py [완료] # 논문 하이퍼파라미터 기본값, Table 4 preset
│   ├── modeling_reflective_vla.py    [완료]  # ReflectiveVLAPytorch + ReflectiveVLAPolicy
│   ├── buffer.py                     [완료]  # 추론용 triplet FIFO 버퍼
│   └── processor_reflective_vla.py   [완료]  # state를 prompt에서 빼고 projector로, 파생키 정규화
├── tests/
│   ├── conftest.py                   [완료]  # tiny 백본 스텁 (실제 attention + KV cache)
│   ├── test_masking.py               [완료]  # 21개: 레이아웃/마스크/position
│   ├── test_model_leakage.py         [완료]  # 10개: 모델 레벨 leakage
│   └── test_inference.py             [완료]  # 6개: KV캐시 동등성 + rollout
└── (미구현, 학습 단계에서) data/ envs/ scripts/ configs/
```

환경: conda env `reflective` (Python 3.12, torch 2.11+cu130, transformers 5.5.4, lerobot 0.6.2).
기존 `fastwam` env와 실행 중인 GPU 작업은 건드리지 않음.

**lerobot 무수정 확인**: `get_policy_class("reflective_vla")` 가 우리 클래스를 그대로 해석한다
(draccus `register_subclass` 기반 convention 해석). fork/패치 불필요.

## 4. 핵심 구현 상세

### 4.1 토큰 레이아웃과 마스크 (`masking.py`) — 가장 중요한 부분

프레임 `t_1 < t_2 < ... < t_K` 에 대해 prefix 블록:

```
k = 1..K-1 (full triplet):
  B_k = [ LANG(n_l) | TAU_k(1) | IMG3rd(O_k)(256) | IMGwrist(O_k)(256) | STATE(O_k)(1)
        ‖ ACT_k(8) | IMG3rd(O'_k)(256) | IMGwrist(O'_k)(256) | STATE(O'_k)(1) ]
k = K (현재 프레임):
  B_K = [ LANG | TAU_K | IMG3rd(O_K) | IMGwrist(O_K) | STATE(O_K) ]
```
`‖` 앞을 **P_k (observable part)**, 뒤를 **E_k (evidence part)** 로 태깅한다.

Suffix (action expert): `Q_1(C) | Q_2(C) | ... | Q_K(C)` — 프레임당 C개 noisy action token.

**마스크 규칙** (토큰마다 `block_id ∈ [0,K)` + `is_evidence ∈ {0,1}` 두 개의 태그만 있으면 전부 벡터화 가능):

| from → to | 규칙 |
|---|---|
| prefix → prefix | `block_id[j] <= block_id[i]` (블록 내부는 bidirectional, 블록 간 causal) |
| suffix `Q_k` → prefix `j` | `block_id[j] < k` **OR** (`block_id[j] == k` **AND** `not is_evidence[j]`) |
| suffix → suffix | 같은 query slot 내부만 (`q_idx[i] == q_idx[j]`) |
| prefix → suffix | 전부 False |

→ 이게 논문 Eq.6 `V_{t_k}` 를 정확히 구현한다. 자기 A_k / O'_k 차단, 이후 triplet 차단, 형제 query 상호 차단 모두 커버.

**position_ids**: prefix는 `cumsum(pad_mask)-1` (기존 lerobot과 동일). suffix `Q_k` 는
**블록 B_k 의 P_k 끝 위치 + 1 .. +C** 를 부여한다. 이렇게 하면 학습 시 query가 자기 프레임 바로 뒤에 붙고,
inference(K=1 query가 prefix 맨 끝에 붙는 상황)와 **동일한 상대 위치**가 되어 train/test mismatch가 없다.

**구현 방식**: `make_att_2d_masks` 를 대체하는 `build_reflective_masks(block_ids, is_evidence, pad_mask, K, C)`
→ bool `[B, T, T]` 반환 → 기존 `prepare_attention_masks_4d` 에 그대로 투입.

### 4.2 모델 (`modeling_reflective_vla.py`)

`PI05Pytorch` 를 상속하고 세 메서드만 오버라이드:
- `embed_prefix(...)`: 프레임 축을 flatten해서 K개 블록 임베딩 + `block_ids`/`is_evidence` 태그 생성.
  - 과거 action chunk → `action_history_proj` (2-layer MLP, `[B,K-1,C,action_dim] → [B,K-1,8,width]`) = 논문 `g_A`.
  - `τ` → sinusoidal(현재 프레임 기준 상대 step) → 1 토큰 projector.
  - 이미지: 기존 `paligemma_with_expert.embed_image` 를 `(B*num_frames)` 로 flatten 호출.
- `embed_suffix(...)`: K개 query slot을 만든다 (`[B, K*C, width]`). flow-matching time은 프레임마다
  독립 샘플링 (batch 내 다양성 ↑, 논문에 명시 없음 → config로 공유/독립 토글).
- `forward(...)`: loss를 `[B,K,C,dim]` 로 받아 valid target에만 평균 → `L_train = Σ_k L_act`.

`ReflectiveVLAPolicy`:
- `reset()`: triplet FIFO buffer + KV cache 초기화 (rollout 경계).
- `select_action()`: chunk 실행이 끝나면 `(O_i, A_i, O'_i)` 를 buffer에 push → 다음 chunk 예측.

### 4.3 Inference buffer & KV cache (`buffer.py`)
- `deque(maxlen=K-1)` 로 완료된 triplet 보관. 에피소드 초반 미충족분은 **pad-mask 처리** (학습 시에도 동일하게 취급).
- **position id는 에피소드 내내 단조 증가**시킨다 (evict해도 남은 블록의 절대 position 유지).
  RoPE는 `q·k` 가 `p_i - p_j` 에만 의존하므로 절대값이 커져도 **상대 offset은 학습 분포 그대로**다 → 캐시 무효화 없음.
- v1은 캐시 없이(매 step prefix 재계산) 정확성부터 확보 → `test_kv_cache.py` 로 동등성 확인 후 캐시 경로 활성화.

### 4.4 데이터 (`data/triplet_dataset.py`)
`LeRobotDataset` 을 감싸는 wrapper. 앵커 프레임 `l_t` (에피소드 내 로컬 인덱스) 하나당:
1. gap `g_1..g_{K-1} ~ U{0..15}` 샘플 (eval은 0).
2. `l_K = l_t`, `l_k = l_{k+1} - (C + g_k)` 로 역방향 인덱스 생성.
3. `l_k < 0` 인 triplet은 **pad-mask=0** 로 마스킹 (에피소드 시작부 = inference 초반 buffer 미충족과 동일 조건).
4. 프레임 k마다: obs(3rd/wrist/state) @ `l_k`, action chunk `actions[l_k : l_k+C]`,
   consequence obs @ `min(l_k+C, ep_end)`.
5. target: **모든 k의 action chunk** (에피소드 끝 넘어가는 부분은 `action_is_pad`).

**IO 병목 주의**: 샘플당 이미지 = K(obs) + (K-1)(consequence) = 15 프레임 × 2 카메라 = **30 프레임 디코딩**.
현재 데이터가 h264 video (512×512)라 random seek 30회는 학습을 죽인다.
→ `build_frame_cache.py` 로 에피소드별 224×224 JPEG를 LMDB/memmap에 미리 풀어둔다 (LIBERO 전체 ~10GB 수준 추정).

### 4.5 Action space
논문은 **absolute EEF control** 로 변환한다. LIBERO 기본은 delta EEF.
- `convert_absolute_ee.py`: 데모를 replay하며 EEF pose를 적분 → absolute target 생성.
- 단, 이건 재현 충실도용이고 **핵심 주장(consequence context)과는 직교**하다.
  → **v1은 LIBERO 네이티브 delta action으로 진행**하고 (reactive baseline도 동일 조건이므로 비교는 공정),
    absolute 변환은 M6에서 추가. config 플래그로 전환.

### 4.6 베이스라인 (논문과 matched)
동일 백본 / projector / action expert / optimizer / 데이터, `K` 와 context 구성만 다르게:
- `reactive_libero.yaml`: `K=1` (= 사실상 pi05)
- `history_only_libero.yaml`: `context_mode="OA"` → prefix에서 O' 블록 제거
- `obs_only`: `context_mode="O"`
- `reflective_libero.yaml`: `context_mode="OAO"` (full)
→ Table 3a 재현이 그대로 나온다.

---

## 5. 재현 충실도 원칙

사용자 결정: **학습은 더 큰 GPU에서 진행**하므로 현재 장비의 메모리 제약에 맞춘 타협(LoRA, 토큰 pooling,
카메라 축소)은 **기본값에서 배제**한다. 구현은 논문을 그대로 따른다.

기본 설정 = 논문 값:
- `K=8`, `C=10`, PaliGemma-3B (`paligemma_variant="gemma_2b"`), action expert hidden **1024**
  (= lerobot `gemma_300m` variant의 width와 정확히 일치)
- 3rd-person + wrist 두 카메라, proprio는 **2-layer MLP projector 토큰** (π0.5의 prompt 이산화가 아님)
- 과거 action chunk → `g_A` 2-layer MLP → **8 토큰**
- triplet마다 instruction 재삽입, `τ` 토큰, frame-boundary 토큰 전부 포함
- **full fine-tuning**: action expert/projector `1e-4`, VLM `0.1×`(=1e-5), 초기 구간 VLM freeze 후
  linear warm-up + cosine decay. AdamW β=(0.9,0.95), wd 0.01, clip 1.0, bf16, DeepSpeed ZeRO-2
- 시작 체크포인트: **PaliGemma-3B (VLM만 pretrained)**, action expert와 신규 projector는 scratch — 논문 그대로.
  `lerobot/pi05_base`(로봇 사전학습 π0.5)에서 시작하는 경로도 옵션으로 남기되 기본값 아님.

참고용 시퀀스 길이 (K=8, 2 camera, 224px → SigLIP 256 tok/img):
triplet 블록 ≈ 경계토큰(4) + lang + τ(1) + obs(513) + act(8) + cons(513) ≈ **1090 tok**,
prefix ≈ **8.2k tok**, suffix = K×C = 80 tok. 논문의 8×H20 + ZeRO-2 규모가 필요하다.

효율 옵션(`history_image_pool`, consequence 단일 카메라 등)은 config 플래그로 존재하되 **기본 off**,
디버깅/스모크 테스트 용도로만 쓴다.

## 6. 마일스톤 / 진행 상황

| M | 내용 | 상태 |
|---|---|---|
| **M0** | conda env `reflective` (py3.12) + lerobot editable + 패키지 스캐폴드 | ✅ 완료 |
| **M1** | config + policy 등록, `K=1` reactive 경로 | ✅ 완료 (`get_policy_class` 해석 확인, K=1 테스트 통과) |
| **M2** | `masking.py` + prefix 조립 (τ, g_A, boundary token) | ✅ 완료 — **leakage 테스트 통과** |
| **M3** | 추론 triplet FIFO 버퍼 + KV 캐시 경로 | ✅ 완료 — cached ≡ joint forward 동등성 확인 |
| **M4** | triplet dataset (random gap 샘플링) + frame cache + collate | ⬜ 미착수 (학습 단계) |
| **M5** | 학습 스크립트 (param group, VLM freeze 스케줄, ZeRO-2) | ⬜ 미착수 |
| **M6** | LIBERO / LIBERO-Plus / Hard 평가 연결, absolute EEF 변환 | ⬜ 미착수 |
| **M7** | context 구성 ablation (O / O,A / O,A,O') 실험 | ⬜ config는 준비됨, 실험 미착수 |

### 통과한 검증 (총 37개)
- **leakage**: query k가 자기 `A_k` / `O'_k` 를 절대 못 봄. `T_{j<k}` 는 전부 보이고 `T_{j>k}` 는 전혀 안 보임.
  형제 query slot 상호 차단. prefix→suffix 역방향 차단.
- **K=1 reactive 동등성**: evidence 없음 → suffix가 prefix 전체를 봄 = π0.5 규칙. position id도 π0.5 공식과 일치.
- **KV 캐시 동등성**: prefill+cached denoise 결과 == 단일 joint forward 결과 (atol 1e-5).
- **masked frame**: 에피소드 초반 미충족 슬롯을 교란해도 이후 query 출력 불변.
- **rollout**: chunk 실행 → consequence 관측 시점에 triplet close, capacity K-1에서 FIFO evict,
  `τ` 가 실제 실행 stride의 배수로 기록됨, `reset()` 이 에피소드 간 컨텍스트 차단.

## 7. 논문 미명시 항목 — 우리 구현의 결정 (전부 config 노출)

| 항목 | 논문 | 구현 결정 | config |
|---|---|---|---|
| `τ` 인코딩 | 존재만 언급, 방식 미명시 | 현재 프레임 기준 **backward env-step offset**의 sinusoidal → 2-layer MLP → 1 토큰. 랜덤 stride를 모델이 구분할 수 있어야 하므로 상대 오프셋이 자연스럽다 | `use_temporal_index` |
| 블록 내 토큰 순서 | 템플릿은 instruction-first (Qwen chat 형식) | **image-first** (PaliGemma 네이티브 = π0.5 prefix 순서). 백본 사전학습 분포와 일치시키기 위함 | `language_first=False` |
| 이미지당 토큰 수 | 미명시 | SigLIP 224/14 → 256 (PaliGemma 기본) | — |
| flow matching inference steps | 미명시 | π0.5 기본값 10 | `num_inference_steps` |
| 프레임별 flow time | 미명시 | **프레임마다 독립 샘플**. `PiGemmaRMSNorm` 이 per-token AdaRMS cond를 지원해서 가능 | — |
| 에피소드 초반 buffer 미충족 | "conditions on whatever context is available" | 앞쪽 슬롯을 **pad + mask**. 학습의 `l_k<0` 처리와 동일 규칙 → train/deploy 동일 표현 | `frame_mask` |
| 블록 내부 attention | 미명시 | **양방향** (완성된 triplet을 한 단위로 캐싱하는 방식과 맞음). `P_k`가 `E_k`를 못 보게 하는 sub-block causal도 옵션 제공 | `intra_block_causal` |
| VLM freeze step 수 | "initial period" 라고만 | `scheduler_warmup_steps` 로 기본 대체 | `vlm_freeze_steps` |
| tokenizer max length | 미명시 | 48. state가 prompt에서 빠져 instruction만 남고, 그게 K번 반복되므로 π0.5 기본값 200은 대부분 padding | `tokenizer_max_length` |
| action normalization | 미명시 | π0.5 기본 QUANTILES. context action은 **모델 출력과 같은 정규화 공간**이어야 한다 (배포 시 버퍼가 policy 자신의 예측으로 채워지므로) | `normalization_mapping` |

## 8. 파인튜닝 방식 — 논문 원문 확인 결과 (Appendix A.1 / C.1 / E)

전문 검색 결과 **`LoRA` / `adapter` / `PEFT` 는 논문에 단 한 번도 등장하지 않는다.**

### 확정된 사실
| 질문 | 답 | 근거 (verbatim) |
|---|---|---|
| full FT vs LoRA | **full fine-tuning** | "trained with AdamW (…) under bf16 mixed precision with **DeepSpeed ZeRO-2**" — LoRA면 optimizer state 샤딩이 불필요 |
| 파라미터 그룹 | expert+projector `1e-4`, VLM `0.1×` = `1e-5` | "The action expert and projectors use a peak learning rate of 10⁻⁴; the VLM uses a 0.1× multiplier" |
| VLM freeze | 초기 구간 frozen → 이후 동일 linear warm-up + cosine decay | "is held frozen for the initial period in Table 4, after which it follows the same linear warm-up and cosine decay schedule" |
| freeze step 수 | **논문 미기재** | Table 4에는 Steps / Batch-per-GPU / K 컬럼뿐, frozen 구간 컬럼이 없다 (서술 누락) |
| 시작 체크포인트 | **pretrained VLM만** (PaliGemma-3B 또는 Qwen3-VL-2B, official release) | A.1 "a **pretrained VLM** encodes…"; E "Pretrained vision-language backbones are obtained from their official releases" |
| π0.5 로봇 사전학습 가중치 로드 | **언급 없음** | "π0.5†"는 *reproduced* reactive baseline = 아키텍처 재현물이지 PI 체크포인트가 아님 |
| 로봇 데이터 대규모 사전학습(OXE 등) | **없음** | 학습 데이터는 LIBERO / LIBERO-Plus(+Hard) / BridgeData V2 뿐 |
| action expert 초기화 | scratch (새로 붙는 모듈, hidden 1024) | A.1 "an action expert with hidden dimension 1024" |
| 신규 projector(g_A, proprio, τ) 초기화 | **미명시** → random init 가정 | "lightweight two-layer nonlinear fully connected projectors" |
| 다단계/curriculum | **없음**, 단일 phase | reactive→reflective 순차 학습 언급 없음 |

정리하면: **VLM만 pretrained, VLA로서는 로봇 데이터로 처음부터 학습**하며 전체 파라미터를 full FT 한다.

### 우리 구현의 재현 방침
학습은 더 큰 GPU에서 수행할 예정이므로 **논문 설정을 그대로 구현**한다.

| 항목 | 논문 | 우리 |
|---|---|---|
| FT 방식 | full FT (DeepSpeed ZeRO-2) | **동일** — full FT 경로가 기본. LoRA는 옵션 플래그로만 존재 |
| 시작 체크포인트 | PaliGemma-3B (VLM만 pretrained), expert scratch | **동일**. `lerobot/pi05_base` 초기화는 선택 옵션 |
| LR 그룹 | expert/projector 1e-4, VLM 0.1× | **동일** (`vlm_lr_multiplier=0.1`) |
| VLM freeze 구간 | "initial period" (**step 수 미기재**) | `vlm_freeze_steps` 로 노출, 기본값 = warmup step 수 (미기재분에 대한 우리 해석) |
| batch / steps | Table 4 그대로 | config preset으로 제공 |

### 구현에 반영할 추가 사항
- **frame-boundary token**: `<|frame_start|>` `<|frame_end|>` `<|frame_sep|>` `<|action_start|>` `<|action_end|>` 를
  **학습 가능한 임베딩**으로 레이아웃에 삽입한다. Limitations에서 shortcut 완화 수단으로 명시했으므로 기능 요소다.
- **stride randomization**: 데이터 파이프라인의 random gap `U[0,15]` 은 성능 튜닝이 아니라 shortcut 방지 장치다. 기본 on.
