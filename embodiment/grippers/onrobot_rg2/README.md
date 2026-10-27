# OnRobot RG2 그리퍼 (robosuite / LIBERO embodiment-swap용) 준비 가이드

이 폴더는 **원래 robosuite·LIBERO에 없던 OnRobot RG2 그리퍼**를 인터넷 공개 자료로 만들어서,
embodiment-swap 데이터 생성 파이프라인(`embodiment_swap_v2/`)에 끼워 넣은 결과물입니다.
다른 서버에서 **이 문서만 보고 똑같이 재현**할 수 있도록 출처 → 변형 내용(이유 포함) → 코드 연결 → 검증 순서로 정리했습니다.

- 작성: 2026-10-26
- 사용처: `UR5e + RG2 / libero_goal task 1, 7` 데이터 (`/workspace/yujin/data/libero_embodiment_gripper/`)

---

## 0. 한눈에 보기

| 항목 | 값 |
|---|---|
| robosuite 등록 이름 | `OnRobotRG2Gripper` (`embodiment_swap_v2/rg2_gripper.py`) |
| MJCF | `grippers/onrobot_rg2/rg2_robosuite.xml` |
| 메시 | `grippers/onrobot_rg2/assets/*.stl` (visual 4개 + collision 4개, 실제 로드는 visual 4 + base collision 1) |
| 구조 | 평행 4-bar 링크, 1-DoF (왼쪽 driver 관절에 position servo, 나머지는 equality로 연동) |
| 최대 벌림 (패드 면 사이) | 82.7 mm (실제 RG2 110 mm, 2F-85 85 mm, Panda 80 mm) |
| 닫힘 | driver 0.78 rad에서 패드 면이 맞닿음 |
| 손끝 전진량 (열림→닫힘) | 21.0 mm (4-bar라 닫히면서 손끝이 앞으로 나감 → `--tip_comp`로 보정) |
| grip_site | 플랜지에서 0.2084 m, 열린 손끝이 grip_site보다 4.4 mm 앞 (Panda·2F-85와 동일) |
| 필요 환경 | robosuite 1.4.1, mujoco 3.3.2, LIBERO (`Lifelong-Robot-Learning/LIBERO` @ `8f1084e`) |

폴더 구성:
```
grippers/onrobot_rg2/
├── README.md              ← 이 문서
├── LICENSE                ← 두 원본 레포의 MIT 저작권 표기
├── rg2_robosuite.xml      ← robosuite용 MJCF (이 문서 3장의 변형이 모두 반영된 최종본)
├── check_rg2.py           ← 검증 스크립트 (5장)
└── assets/
    ├── base_link_vis.stl  base_link_col.stl
    ├── outer_knuckle_vis.stl  outer_knuckle_col.stl
    ├── inner_knuckle_vis.stl  inner_knuckle_col.stl
    └── inner_finger_vis.stl   inner_finger_col.stl
```

---

## 1. 배경: 왜 직접 만들어야 했나

- robosuite 1.4.1 기본 그리퍼: Panda, Rethink, Robotiq 85/140, Robotiq 3-finger, Jaco, Wiping 뿐. **RG2 없음.**
- 처음 참고하려던 [JiyunJang-24/UR5e_simulator](https://github.com/JiyunJang-24/UR5e_simulator)에는 **RG2 시뮬레이션 에셋이 없음.**
  RG2는 실로봇 제어 submodule([onrobot-rg](https://github.com/JiyunJang-24/onrobot-rg), 원본 [takuya-ki/onrobot-rg](https://github.com/takuya-ki/onrobot-rg))에서
  Modbus 통신으로만 쓰이고(메시 없음), 시뮬레이션은 Robotiq 2F-85를 씀.
- 그래서 GitHub 검색(`rg2 mujoco`, `rg2 urdf`, `onrobot description`)으로 후보를 모아 비교함.

| 후보 | 형식 | 판단 |
|---|---|---|
| **[John-A-Chen/onrobot-rg2-mujoco](https://github.com/John-A-Chen/onrobot-rg2-mujoco)** | MuJoCo MJCF (4-bar를 joint equality로 구현) | **채택 (링크 구조)**. MuJoCo에 바로 로드되고 2F-85 Menagerie와 같은 방식이라 기존 파이프라인과 잘 맞음 |
| **[tonydle/OnRobot_ROS2_Description](https://github.com/tonydle/OnRobot_ROS2_Description)** | ROS2 URDF/xacro + STL | **채택 (메시 + 치수 대조)**. 위 MJCF가 이 메시를 submodule로 씀 |
| [UOsaka-Harada-Laboratory/onrobot](https://github.com/UOsaka-Harada-Laboratory/onrobot) | ROS1 URDF | 미사용 (같은 계열 메시, MJCF 아님) |
| [ekorudiawan/rg2_simulation](https://github.com/ekorudiawan/rg2_simulation) | Gazebo URDF | 미사용 (직접 변환 필요) |
| [AndrejOrsula/ur5_rg2_ign](https://github.com/AndrejOrsula/ur5_rg2_ign) | Ignition URDF/SDF | 미사용 (손가락이 관절 하나로 단순화) |

추가로 참고한 것:
- [OnRobot RG2 공식 자료](https://learn.onrobot.com/en/material/84?area=13): 스트로크 0–110 mm, 파지력 3–40 N, 무게 0.65 kg
- [MuJoCo Menagerie Robotiq 2F-85](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotiq_2f85) 및 그걸 robosuite용으로 바꾼 로컬 파일
  `embodiment_swap/grippers/robotiq_2f85_menagerie_orig/2f85_menagerie_localfinger_nolinkcol_robosuite.xml`
  (이전 Robotiq 실험에서 쓴 파일 — **site 배치·패드·solver 값·충돌 규칙을 이 파일과 똑같은 규약으로 맞춤**)

라이선스: 두 채택 레포 모두 MIT. 저작권 문구는 `LICENSE`에 같이 넣어 둠.

---

## 2. 원본 받기

```bash
mkdir -p ~/rg2_src && cd ~/rg2_src
git clone https://github.com/John-A-Chen/onrobot-rg2-mujoco.git && (cd onrobot-rg2-mujoco && git checkout 2cd1bf7)
git clone https://github.com/tonydle/OnRobot_ROS2_Description.git
```
- 우리가 쓴 원본 파일: `onrobot-rg2-mujoco/rg2.xml` (그리퍼 단독 MJCF, 액추에이터 없음)
  - 같은 레포의 `scene.xml`에만 그리퍼 액추에이터가 있음: `position joint="left_driver_joint" kp=20 kv=2 ctrlrange="0 0.8"` (토크 제한 없음)
- 메시: `OnRobot_ROS2_Description/meshes/rg2/{visual,collision}/{base_link,outer_knuckle,inner_knuckle,inner_finger}.stl`
- 치수 대조용: `OnRobot_ROS2_Description/urdf/rg2_macro.xacro`

메시 복사 (이름만 `_vis` / `_col`로 바꿈, 내용 그대로):
```bash
G=embodiment_swap_v2/grippers/onrobot_rg2; mkdir -p $G/assets
for p in base_link outer_knuckle inner_knuckle inner_finger; do
  cp OnRobot_ROS2_Description/meshes/rg2/visual/$p.stl    $G/assets/${p}_vis.stl
  cp OnRobot_ROS2_Description/meshes/rg2/collision/$p.stl $G/assets/${p}_col.stl
done
```

원본 확인 결과 (mujoco 3.3.2에서 그대로 로드 성공):
- 키네마틱스가 URDF와 일치: driver 각도 θ에 따른 손가락 면 간격 — θ=−0.6: 104.9 mm, θ=0: 75.6 mm, θ=0.4: 39.7 mm, θ≈0.78: 0 mm
  (URDF `finger_joint` 닫힘 한계 0.785 rad와 일치)
- 간이 씬(박스 4×4×5 cm, 0.1 kg)에서 닫기 → 10 cm 들기 → 놓기 성공

---

## 3. 원본 → 우리 버전: 변형한 것 전부 (이유 포함)

**바꾸지 않은 것**: 링크 형상(body 위치/방향), 관절 축, 질량·관성, 메시 내용, 4-bar 연동 관계(계수).
최종 결과는 `rg2_robosuite.xml` 한 파일에 모두 반영되어 있으니, 재현할 때는 이 파일을 그대로 쓰면 됩니다.
아래는 "원본 `rg2.xml`에서 무엇을 왜 바꿨는지"입니다.

### 3-1. robosuite 그리퍼 규약 맞추기

robosuite `GripperModel`은 그리퍼 XML 안에 특정 이름의 site/센서가 있어야 팔에 붙이고 제어할 수 있습니다.
기존 2F-85 robosuite 파일과 **완전히 같은 구조**로 넣었습니다.

```xml
<body name="base_mount" pos="0 0 0">            <!-- 팔 플랜지에 붙는 루트. RG2 base_link 원점 = 플랜지 (URDF도 tool0에 offset 0으로 붙음) -->
  <site name="ft_frame" .../>                    <!-- 힘/토크 센서 위치 -->
  <body name="eef" pos="0 0 0.2084" quat="0.707105 0 0 -0.707105">
    <site name="grip_site" .../>                 <!-- OSC 제어·IK·state가 기준으로 쓰는 점 -->
    <site name="ee_x" .../> <site name="ee_y" .../> <site name="ee_z" .../> <site name="grip_site_cylinder" .../>
  </body>
  <body name="base" pos="0 0 0"> ... 원본 rg2_base_link 내용 ... </body>
</body>
...
<sensor> <force name="force_ee" site="ft_frame"/> <torque name="torque_ee" site="ft_frame"/> </sensor>
```

- **eef 방향 `quat="0.707105 0 0 -0.707105"` (z축 −90°)**: RG2도 2F-85도 손가락이 base 기준 y축 방향으로 벌어짐.
  2F-85 파일과 같은 회전을 줘서 **손가락 벌어지는 방향 = grip_site x축**으로 통일.
  (팔이 바뀌어도 grip_site 자세를 Panda에 IK로 맞추기 때문에, 이 규약이 같아야 손가락 방향도 Panda와 같아짐)
- **grip_site 높이 0.2084 m**: 원본 `gripper_tcp`(0.218 m)는 쓰지 않음. 대신 **"열린 상태에서 손끝(충돌 패드 끝)이 grip_site보다 4.4 mm 앞"**이 되도록 맞춤.
  Panda와 2F-85 둘 다 정확히 +4.4 mm라서 같은 기준을 적용한 것.
  - 처음엔 0.200 m로 했다가 측정해 보니 +12.8 mm여서 8.4 mm 올림 → 0.2084 m. (측정은 `check_rg2.py`의 `[tip]` 줄)
  - 이게 다르면 같은 Panda 궤적을 따라가도 손끝이 물체를 잡는 깊이가 그리퍼마다 달라짐.
- geom group을 robosuite 규약으로: visual = group 1 (`contype=0 conaffinity=0`), collision = group 0.
  (원본은 visual 1 / collision 3 클래스였음)

### 3-2. 최대 벌림 넓히기 (75.6 mm → 82.7 mm)

원본은 driver 범위가 `0 ~ 0.8`이라 θ=0(완전 열림)에서 손가락 면 간격이 75.6 mm 뿐 → Panda(80 mm)보다 좁아서
Panda 데모를 따라 물체에 접근할 때 손가락이 물체에 걸릴 수 있음. 열림 쪽으로 0.1 rad 확장:

| 관절 | 원본 range | 우리 range |
|---|---|---|
| `left/right_driver_joint` | `0 0.8` | `-0.1 0.8` |
| `left/right_inner_knuckle_joint` (= driver) | `0 0.8` | `-0.1 0.8` |
| `left/right_inner_finger_joint` (= −driver) | `-0.8 0` | `-0.8 0.1` |

→ θ=−0.1에서 82.7 mm (2F-85 85 mm / Panda 80 mm와 비슷한 수준). 실제 RG2(110 mm)처럼 하려면 −0.56까지 늘리면 됨 (약 103 mm).

### 3-3. 충돌 형상: 링크 메시 끄고, 손가락 안쪽 면에 box 패드

이전 실험의 `Robotiq85GripperMenagerieNoLinkCol`과 같은 방식("베이스와 패드만 충돌"):
- 원본의 knuckle/finger **충돌 메시는 전부 제거**. MuJoCo는 메시 충돌을 convex hull로 처리해서 휘어진 손가락은 접촉이 부정확하고,
  링크가 물체·장면에 걸리는 문제도 줄이기 위함. 베이스 충돌 메시(`base_link_col.stl`)만 유지.
- 대신 **손가락 안쪽의 평평한 면에 box 패드 2개씩** (손가락 frame 기준):
  ```xml
  <geom name="left_pad1" type="box" pos="0 0.015 0.0445" size="0.0095 0.004 0.0085" friction="0.8" solimp="0.95 0.99 0.001" solref="0.004 1" priority="1" .../>
  <geom name="left_pad2" type="box" pos="0 0.015 0.0275" size="0.0095 0.004 0.0085" friction="0.7" solimp="0.95 0.99 0.001" solref="0.004 1" priority="1" .../>
  ```
  - 위치 결정 방법: θ=0에서 inner finger 충돌 메시 꼭짓점을 분석 → 안쪽 면이 손가락 frame에서 **y=+0.019, z=0.019~0.053 구간이 평평**
    (플랜지 기준 z 0.183~0.217, 폭 x ±10 mm). 패드 바깥 면이 정확히 그 면(y=0.019)에 오도록 중심 y = 0.019 − 0.004 = 0.015.
    z 구간을 반씩 나눠 pad1(손끝 쪽) 중심 0.0445, pad2 중심 0.0275, 반높이 0.0085.
  - 패드 이름을 `*_pad1/2`로 한 이유: 파이프라인(`swap_core.py`)이 2F-85용으로 이미 `gripper0_{left,right}_pad1`을 찾아 손가락 간격·손끝 위치를 계산하므로, 이름만 맞추면 같은 코드가 그대로 동작.
  - 마찰 0.8/0.7, solimp/solref/priority는 2F-85 패드 값 그대로.
- contact exclude도 이에 맞춰 단순화: base↔모든 링크, 왼손가락↔오른손가락.

### 3-4. 링크 연동 제약(equality)에 solver 값 지정

원본의 5개 joint equality(오른 driver = 왼 driver, inner knuckle = driver, inner finger = −driver)는 계수 그대로 두고,
2F-85 Menagerie와 같은 `solimp="0.95 0.99 0.001" solref="0.005 1"`을 추가 (원본은 기본값).
물체를 세게 쥘 때 링크가 덜 휘도록 하기 위함 (원본 기본값에선 하중 시 연동 오차가 약 0.04 rad까지 났음).

### 3-5. 액추에이터 추가 (원본 `rg2.xml`에는 없음)

robosuite는 그리퍼 XML 안의 액추에이터를 찾아 `[-1, 1]` 그리퍼 명령을 `ctrlrange`로 선형 매핑함 (−1 → 하한 = 열림, +1 → 상한 = 닫힘).
```xml
<general name="fingers_actuator" joint="left_driver_joint" ctrlrange="-0.1 0.78" forcerange="-5 5"
         gainprm="50 0 0" biasprm="0 -50 -1" biastype="affine"/>   <!-- position servo: kp 50, kv 1, 토크 제한 ±5 Nm -->
```
- ctrlrange 하한 −0.1 = 3-2의 새 열림 한계, 상한 0.78 = 패드가 맞닿는 각도.
- 오른쪽 손가락은 equality로 따라감 (액추에이터는 왼쪽 driver 하나).
- **주의: kp/토크 값은 실제 RG2 사양으로 보정한 값이 아님.** 간이 파지 테스트에서 물체를 놓치지 않을 정도로 정함.
  손끝 힘으로 환산하면 대략 80–120 N 수준으로 실제 RG2(최대 40 N)보다 2–3배 셈.
  실제 사양에 맞추려면 `forcerange="-2 2"` 정도로 낮추면 됨 (현재 생성된 데이터는 ±5 Nm 기준).
  참고로 원본 `scene.xml`의 그리퍼 액추에이터(kp 20, 토크 제한 없음)도 실제 사양 보정값은 아님.

### 3-6. 기타
- 손가락 색 rgba 0.05 → 0.12 (렌더에서 형태가 보이도록, 외형만 변경)
- 시작 자세는 열린 상태 (Python 쪽 `init_qpos`, 4장)

---

## 4. robosuite 등록 + 파이프라인 연결

### 4-1. 그리퍼 클래스 (`embodiment_swap_v2/rg2_gripper.py`)

```python
import os
import numpy as np
from robosuite.models.grippers import GRIPPER_MAPPING
from robosuite.models.grippers.gripper_model import GripperModel

class OnRobotRG2Gripper(GripperModel):
    def __init__(self, idn=0):
        super().__init__(os.path.join(os.path.dirname(os.path.abspath(__file__)), "grippers/onrobot_rg2/rg2_robosuite.xml"), idn=idn)

    def format_action(self, action):          # 2F-85(Robotiq85GripperMenagerie*)와 동일: -1 열기 / +1 닫기, 매 substep speed만큼 이동
        assert len(action) == 1
        self.current_action = np.clip(self.current_action + self.speed * np.sign(action), -1.0, 1.0)
        return self.current_action

    @property
    def init_qpos(self):                      # XML 관절 순서: L driver, L inner finger, L inner knuckle, R driver, R inner finger, R inner knuckle (열림)
        return np.array([-0.1, 0.1, -0.1, -0.1, 0.1, -0.1])

    @property
    def speed(self): return 0.20              # 데이터 생성 시 --grip_speed 0.01로 덮어씀 (Panda와 같은 닫힘 속도)

    @property
    def dof(self): return 1

    @property
    def _important_geoms(self):
        return {"left_finger": ["left_pad1", "left_pad2"], "right_finger": ["right_pad1", "right_pad2"],
                "left_fingerpad": ["left_pad1", "left_pad2"], "right_fingerpad": ["right_pad1", "right_pad2"]}

GRIPPER_MAPPING["OnRobotRG2Gripper"] = OnRobotRG2Gripper
```
등록은 `common.py`에서 import 한 줄로: `import rg2_gripper  # noqa: F401  registers "OnRobotRG2Gripper"`
(이후 `ControlEnv(..., gripper_types='OnRobotRG2Gripper')`로 사용)

### 4-2. `swap_core.py` 수정 (2곳)

관절 이름(`gripper0_{left,right}_driver_joint`, hinge)과 패드 이름이 2F-85 Menagerie와 같아서
**기존 2F-85 코드 경로(`self.menagerie=True`)를 그대로 탐**: 손가락 간격(state의 손가락 2차원) = 패드 중심 거리 − 2×패드 반두께,
초기 그리퍼 열기, `tip_advance` 등. 다른 건 4-bar의 연동 관절 이름뿐이라 그 부분만 분기:

```python
# Swap.__init__ — self.menagerie 정의 바로 다음
self.rg2 = nid(dm, MJ.mjOBJ_JOINT, 'gripper0_left_inner_finger_joint') >= 0

# Swap._tip_table — 손끝 전진량 테이블 (닫힘 각도 → 손끝이 approach 축으로 나가는 거리)
same, neg = ('inner_knuckle', 'inner_finger') if self.rg2 else ('spring_link', 'follower')
self._tip_th = np.linspace(-0.1, 0.8, 91) if self.rg2 else np.linspace(0, 0.8, 81)   # RG2는 -0.1(열림)이 기준
for th in self._tip_th:
    for s_ in ('right', 'left'):
        d.qpos[adr(s_, 'driver')] = th; d.qpos[adr(s_, same)] = th; d.qpos[adr(s_, neg)] = -th
    ...
```

### 4-3. 데이터 생성 플래그 (2F-85와 동일)

```bash
python render_emb_shard.py --suite libero_goal --task 1 --robot UR5e --demo-start 0 --demo-end 25 \
  --gripper OnRobotRG2Gripper --menagerie_opt --tip_comp --grip_speed 0.01 --modes goal --out <shards>/UR5e_RG2_libero_goal
```
- `--menagerie_opt`: impratio=10, elliptic cone (링크 그리퍼 파지 안정화)
- `--tip_comp`: 닫히면서 손끝이 나가는 만큼(최대 21 mm) 목표를 뒤로 당겨서, grip_site가 아니라 **손끝이 Panda를 따라가게** 함
- `--grip_speed 0.01`: 닫힘 속도를 Panda와 맞춤
- 전체 실행: `embodiment_swap_v2/run_gripper_shards.sh` → `chain_pack.sh` (선택 + LeRobot v3.0 변환)
- eval 환경에서도 같은 설정이어야 함 (기존 2F-85와 같은 방식: `EMB_GRIPPER=OnRobotRG2Gripper EMB_MENAGERIE_OPT=1 EMB_GRIP_SPEED=0.01`)

---

## 5. 검증

```bash
cd embodiment_swap_v2 && MUJOCO_GL=egl python grippers/onrobot_rg2/check_rg2.py
```
기대 출력 (2026-10-26, 이 서버):
```
[tip] Panda+default: finger colliders -11.6 .. +4.4 mm along approach axis from grip_site
[tip] UR5e+Robotiq85GripperMenagerieNoLinkCol: finger colliders -33.1 .. +4.4 mm along approach axis from grip_site
[tip] UR5e+OnRobotRG2Gripper: finger colliders -29.6 .. +4.4 mm along approach axis from grip_site
[swap] panda_gripper False menagerie(linkage) True rg2 True tip-advance table max 21.0 mm
[open  ] driver -0.098 rad  gap  82.7 mm  tip advance  0.1 mm  mimic err 0.0000 rad
[close ] driver +0.780 rad  gap  -1.1 mm  tip advance 21.0 mm  mimic err 0.0000 rad
[reopen] driver -0.098 rad  gap  82.7 mm  tip advance  0.1 mm  mimic err 0.0000 rad
```
확인 포인트:
1. `[tip]` 세 줄의 손끝(+값)이 모두 **+4.4 mm** → grip_site 규약 일치
2. `rg2 True`, `menagerie(linkage) True` → 파이프라인이 RG2를 링크 그리퍼로 인식
3. 열림 약 83 mm / 닫힘 약 0 mm (빈 상태로 닫으면 패드가 살짝 겹쳐 −1 mm), 다시 열림 → 개폐 정상
4. mimic err ≈ 0 → 4-bar 연동 정상

실제 데이터 생성 결과 (UR5e + RG2, 50개 데모 중 성공):
- libero_goal task 1 (put the bowl on the stove): **50/50**
- libero_goal task 7 (turn on the stove): **47/50**
- 추종 오차(평균/최대) 약 12–15 / 26–34 mm. 이전 UR5e+2F-85 데이터(평균 10.6 mm, 최대 평균 24.3 mm)와 비슷한 수준

---

## 6. 다른 서버로 옮길 때 체크리스트

1. 환경: robosuite **1.4.1**, mujoco **3.3.2**, LIBERO (`8f1084e`), `MUJOCO_GL=egl`
   - RG2 자체는 이 서버 robosuite에 들어간 로컬 패치(`Robotiq85Gripper` 이름 교체)와 무관함
2. 복사할 것 (`embodiment_swap_v2/` 기준):
   - `grippers/onrobot_rg2/` 폴더 통째로 (XML이 `assets/*.stl`을 상대 경로로 읽음)
   - `rg2_gripper.py` + `common.py`의 import 한 줄
   - `swap_core.py`의 4-2 수정 2곳 (embodiment-swap 파이프라인을 쓸 때만 필요. 그리퍼만 쓸 거면 1–2번으로 충분)
3. `check_rg2.py`를 돌려서 5장 기대 출력과 같은지 확인
4. 조정하고 싶을 수 있는 값:
   - 파지 강도: `fingers_actuator`의 `forcerange` (현재 ±5 Nm, 실제 RG2에 가깝게 하려면 ±2 Nm)
   - 최대 벌림: driver / inner_knuckle / inner_finger의 range + actuator `ctrlrange` 하한, 그리고 `rg2_gripper.py`의 `init_qpos`,
     `swap_core.py`의 `_tip_th` 시작값을 **같이** 바꿀 것
   - grip_site 높이를 바꾸면 `[tip]`이 +4.4 mm가 되는지 다시 확인
