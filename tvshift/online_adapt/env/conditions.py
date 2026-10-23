"""Domain variations applied to a native LIBERO env.

Runs inside the LIBERO interpreter (``/opt/conda/envs/vera``), never inside the
trainer, so it must not import torch or anything from ``wan_va``.

Two conditions are supported, matching the two 4-GPU experiment arms:

``camera``
    Rotate the ``agentview`` camera by ``--camera-azimuth-degrees``. ``Pose``,
    ``rotate_camera_based_on_robot_base`` and ``reposition_camera`` below are
    copied verbatim from ``JiyunJang-24/LIBERO``
    ``xyg_scripts/rotate_recolor_dataset.py``; ``apply_camera_condition`` is only
    the adapter that feeds them our env and reports what they did.

``embodiment``
    Swap LIBERO's ``MountedPanda`` for a ``MountedUR5e`` carrying the Franka
    (``PandaGripper``) end effector.
"""

from __future__ import annotations

import math
import os
import pathlib
import re
from typing import Any

import numpy as np
import transforms3d.euler as euler
import transforms3d.quaternions as quaternions
import transforms3d.quaternions as quaternions
import transforms3d.euler as euler
import transforms3d.axangles as axangles

CONDITIONS = ("none", "camera", "embodiment")

CAMERA_AZIMUTH_DEGREES = 15.0

CAMERA_PIVOT_BODY = "robot0_link0"
# MuJoCo stores camera/body quaternions as wxyz, which is what the reference's
# Pose algebra expects; the flag exists because the reference carries it.
MUJOCO_WXYZ = True

# ``rotate_camera``'s own default: leave the camera where the rotation put it.
CAMERA_REPOSITION_SCALE = 1.0

# Only read by the copied ``debug`` branches, which this module never enables.
IMAGE_RESOLUTION = 512
IMAGE_SAVE_PATH = os.environ.get("LIBERO_ROTATE_DEBUG_DIR", "/tmp/libero_rotate_debug")
# LIBERO prepends a family-specific prefix to the robot name it is handed.
UR5E_ENV_ROBOT_ARG = "UR5e"
UR5E_MOUNTED_NAME = "MountedUR5e"
UR5E_GROUND_NAME = "OnTheGroundUR5e"
UR5E_GRIPPER = "PandaGripper"

# LIBERO's Panda init states carry 7 arm joints + 2 Panda finger joints before
# the object qpos block. UR5e has 6 arm joints and keeps the 2 Panda fingers.
PANDA_ARM_DOF = 7
PANDA_GRIPPER_DOF = 2


# --------------------------------------------------------------------------- #
# embodiment
# --------------------------------------------------------------------------- #
def register_ur5e_variants() -> None:
    """Register LIBERO-style UR5e robots with robosuite.

    LIBERO prefixes the robot name per problem family and only ships the Panda
    variants: ``libero_tabletop`` / ``kitchen`` / ``study`` ask for
    ``Mounted<name>``, while ``libero_floor`` / ``living_room`` / ``coffee_table``
    ask for ``OnTheGround<name>``. LIBERO-Long spans both, so both UR5e variants
    must exist. Each mirrors its Panda counterpart in
    ``libero.libero.envs.robots`` -- same mount and same base offsets -- with the
    stock robosuite UR5e MJCF in place of the Panda's.
    """
    from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
    from robosuite.robots import ROBOT_CLASS_MAPPING
    from robosuite.robots.single_arm import SingleArm
    from robosuite.utils.mjcf_utils import xml_path_completion

    if UR5E_MOUNTED_NAME in ROBOT_CLASS_MAPPING:
        return

    class _UR5eBase(ManipulatorModel):
        def __init__(self, idn=0):
            super().__init__(xml_path_completion("robots/ur5e/robot.xml"), idn=idn)
            self.set_joint_attribute(
                attrib="damping", values=np.array((0.1, 0.1, 0.1, 0.01, 0.01, 0.01))
            )

        @property
        def default_gripper(self):
            # The experiment deliberately runs the Franka hand on the UR5e arm.
            return UR5E_GRIPPER

        @property
        def default_controller_config(self):
            return "default_ur5e"

        @property
        def init_qpos(self):
            return np.array([-0.470, -1.735, 2.480, -2.275, -1.590, -1.991])

        @property
        def top_offset(self):
            return np.array((0, 0, 1.0))

        @property
        def _horizontal_radius(self):
            return 0.5

        @property
        def arm_type(self):
            return "single"

    class MountedUR5e(_UR5eBase):
        """Mirrors ``MountedPanda``: RethinkMount, table/study/kitchen offsets."""

        @property
        def default_mount(self):
            return "RethinkMount"

        @property
        def base_xpos_offset(self):
            return {
                "bins": (-0.5, -0.1, 0),
                "empty": (-0.6, 0, 0),
                "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
                "study_table": lambda table_length: (-0.25 - table_length / 2, 0, 0),
                "kitchen_table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
            }

    class OnTheGroundUR5e(_UR5eBase):
        """Mirrors ``OnTheGroundPanda``: no mount, coffee/living-room offsets."""

        @property
        def default_mount(self):
            return None

        @property
        def base_xpos_offset(self):
            return {
                "bins": (-0.5, -0.1, 0),
                "empty": (-0.6, 0, 0),
                "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
                "coffee_table": lambda table_length: (-0.16 - table_length / 2, 0, 0.41),
                "living_room_table": lambda table_length: (
                    -0.16 - table_length / 2,
                    0,
                    0.42,
                ),
            }

    for name, cls in ((UR5E_MOUNTED_NAME, MountedUR5e), (UR5E_GROUND_NAME, OnTheGroundUR5e)):
        # robosuite resolves robot models by class name, so keep a live handle.
        globals()[name] = cls
        ROBOT_CLASS_MAPPING[name] = SingleArm


def env_kwargs_for_condition(condition: str) -> dict[str, Any]:
    """Extra ``OffScreenRenderEnv`` kwargs implied by the condition."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; expected one of {CONDITIONS}")
    if condition != "embodiment":
        return {}
    register_ur5e_variants()
    return {"robots": [UR5E_ENV_ROBOT_ARG], "gripper_types": UR5E_GRIPPER}


# --------------------------------------------------------------------------- #
# camera
# --------------------------------------------------------------------------- #
def quat_to_euler(quat):
    return euler.quat2euler(quat, "sxyz")


def euler_to_quat(euler_angles):
    return euler.euler2quat(euler_angles[0], euler_angles[1], euler_angles[2], "sxyz")


class Pose:
    # must be wxyz quaterion format
    def __init__(self, position=None, orientation=None):
        if position is None:
            position = np.zeros(3)  
        if orientation is None:
            orientation = np.array([1.0, 0, 0, 0])

        self.position = np.array(position)
        self.orientation = np.array(orientation)

    def get_position(self):
        return self.position

    def get_orientation(self):
        return self.orientation

    def set_position(self, position):
        self.position = np.array(position)

    def set_orientation(self, orientation):
        self.orientation = np.array(orientation)

    def transform(self, pose):
        new_position = self.position + quaternions.quat2mat(self.orientation) @ pose.position
        new_orientation = quaternions.qmult(self.orientation, pose.orientation)
        return Pose(new_position, new_orientation)

    def orientation_inv(self):
        orientation = self.orientation
        orientation[1:] *= -1  
        return orientation

    def inverse(self):
        return Pose(-1 * quaternions.quat2mat(self.orientation_inv()) @ self.position, self.orientation_inv())

def rotate_camera_based_on_robot_base(cur_camera_pos, cur_camera_quat, robot_base_pos, robot_base_quat, theta):
    """
        given wxyz quaterion format; camera pose rotate around robot base;
    """
    camera_world_pose = Pose(cur_camera_pos, cur_camera_quat)
    robot_world_pose = Pose(robot_base_pos, robot_base_quat)

    robot_robot_pose = Pose(np.array([0, 0, 0]), np.array([1, 0, 0, 0]))

    world_to_robot_transform = robot_robot_pose.transform(robot_world_pose.inverse())


    camera_robot_pose = world_to_robot_transform.transform(camera_world_pose)

    new_camera_robot_pose = Pose(np.array([0, 0, 0]), euler.euler2quat(0, 0, theta / 180 * np.pi)).transform(camera_robot_pose)

    new_camera_world_pose = world_to_robot_transform.inverse().transform(new_camera_robot_pose)

    return new_camera_world_pose.get_position(), new_camera_world_pose.get_orientation()

def apply_camera_condition(
    env: Any,
    camera_name: str = "agentview",
    azimuth_degrees=CAMERA_AZIMUTH_DEGREES,
    pivot_body: str = CAMERA_PIVOT_BODY,
    reposition_scale: float = CAMERA_REPOSITION_SCALE,
) -> dict[str, Any]:
    """Adapter around the copied reference rotation.

    All geometry lives in ``rotate_camera_based_on_robot_base`` and
    ``reposition_camera`` above, which are verbatim from the reference. This
    function only does what the reference's own ``rotate_camera`` wrapper does --
    read the camera and pivot poses out of ``sim.model``, call the rotation,
    write the result back, then reposition -- with the reference's default
    arguments (``robot_base_name="robot0_base"``, ``reposition_camera_scale=1.0``)
    and no debug rendering.

    One detail carried over from the reference, deliberate:

    * The pivot is read from ``sim.model.body_pos`` / ``body_quat``, a body's pose
      relative to its *parent*. The reference's ``main()`` passes
      ``robot0_link0``, a child body whose parent-frame position is the origin,
      so the camera orbits the world origin rather than the robot's actual base
      -- roughly halving the arc for a given angle. Kept as-is (rather than
      switching to ``robot0_base``) to match the reference exactly.
    * ``env`` is LIBERO's ``OffScreenRenderEnv``, whose underlying robosuite env
      is ``env.env``. The copied code indexes ``env.sim`` directly, so it is that
      inner env which gets passed to it.

    Returns the metadata ``worker.py`` forwards to the trainer as ``camera_info``.

    Must be called after every ``env.reset()``: LIBERO rebuilds the sim model on
    reset and the perturbation lives in ``sim.model``.
    """
    inner = env.env
    sim = inner.sim
    camera_id = sim.model.camera_name2id(camera_name)

    cur_camera_pos = sim.model.cam_pos[camera_id].copy()
    cur_camera_quat = sim.model.cam_quat[camera_id].copy()

    robot_base_id = sim.model.body_name2id(pivot_body)
    robot_base_pos = sim.model.body_pos[robot_base_id].copy()
    robot_base_quat = sim.model.body_quat[robot_base_id].copy()

    if not MUJOCO_WXYZ:  # xyzw -> wxyz
        cur_camera_quat = np.array([cur_camera_quat[3], cur_camera_quat[0], cur_camera_quat[1], cur_camera_quat[2]])
        robot_base_quat = np.array([robot_base_quat[3], robot_base_quat[0], robot_base_quat[1], robot_base_quat[2]])

    tgt_camera_pos, tgt_camera_quat = rotate_camera_based_on_robot_base(
        cur_camera_pos, cur_camera_quat, robot_base_pos, robot_base_quat, azimuth_degrees
    )

    if not MUJOCO_WXYZ:  # wxyz -> xyzw
        tgt_camera_quat = np.array([tgt_camera_quat[1], tgt_camera_quat[2], tgt_camera_quat[3], tgt_camera_quat[0]])

    sim.model.cam_pos[camera_id] = tgt_camera_pos
    sim.model.cam_quat[camera_id] = tgt_camera_quat
    sim.forward()
    reposition_camera(inner, camera_name, camera_id, pivot_body, scale=reposition_scale, debug=False)

    return {
        "camera": camera_name,
        "azimuth_degrees": float(azimuth_degrees),
        "pivot_body": pivot_body,
        "pivot_pos": np.asarray(robot_base_pos, dtype=float).tolist(),
        "reposition_scale": float(reposition_scale),
        "orbit_radius": float(np.linalg.norm((np.asarray(cur_camera_pos, dtype=float)
                                              - np.asarray(robot_base_pos, dtype=float))[:2])),
        "original_pos": np.asarray(cur_camera_pos, dtype=float).tolist(),
        "new_pos": np.asarray(sim.model.cam_pos[camera_id], dtype=float).tolist(),
    }


def reposition_camera(
        env,
        camera_name: str,
        camera_id: int,
        robot_base_name: str = "robot0_base",
        scale: float = 0.85,          # 0.85면 베이스 쪽으로 15% 가까워짐, 1.15면 15% 멀어짐
        debug: bool = False,
    ):
    # --- target(base) world pos ---
    base_id = env.sim.model.body_name2id(robot_base_name)
    base_pos_w = env.sim.data.xpos[base_id].copy()   # world

    # --- camera current world pos ---
    cam_bodyid = int(env.sim.model.cam_bodyid[camera_id])
    if cam_bodyid == -1:
        # camera is defined in world frame
        cam_pos_w = env.sim.model.cam_pos[camera_id].copy()
    else:
        # camera is defined in cam_body local frame -> convert local -> world
        body_pos_w = env.sim.data.xpos[cam_bodyid].copy()
        body_R_w = env.sim.data.xmat[cam_bodyid].reshape(3, 3).copy()
        cam_pos_local = env.sim.model.cam_pos[camera_id].copy()
        cam_pos_w = body_pos_w + body_R_w @ cam_pos_local

    # --- move along line: base + scale*(cam-base) ---
    v = cam_pos_w - base_pos_w
    new_cam_pos_w = base_pos_w + scale * v

    # --- write back to model.cam_pos (world or local depending on cam_bodyid) ---
    if cam_bodyid == -1:
        env.sim.model.cam_pos[camera_id] = new_cam_pos_w
    else:
        body_pos_w = env.sim.data.xpos[cam_bodyid].copy()
        body_R_w = env.sim.data.xmat[cam_bodyid].reshape(3, 3).copy()
        new_cam_pos_local = body_R_w.T @ (new_cam_pos_w - body_pos_w)
        env.sim.model.cam_pos[camera_id] = new_cam_pos_local

    env.sim.forward()
    if debug:
        camera_img = env.sim.render(
            camera_name=camera_name,  # 指定相机名称
            width=IMAGE_RESOLUTION,                  # 图像宽度
            height=IMAGE_RESOLUTION,                 # 图像高度
            depth=False,                # 是否需要深度图
            mode='offscreen'            # 离屏渲染模式
        )

        Image.fromarray(camera_img[::-1]).save(os.path.join(IMAGE_SAVE_PATH, f"scale_{scale:.2f}.png"))
    return env

def apply_post_reset_condition(
    env: Any, condition: str, azimuth_degrees: float = CAMERA_AZIMUTH_DEGREES
) -> dict[str, Any]:
    """Apply whatever the condition needs after each ``env.reset()``."""
    if condition == "camera":
        return apply_camera_condition(env, azimuth_degrees=azimuth_degrees)
    return {}


# --------------------------------------------------------------------------- #
# init states
# --------------------------------------------------------------------------- #
def set_init_state(env: Any, init_state: np.ndarray, condition: str) -> dict[str, Any]:
    """Load a LIBERO Panda init state, adapting it when the robot differs.

    LIBERO ships init states recorded on the Panda. Under the embodiment arm the
    arm has a different joint count, so only the object block transfers and the
    UR5e keeps its own home pose.
    """
    if condition != "embodiment":
        return env.set_init_state(init_state)

    sim = env.env.sim
    target_state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
    source_state = np.asarray(init_state, dtype=np.float64)

    target_arm_dof = len(env.env.robots[0].robot_model.joints)
    target_gripper_dof = len(env.env.robots[0].gripper.joints)
    target_robot_dof = target_arm_dof + target_gripper_dof
    source_robot_dof = PANDA_ARM_DOF + PANDA_GRIPPER_DOF

    target_nq = int(sim.model.nq)
    target_nv = int(sim.model.nv)
    dof_delta = target_robot_dof - source_robot_dof
    source_nq = target_nq - dof_delta
    source_nv = target_nv - dof_delta
    expected = 1 + source_nq + source_nv + int(sim.model.na)
    if source_state.shape != (expected,):
        raise ValueError(
            f"Unexpected LIBERO Panda init-state shape {source_state.shape}; expected {(expected,)}"
        )

    source_qpos = source_state[1 : 1 + source_nq]
    source_qvel = source_state[1 + source_nq : 1 + source_nq + source_nv]
    target_qpos = target_state[1 : 1 + target_nq]
    target_qvel = target_state[1 + target_nq : 1 + target_nq + target_nv]

    target_state[0] = source_state[0]
    # Objects only: the robot block stays at the UR5e's own reset pose.
    target_qpos[target_robot_dof:] = source_qpos[source_robot_dof:]
    target_qvel[target_robot_dof:] = source_qvel[source_robot_dof:]
    return env.regenerate_obs_from_state(target_state)


def condition_metadata(
    condition: str, azimuth_degrees: float = CAMERA_AZIMUTH_DEGREES
) -> dict[str, Any]:
    """Static description of the arm, for W&B config and result manifests."""
    if condition == "camera":
        return {
            "condition": "camera",
            "camera_angle": azimuth_degrees,
            "embodiment": "MountedPanda+PandaGripper",
            "camera_pivot": CAMERA_PIVOT_BODY,
            "description": (
                f"agentview rotated {azimuth_degrees:g} deg about the {CAMERA_PIVOT_BODY} "
                "frame, per JiyunJang-24/LIBERO rotate_recolor_dataset.py"
            ),
        }
    if condition == "embodiment":
        return {
            "condition": "embodiment",
            "camera_angle": 0.0,
            "embodiment": f"UR5e+{UR5E_GRIPPER}",
            "description": "UR5e arm carrying the Franka (Panda) gripper",
        }
    return {
        "condition": "none",
        "camera_angle": 0.0,
        "embodiment": "MountedPanda+PandaGripper",
        "description": "native LIBERO",
    }


# --------------------------------------------------------------------------- #
# object perturbation
#
# A causal probe: let the arm approach a graspable object and, exactly once,
# teleport that object further away. If the policy is genuinely re-planning off
# the KV cache it should notice and re-reach; if it is executing a plan it
# committed to earlier, it will close on empty air.
#
# The geometry helpers below are ported from JiyunJang-24/LIBERO
# ``xyg_scripts/rotate_recolor_dataset.py`` (``_body_depth``,
# ``_find_body_id_by_substring``, ``_find_eef_site_id``,
# ``robot_object_distance_xyz``, ``reposition_object``), trimmed to the parts
# this probe needs -- no rendering, no debug images.
# --------------------------------------------------------------------------- #

# Roughly where a Panda grasp commits: the fingers span ~8 cm open, so once the
# grip site is this close the reach is essentially finished.
PERTURB_TRIGGER_DISTANCE = 0.05
PERTURB_OFFSET = 0.05
# World-frame directions, named as they read in the agentview image: the camera
# sits at +x looking back along -x with +z up, so image-right is +y and the
# near edge of the table is +x.
PERTURB_DIRECTIONS = {
    "away": None,            # directly away from the grip site (default)
    "right": (0.0, 1.0),     # +y
    "left": (0.0, -1.0),     # -y
    "front": (1.0, 0.0),     # +x, toward the camera
    "back": (-1.0, 0.0),     # -x, away from the camera
}
EEF_SITE_CANDIDATES = ("gripper0_grip_site", "robot0_grip_site", "gripper0_ee_z")


def _body_depth(model, body_id: int) -> int:
    """Depth from worldbody; smaller means closer to the root."""
    depth = 0
    cur = int(body_id)
    while cur != 0:
        cur = int(model.body_parentid[cur])
        depth += 1
        if depth > 10_000:
            break
    return depth


def _eef_site_id(model) -> int:
    for name in EEF_SITE_CANDIDATES:
        try:
            return model.site_name2id(name)
        except Exception:
            continue
    for index, name in enumerate(model.site_names):
        if name and any(key in name.lower() for key in ("grip", "eef", "end")):
            return index
    raise ValueError("No end-effector site found")


def goal_object_names(bddl_path: str) -> list[str]:
    """Object names the task's goal predicate mentions.

    A LIBERO scene is mostly distractors -- task 0 has seven objects on the table
    and the goal names two of them. Perturbing whichever object the arm happens
    to pass nearest tests nothing: the policy sails on because the thing it was
    reaching for never moved. The goal block is the only place that says which
    objects the task is actually about.

        (:goal (And (In alphabet_soup_1 basket_1_contain_region) ...))

    Region names such as ``basket_1_contain_region`` are receptacles, not things
    to be picked up, so the ``_contain_region`` suffix is stripped and the result
    matched against body names by prefix.
    """
    text = pathlib.Path(bddl_path).read_text()
    start = text.find("(:goal")
    if start < 0:
        return []
    depth, end = 0, len(text)
    for index in range(start, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    goal = text[start:end]
    names = []
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", goal):
        if token.lower() in ("goal", "and", "or", "not", "in", "on", "open", "close", "turnon", "turnoff"):
            continue
        for suffix in ("_contain_region", "_region"):
            if token.endswith(suffix):
                token = token[: -len(suffix)]
                break
        if re.search(r"_\d+$", token) and token not in names:
            names.append(token)
    return names


def movable_object_bodies(sim, goal_names: list[str] | None = None,
                          select: str = "goal") -> list[int]:
    """Free-jointed bodies, split into the task's goal objects and the rest.

    A free joint is what makes a body pickable, but that alone includes every
    distractor on the table. ``select="goal"`` keeps only the objects the task is
    scored on; ``select="distractor"`` keeps only the others, which is the
    control -- moving something the task does not care about should cost nothing,
    and any failure there would mean the probe is measuring a visual
    discontinuity rather than a real invalidation of the reach.
    ``select="any"`` restores the unfiltered set.
    """
    model = sim.model
    bodies = []
    for index, name in enumerate(model.body_names):
        if not name or model.body_jntnum[index] <= 0:
            continue
        joint = int(model.body_jntadr[index])
        if int(model.jnt_type[joint]) != 0:  # not mjJNT_FREE
            continue
        is_goal = bool(goal_names) and any(name.startswith(goal) for goal in goal_names)
        if select == "goal" and goal_names and not is_goal:
            continue
        if select == "distractor" and is_goal:
            continue
        bodies.append(index)
    return bodies


def body_radius(model, geoms: list[int]) -> float:
    """Rough horizontal half-extent of a body, for clearance checks."""
    return max((float(np.max(model.geom_size[g])) for g in geoms), default=0.0)


def clearance_at(sim, body_id: int, target_xy: np.ndarray,
                 others: list[int], geom_cache: dict[int, list[int]]) -> tuple[float, str]:
    """Smallest gap between a body placed at ``target_xy`` and any other body.

    Negative means the two would overlap. Only the horizontal plane is checked;
    everything here stays on its surface.
    """
    model, data = sim.model, sim.data
    if body_id not in geom_cache:
        geom_cache[body_id] = body_geoms(model, body_id)
    own = body_radius(model, geom_cache[body_id])
    worst, culprit = float("inf"), ""
    for other in others:
        if other == body_id:
            continue
        if other not in geom_cache:
            geom_cache[other] = body_geoms(model, other)
        gap = float(np.linalg.norm(np.asarray(data.xpos[other])[:2] - target_xy))
        gap -= own + body_radius(model, geom_cache[other])
        if gap < worst:
            worst, culprit = gap, model.body_names[other]
    return worst, culprit


def body_geoms(model, body_id: int) -> list[int]:
    return [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == body_id]


def surface_distance(sim, body_id: int, eef: np.ndarray, geoms: list[int]) -> float:
    """Approximate gap between the grip site and a body's *surface*.

    Measuring to the body origin makes the trigger depend on how big the object
    is -- a book's centre stays 10 cm away while it is being grasped, a soup can's
    is 1 cm. Subtracting each geom's extent removes that dependence, so one
    threshold means the same thing across the suite.
    """
    data, model = sim.data, sim.model
    best = float("inf")
    for geom in geoms:
        centre = np.asarray(data.geom_xpos[geom], dtype=np.float64)
        radius = float(np.max(model.geom_size[geom]))
        best = min(best, float(np.linalg.norm(centre - eef)) - radius)
    return max(best, 0.0)


def nearest_object(sim, body_ids: list[int], geom_cache: dict[int, list[int]]
                   ) -> tuple[int, float, float]:
    """Closest movable body to the grip site, by surface distance.

    Returns ``(body_id, surface_distance, centre_distance)``; the centre distance
    is kept only for reporting, so the two can be compared in the logs.
    """
    sim.forward()
    eef = np.asarray(sim.data.site_xpos[_eef_site_id(sim.model)], dtype=np.float64)
    best_id, best_surface, best_centre = -1, float("inf"), float("inf")
    for body_id in body_ids:
        if body_id not in geom_cache:
            geom_cache[body_id] = body_geoms(sim.model, body_id)
        surface = surface_distance(sim, body_id, eef, geom_cache[body_id])
        if surface < best_surface:
            centre = float(np.linalg.norm(np.asarray(sim.data.xpos[body_id]) - eef))
            best_id, best_surface, best_centre = body_id, surface, centre
    return best_id, best_surface, best_centre


def translate_body(sim, body_id: int, delta_world: np.ndarray) -> np.ndarray:
    """Move a free-jointed body by ``delta_world`` metres, returning its new pos.

    A free joint carries the body's pose in ``qpos``, so that is what has to
    move; ``body_pos`` is updated alongside it to keep the model and the state
    consistent, which is what the reference implementation does.
    """
    model, data = sim.model, sim.data
    parent = int(model.body_parentid[body_id])
    if parent == 0:
        delta_parent = delta_world
    else:
        parent_rotation = np.asarray(data.xmat[parent], dtype=np.float64).reshape(3, 3)
        delta_parent = parent_rotation.T @ delta_world

    joint = int(model.body_jntadr[body_id])
    if model.body_jntnum[body_id] > 0 and int(model.jnt_type[joint]) == 0:
        address = int(model.jnt_qposadr[joint])
        data.qpos[address : address + 3] += delta_world
    model.body_pos[body_id] += delta_parent
    sim.forward()
    return np.asarray(data.xpos[body_id], dtype=np.float64).copy()


class ObjectPerturbation:
    """Fires once per episode, the first time the arm gets close to an object.

    ``distance`` is the trigger radius from the grip site, measured to the
    object's surface rather than its origin; ``offset`` is how far
    the object is pushed. The push is horizontal and directly away from the grip
    site, so the object stays on its surface and stays reachable -- the point is
    to invalidate the reach that was in flight, not to remove the object.
    """

    def __init__(self, distance: float = PERTURB_TRIGGER_DISTANCE,
                 offset: float = PERTURB_OFFSET, enabled: bool = False,
                 select: str = "goal", direction: str = "away",
                 object_name: str | None = None, min_clearance: float = 0.01):
        self.distance = float(distance)
        self.offset = float(offset)
        self.enabled = bool(enabled)
        self.select = select
        self.direction = direction
        # Name (substring) of a specific body to perturb, when the arm's own
        # choice is not the one under test.
        self.object_name = object_name or None
        # Required gap to the nearest other object after the push. "Away from the
        # gripper" is the intent, but a LIBERO table is crowded and that ray often
        # runs straight into a neighbour, so the heading is searched rather than
        # assumed -- see ``_choose_direction``.
        self.min_clearance = float(min_clearance)
        self.goal_names: list[str] = []
        self.reset()

    def set_task(self, bddl_path: str) -> list[str]:
        """Read the goal objects for the task about to be rolled out."""
        try:
            self.goal_names = goal_object_names(bddl_path)
        except Exception:
            self.goal_names = []
        return self.goal_names

    def reset(self) -> None:
        self.fired = False
        self.record: dict[str, Any] = {}
        self._bodies: list[int] | None = None
        self._geoms: dict[int, list[int]] = {}
        self.min_distance = float("inf")        # surface
        self.min_centre_distance = float("inf")
        # Recovery bookkeeping: once the object has been moved, how long the arm
        # takes to close on it again is the thing the probe is really asking
        # about -- success alone cannot separate "never noticed" from "noticed
        # and re-reached".
        self._moved_body: int | None = None
        self.fired_step: int | None = None
        self.recovery_step: int | None = None
        self.post_min_distance = float("inf")
        self.post_max_distance = 0.0
        self.distance_trace: list[tuple[int, float]] = []

    def _preferred_direction(self, sim, body_id: int) -> np.ndarray:
        """The heading the push would take if nothing were in the way."""
        fixed = PERTURB_DIRECTIONS.get(self.direction)
        if fixed is not None:
            return np.array([fixed[0], fixed[1], 0.0], dtype=np.float64)
        eef = np.asarray(sim.data.site_xpos[_eef_site_id(sim.model)], dtype=np.float64)
        away = np.asarray(sim.data.xpos[body_id], dtype=np.float64) - eef
        away[2] = 0.0
        norm = float(np.linalg.norm(away))
        if norm < 1e-3:
            base = np.asarray(
                sim.model.body_pos[sim.model.body_name2id(CAMERA_PIVOT_FALLBACK)],
                dtype=np.float64)
            away = np.asarray(sim.data.xpos[body_id], dtype=np.float64) - base
            away[2] = 0.0
            norm = max(float(np.linalg.norm(away)), 1e-6)
        return away / norm

    def _choose_direction(self, sim, body_id, before, neighbours):
        """Rotate off the preferred heading until the destination is free.

        Candidates are the preferred heading, then alternating rotations to
        either side in 15-degree steps, so the push stays as close as it can to
        "away from the arm" while clearing every neighbour by ``min_clearance``.
        Returns ``(unit, clearance, nearest, searched)``; ``unit`` is None when no
        heading clears at this offset.
        """
        preferred = self._preferred_direction(sim, body_id)
        angle = math.atan2(preferred[1], preferred[0])
        offsets = [0.0]
        for step in range(1, 13):
            offsets += [step * math.pi / 12.0, -step * math.pi / 12.0]
        best = None
        for index, delta in enumerate(offsets, 1):
            theta = angle + delta
            unit = np.array([math.cos(theta), math.sin(theta), 0.0], dtype=np.float64)
            gap, culprit = clearance_at(sim, body_id, (before + unit * self.offset)[:2],
                                        neighbours, self._geoms)
            if gap >= self.min_clearance:
                return unit, gap, culprit, index
            if best is None or gap > best[1]:
                best = (unit, gap, culprit, index)
        return None, best[1], best[2], len(offsets)

    def _track_recovery(self, sim, step_index: int) -> None:
        """After the move, watch the arm close the gap to *that* object again.

        The nearest goal object can change once a stage completes, so recovery is
        measured against the body that was actually displaced rather than
        whatever is closest now.
        """
        if self._moved_body is None:
            return
        eef = np.asarray(sim.data.site_xpos[_eef_site_id(sim.model)], dtype=np.float64)
        if self._moved_body not in self._geoms:
            self._geoms[self._moved_body] = body_geoms(sim.model, self._moved_body)
        distance = surface_distance(sim, self._moved_body, eef, self._geoms[self._moved_body])
        self.post_min_distance = min(self.post_min_distance, distance)
        self.post_max_distance = max(self.post_max_distance, distance)
        if step_index % 4 == 0:
            self.distance_trace.append((int(step_index), float(distance)))
        if self.recovery_step is None and distance <= self.distance:
            self.recovery_step = int(step_index)

    def recovery(self) -> dict[str, Any]:
        """Recovery summary, or an explanation of why there is none."""
        if self.fired_step is None:
            return {}
        return {
            "fired_step": self.fired_step,
            "recovery_step": self.recovery_step,
            "recovery_steps": (None if self.recovery_step is None
                               else self.recovery_step - self.fired_step),
            "recovered": self.recovery_step is not None,
            "post_min_distance": (None if self.post_min_distance == float("inf")
                                  else float(self.post_min_distance)),
            "post_max_distance": float(self.post_max_distance),
            "distance_trace": list(self.distance_trace),
        }

    def observe(self, env, step_index: int) -> dict[str, Any] | None:
        """Check the arm's distance and, if it has arrived, move the object once."""
        sim = env.env.sim
        if self._bodies is None:
            self._bodies = movable_object_bodies(sim, self.goal_names, self.select)
            if self.object_name:
                named = [b for b in self._bodies
                         if self.object_name in sim.model.body_names[b]]
                # Fall back to the unfiltered set rather than silently disabling
                # the probe if the name does not match this task's scene.
                self._bodies = named or self._bodies
        if not self._bodies:
            return None

        if self.fired:
            self._track_recovery(sim, step_index)
            return None

        body_id, distance, centre = nearest_object(sim, self._bodies, self._geoms)
        self.min_distance = min(self.min_distance, distance)
        self.min_centre_distance = min(self.min_centre_distance, centre)
        if not self.enabled or distance > self.distance:
            return None

        before = np.asarray(sim.data.xpos[body_id], dtype=np.float64).copy()
        neighbours = movable_object_bodies(sim, None, "any")
        unit, gap, culprit, searched = self._choose_direction(sim, body_id, before, neighbours)
        if unit is None:
            # Every heading is blocked at this offset. Moving anyway would
            # measure a collision rather than a displacement, so the probe stays
            # armed and retries on a later step once the scene has shifted.
            return None
        delta = unit * self.offset
        after = translate_body(sim, body_id, delta)
        self.fired = True
        self._moved_body = body_id
        self.fired_step = int(step_index)
        self.record = {
            "object": sim.model.body_names[body_id],
            "select": self.select,
            "step": int(step_index),
            "trigger_distance": distance,
            "trigger_centre_distance": centre,
            "offset": self.offset,
            "direction": self.direction,
            "direction_unit": [float(unit[0]), float(unit[1])],
            "directions_searched": searched,
            "clearance": gap,
            "nearest_neighbour": culprit,
            "before": before.tolist(),
            "after": after.tolist(),
        }
        return self.record

ACTION_SHIFT_MODES = ("none", "gain", "rotate", "delay")


class ActionShift:
    """Change how commanded actions move the arm, leaving the scene untouched.

    ``camera`` and ``embodiment`` shift what the robot *looks like*, so a single
    frame already reveals that something is different. That makes them useless
    for asking whether the policy mines its history: it never has to, because the
    current observation carries the same news.

    These shifts are invisible by construction. The scene renders exactly as it
    did in training; only the map from commanded action to resulting motion
    changes. The evidence therefore exists nowhere except across an
    ``(observation, action, observation)`` transition -- which is to say, only in
    the KV cache. If the policy calibrates from history at all, this is where it
    would have to show.

    ``gain``    the arm responds less (or more) than commanded
    ``rotate``  the commanded xy direction lands rotated in the world
    ``delay``   each command takes effect ``delay`` steps late

    The gripper channel is left alone in every mode: squeezing it would change
    grasping itself rather than the dynamics the policy would need to identify.
    """

    def __init__(self, mode: str = "none", gain: float = 1.0,
                 degrees: float = 0.0, delay: int = 0):
        if mode not in ACTION_SHIFT_MODES:
            raise ValueError(f"mode must be one of {ACTION_SHIFT_MODES}, got {mode!r}")
        self.mode = mode
        self.gain = float(gain)
        self.degrees = float(degrees)
        self.delay = int(delay)
        self._queue: list[np.ndarray] = []

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    def reset(self) -> None:
        self._queue = []

    def apply(self, action: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return action
        action = np.asarray(action, dtype=np.float32).copy()
        if self.mode == "gain":
            # Motion channels only; index 6 is the gripper.
            action[:6] *= self.gain
        elif self.mode == "rotate":
            theta = np.deg2rad(self.degrees)
            cos, sin = np.cos(theta), np.sin(theta)
            x, y = action[0], action[1]
            action[0] = cos * x - sin * y
            action[1] = sin * x + cos * y
        elif self.mode == "delay":
            self._queue.append(action)
            if len(self._queue) <= self.delay:
                # Nothing has landed yet: hold still rather than invent a command.
                held = np.zeros_like(action)
                held[6] = action[6]
                return held
            action = self._queue.pop(0)
        return action

    def metadata(self) -> dict:
        if not self.enabled:
            return {}
        return {"mode": self.mode, "gain": self.gain,
                "degrees": self.degrees, "delay": self.delay}

def scale_object_mass(sim, bddl_path: str, scale: float,
                      select: str = "goal") -> dict:
    """Make the task's target object heavier, changing nothing you can see.

    ``camera`` and ``embodiment`` announce themselves in the first frame, and
    even the action shifts are a global property of the controller. Mass is
    neither: the object renders identically, and the change surfaces only when
    the arm actually picks it up -- the wrist sags, the object lags, a marginal
    grasp slips. So the evidence is not merely absent from a single frame, it is
    absent until the policy interacts, and then it lives in the interaction
    itself.

    Inertia is scaled with the mass so the body stays physically coherent;
    scaling mass alone would leave a heavy object that spins like a light one.
    """
    goal_names = goal_object_names(bddl_path)
    bodies = movable_object_bodies(sim, goal_names, select=select)
    changed = {}
    for body_id in bodies:
        name = sim.model.body_id2name(body_id)
        before = float(sim.model.body_mass[body_id])
        sim.model.body_mass[body_id] = before * scale
        sim.model.body_inertia[body_id] *= scale
        changed[name] = {"before": before, "after": before * scale}
    return changed

def set_impedance_gains(env, kp: float | None = None,
                        damping_ratio: float = 1.0) -> dict:
    """Soften (or stiffen) the OSC controller after the env is built.

    LIBERO runs OSC_POSE at ``kp=150``, ``damping_ratio=1``. Scaling the *action*
    only shrinks the setpoint: the arm still tracks it exactly, just less far, so
    one ``(observation, action, observation)`` pair solves for the factor. That
    asks whether history is used at all, not whether *more* of it helps.

    Lowering ``kp`` is the harder question. The arm now undershoots by an amount
    that depends on load, speed and configuration, so no single transition
    identifies it -- an estimate needs samples spread across conditions. If the
    policy ever benefits from a longer cache, this is the shift where it should.

    Set here rather than through ``controller_configs``: LIBERO's env wrapper
    builds that dict itself from the ``controller`` name and passes it
    positionally, so a keyword of the same name collides.

    ``kd`` is rederived from the new ``kp`` the way the controller does at
    construction; leaving the old ``kd`` would change the damping ratio too and
    confound stiffness with damping.
    """
    if kp is None:
        return {}
    controller = env.env.robots[0].controller
    before = float(np.mean(controller.kp))
    controller.kp = np.ones(6, dtype=np.float64) * float(kp)
    controller.kd = 2 * np.sqrt(controller.kp) * damping_ratio
    return {"kp_before": before, "kp_after": float(kp),
            "damping_ratio": float(damping_ratio)}

class ApproachTracker:
    """Closest the grip site came to every movable object in the scene.

    Success rate cannot answer "which object did the policy go for": LIBERO
    scores against the BDDL goal, so a rewritten instruction is graded against
    the wrong target and every run reads as a failure. Behaviour has to be read
    off the trajectory instead.

    Surface distance rather than centre distance, for the same reason the
    perturbation probe uses it: a can and a book have very different radii, and
    a centre-distance threshold would quietly mean "closer" for one than the
    other.
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.reset()

    def reset(self) -> None:
        self.closest: dict[str, float] = {}
        self._bodies: list[int] | None = None
        self._geoms: dict[int, list[int]] = {}

    def observe(self, env) -> None:
        if not self.enabled:
            return
        sim = env.env.sim
        if self._bodies is None:
            self._bodies = movable_object_bodies(sim, None, select="any")
            self._geoms = {b: body_geoms(sim.model, b) for b in self._bodies}
        eef = np.asarray(sim.data.site_xpos[_eef_site_id(sim.model)], dtype=np.float64)
        for body_id in self._bodies:
            name = sim.model.body_id2name(body_id).replace("_main", "")
            distance = surface_distance(sim, body_id, eef, self._geoms[body_id])
            previous = self.closest.get(name)
            if previous is None or distance < previous:
                self.closest[name] = float(distance)

    def result(self) -> dict:
        return dict(sorted(self.closest.items(), key=lambda kv: kv[1]))

def swap_object_positions(sim, name_a: str, name_b: str) -> dict:
    """Exchange where two free-jointed objects sit, changing nothing else.

    The text experiments held the scene fixed and moved the words. This is the
    mirror: the words and the object inventory are untouched, only the layout
    differs. It separates two things the ordinary rollout confounds --

      tracking   the arm goes wherever the named object now is
      prior      the arm goes where that object sat during training,
                 and picks up whatever occupies the spot instead

    Both bodies keep their own orientation; only translation is exchanged, so a
    swap cannot fail because a can ended up lying like a book.

    Names match by substring against the body names, so "tomato_sauce" finds
    "tomato_sauce_1_main".
    """
    model, data = sim.model, sim.data

    def free_joint_qpos_address(fragment: str) -> tuple[int, str]:
        for index in range(model.nbody):
            name = model.body_id2name(index) or ""
            if fragment not in name:
                continue
            if model.body_jntnum[index] <= 0:
                continue
            joint = int(model.body_jntadr[index])
            if int(model.jnt_type[joint]) != 0:  # mjJNT_FREE
                continue
            return int(model.jnt_qposadr[joint]), name
        raise ValueError(f"no free-jointed body matching {fragment!r}")

    addr_a, full_a = free_joint_qpos_address(name_a)
    addr_b, full_b = free_joint_qpos_address(name_b)
    pos_a = np.array(data.qpos[addr_a:addr_a + 3])
    pos_b = np.array(data.qpos[addr_b:addr_b + 3])
    data.qpos[addr_a:addr_a + 3] = pos_b
    data.qpos[addr_b:addr_b + 3] = pos_a
    sim.forward()
    return {"a": full_a, "b": full_b,
            "a_from": pos_a.tolist(), "a_to": pos_b.tolist()}

BACKGROUND_MODES = ("none", "tint", "dark", "bright", "invert")

_BACKGROUND_TINTS = {
    "tint":   (0.45, 0.75, 1.25),   # cool cast
    "dark":   (0.40, 0.40, 0.40),
    "bright": (1.60, 1.60, 1.60),
    "invert": (1.30, 0.55, 0.55),   # warm cast, the opposite end from "tint"
}


def shift_background(sim, mode: str = "none") -> dict:
    """Recolour everything that is not an object or the robot.

    The text experiments moved the words with the scene fixed; the swap
    experiments moved the objects with the words fixed. This moves neither --
    every object keeps its identity, its pose and its name, and the instruction
    is untouched. Only the table, walls and floor change colour.

    Nothing task relevant changes, so a policy that had bound its plan to task
    structure should not notice. One that had bound it to the look of the scene
    should. That is the whole question, isolated: appearance with geometry and
    language held constant.

    Robot geoms are left alone; recolouring the arm would reintroduce the
    embodiment confound this condition is meant to avoid.
    """
    if mode == "none":
        return {}
    if mode not in _BACKGROUND_TINTS:
        raise ValueError(f"mode must be one of {BACKGROUND_MODES}, got {mode!r}")
    tint = np.asarray(_BACKGROUND_TINTS[mode], dtype=np.float64)

    model = sim.model
    object_bodies = set(movable_object_bodies(sim, None, select="any"))
    touched = 0
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        if body in object_bodies:
            continue
        name = model.body_id2name(body) or ""
        if name.startswith(("robot", "gripper")):
            continue
        rgba = model.geom_rgba[geom].copy()
        rgba[:3] = np.clip(rgba[:3] * tint, 0.0, 1.0)
        model.geom_rgba[geom] = rgba
        touched += 1
    # Materials carry the visible colour when a geom defers to one, so they get
    # the same treatment or the tint would silently do nothing.
    for mat in range(model.nmat):
        rgba = model.mat_rgba[mat].copy()
        rgba[:3] = np.clip(rgba[:3] * tint, 0.0, 1.0)
        model.mat_rgba[mat] = rgba
    sim.forward()
    return {"mode": mode, "geoms": touched, "materials": int(model.nmat)}


STRUCTURE_MODES = ("none", "notex", "swaptex", "nofurn", "all")


def restructure_background(sim, mode: str = "none") -> dict:
    """Change what the background *is*, not what colour it is.

    ``shift_background`` multiplies every static colour by a tint. That leaves
    edges, layout and texture pattern untouched, so the scene is displaced by
    something close to a low-dimensional colour offset -- and the policy did not
    notice it at all. The obvious objection is that recolouring is the easy case:
    it only shows the model tolerates a shift the representation can carry as a
    small perturbation, not that it genuinely re-reads the background.

    This makes the harder change. The pattern on the surfaces is replaced or
    removed and the furniture is taken out of the room, so the background stops
    being a recoloured member of the training family and becomes a scene the
    model has not seen. If a plan bound to task structure survives this too, the
    background really is ignored. If it breaks here but not under a tint, what
    the model has is interpolation inside the family it was trained on rather
    than an ability to copy an arbitrary background forward.

    Physics is untouched by construction. LIBERO separates visual geoms from
    collision geoms -- the visible table is a mesh whose colliders are separate
    boxes, and the walls are named ``*_visual`` -- so appearance and texture
    changes cannot move a contact. Objects and the robot are excluded for the
    same reason as in ``shift_background``: recolouring them would mix in the
    object-identity and embodiment confounds these conditions isolate.
    """
    if mode == "none":
        return {}
    if mode not in STRUCTURE_MODES:
        raise ValueError(f"mode must be one of {STRUCTURE_MODES}, got {mode!r}")

    model = sim.model
    object_bodies = set(movable_object_bodies(sim, None, select="any"))

    # Materials reachable from static geometry only. Object materials carry the
    # object's identity, so retexturing them would change the task, not the room.
    static_mats: list[int] = []
    hidden = 0
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        if body in object_bodies:
            continue
        name = model.body_id2name(body) or ""
        if name.startswith(("robot", "gripper")):
            continue
        mat = int(model.geom_matid[geom])
        if mat >= 0 and mat not in static_mats:
            static_mats.append(mat)
        if mode in ("nofurn", "all"):
            # The free-standing furniture meshes are decoration: removing them
            # changes the contents of the room without moving a collider.
            gname = model.geom_id2name(geom) or ""
            if gname in ("wall_decoration", "living_room") or "decoration" in gname:
                rgba = model.geom_rgba[geom].copy()
                rgba[3] = 0.0
                model.geom_rgba[geom] = rgba
                hidden += 1

    # mat_texid is (nmat,) on older MuJoCo and (nmat, nrole) on 3.x, where the
    # RGB role sits at index 1. Write through a view so both layouts work.
    texid = model.mat_texid
    two_d = getattr(texid, "ndim", 1) > 1

    def _get(mat):
        return int(texid[mat][1]) if two_d else int(texid[mat])

    def _set(mat, value):
        if two_d:
            texid[mat][1] = value
        else:
            texid[mat] = value

    retextured = 0
    if mode in ("notex", "all"):
        for mat in static_mats:
            if _get(mat) >= 0:
                _set(mat, -1)          # fall back to the flat material colour
                retextured += 1
    elif mode == "swaptex":
        textured = [m for m in static_mats if _get(m) >= 0]
        if len(textured) > 1:
            # Rotate the assignment: the wall wears the floor's texture and so
            # on. Every surface keeps a real texture, so the change is in the
            # pattern rather than in the presence of pattern.
            values = [_get(m) for m in textured]
            for m, v in zip(textured, values[1:] + values[:1]):
                _set(m, v)
                retextured += 1

    sim.forward()
    return {"mode": mode, "materials": len(static_mats),
            "retextured": retextured, "hidden_geoms": hidden}
