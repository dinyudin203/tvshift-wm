"""
Stage A: LIBERO demo 를 states 리플레이하면서 agentview 를 여러 theta 로 재렌더링한다.

- 카메라는 robot base(robot0_link0) 의 z축 기준으로 theta 도 회전 (only_rotate_dataset.py 와 동일한 수식)
- wrist(eye_in_hand) 는 카메라 회전과 무관하므로 demo 당 한 번만 렌더링해서 모든 theta 가 공유
- 에피소드 단위로 mp4(h264) + parquet 을 shard 디렉토리에 떨궈두고, Stage B(pack_lerobot.py)가
  이것들을 LeRobot v3.0 데이터셋으로 합친다.

usage:
  MUJOCO_GL=egl PYTHONPATH=/workspace/yujin/LIBERO-JY \
  python render_shard.py --task 0 --demo-start 0 --demo-end 10 --out /workspace/yujin/data/libero_spatial_rot/shards
"""
import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import transforms3d.euler as euler
import transforms3d.quaternions as quaternions

CAMERA_NAME = "agentview"
WRIST_CAMERA_NAME = "robot0_eye_in_hand"
ROBOT_BASE_NAME = "robot0_link0"
DEFAULT_THETAS = [0, 30, 60, 90, 120, 240, 270, 300, 330]


# ---------------------------------------------------------------- camera math
# only_rotate_dataset.py 의 Pose / rotate_camera_based_on_robot_base 와 동일 (wxyz quaternion)
class Pose:
    def __init__(self, position=None, orientation=None):
        self.position = np.zeros(3) if position is None else np.array(position, dtype=np.float64)
        self.orientation = np.array([1.0, 0, 0, 0]) if orientation is None else np.array(orientation, dtype=np.float64)

    def transform(self, pose):
        new_position = self.position + quaternions.quat2mat(self.orientation) @ pose.position
        new_orientation = quaternions.qmult(self.orientation, pose.orientation)
        return Pose(new_position, new_orientation)

    def orientation_inv(self):
        o = self.orientation.copy()
        o[1:] *= -1
        return o

    def inverse(self):
        inv = self.orientation_inv()
        return Pose(-1 * quaternions.quat2mat(inv) @ self.position, inv)


def rotated_camera_pose(cam_pos, cam_quat, base_pos, base_quat, theta_deg):
    cam_world = Pose(cam_pos, cam_quat)
    base_world = Pose(base_pos, base_quat)
    world_to_base = Pose().transform(base_world.inverse())
    cam_base = world_to_base.transform(cam_world)
    rot = Pose(np.zeros(3), euler.euler2quat(0, 0, np.deg2rad(theta_deg)))
    new_cam_base = rot.transform(cam_base)
    new_cam_world = world_to_base.inverse().transform(new_cam_base)
    return new_cam_world.position, new_cam_world.orientation


# ---------------------------------------------------------------- no-op filter
def is_noop(action, prev_action=None, threshold=1e-4):
    """openvla/modified_libero_rlds 와 동일한 no-op 판정."""
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold
    return np.linalg.norm(action[:-1]) < threshold and action[-1] == prev_action[-1]


# ---------------------------------------------------------------- video writer
def write_mp4(path, frames, fps, crf=18):
    import av

    os.makedirs(os.path.dirname(path), exist_ok=True)
    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = frames[0].shape[1]
    stream.height = frames[0].shape[0]
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(crf), "preset": "medium", "g": "1"}
    for frame in frames:
        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in stream.encode(av_frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--raw-dir", default="/workspace/yujin/data/libero_raw/libero_spatial")
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--demo-start", type=int, default=0)
    ap.add_argument("--demo-end", type=int, default=10**9)
    ap.add_argument("--thetas", type=int, nargs="+", default=DEFAULT_THETAS)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task)
    language = task.language

    # language -> hdf5 파일 매칭
    h5_path = None
    for name in sorted(os.listdir(args.raw_dir)):
        if not name.endswith(".hdf5"):
            continue
        with h5py.File(os.path.join(args.raw_dir, name), "r") as f:
            info = json.loads(f["data"].attrs["problem_info"])
        if info["language_instruction"].strip() == language.strip():
            h5_path = os.path.join(args.raw_dir, name)
            break
    assert h5_path is not None, f"no hdf5 for task {args.task}: {language}"

    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=args.res,
        camera_widths=args.res,
        camera_names=[CAMERA_NAME, WRIST_CAMERA_NAME],
        hard_reset=False,
    )
    env.seed(0)
    env.reset()
    sim = env.env.sim

    cam_id = sim.model.camera_name2id(CAMERA_NAME)
    orig_cam_pos = sim.model.cam_pos[cam_id].copy()
    orig_cam_quat = sim.model.cam_quat[cam_id].copy()
    base_id = sim.model.body_name2id(ROBOT_BASE_NAME)
    base_pos = sim.model.body_pos[base_id].copy()
    base_quat = sim.model.body_quat[base_id].copy()

    theta_poses = {}
    for theta in args.thetas:
        theta_poses[theta] = rotated_camera_pose(orig_cam_pos, orig_cam_quat, base_pos, base_quat, theta)

    f = h5py.File(h5_path, "r")
    demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
    demo_keys = demo_keys[args.demo_start : args.demo_end]

    t_start = time.time()
    for demo_key in demo_keys:
        demo_idx = int(demo_key.split("_")[1])
        stem = f"task{args.task:02d}_demo{demo_idx:03d}"
        done_marker = os.path.join(args.out, "_done", f"{stem}.json")
        if args.skip_existing and os.path.exists(done_marker):
            print(f"[skip] {stem}", flush=True)
            continue

        grp = f["data"][demo_key]
        states = grp["states"][()]
        actions = grp["actions"][()].astype(np.float32)
        ee_states = grp["obs/ee_states"][()].astype(np.float32)
        joint_states = grp["obs/joint_states"][()].astype(np.float32)
        gripper_states = grp["obs/gripper_states"][()].astype(np.float32)

        # no-op 프레임 제거 (openvla 방식)
        keep = []
        prev = None
        for t in range(len(actions)):
            if not is_noop(actions[t], prev):
                keep.append(t)
            prev = actions[t]
        if len(keep) < 8:
            print(f"[drop-short] {stem} kept={len(keep)}", flush=True)
            continue

        # demo 성공 여부 확인 (마지막 state 기준)
        sim.set_state_from_flattened(states[-1])
        sim.forward()
        success = bool(env.check_success())
        if not success:
            print(f"[drop-fail] {stem}", flush=True)
            continue

        wrist_frames = []
        theta_frames = {theta: [] for theta in args.thetas}
        for t in keep:
            sim.set_state_from_flattened(states[t])
            sim.forward()
            wrist_frames.append(sim.render(camera_name=WRIST_CAMERA_NAME, width=args.res, height=args.res)[::-1].copy())
            for theta in args.thetas:
                pos, quat = theta_poses[theta]
                sim.model.cam_pos[cam_id] = pos
                sim.model.cam_quat[cam_id] = quat
                sim.forward()
                theta_frames[theta].append(
                    sim.render(camera_name=CAMERA_NAME, width=args.res, height=args.res)[::-1].copy()
                )
        sim.model.cam_pos[cam_id] = orig_cam_pos
        sim.model.cam_quat[cam_id] = orig_cam_quat

        idx = np.array(keep)
        state8 = np.concatenate([ee_states[idx], gripper_states[idx]], axis=1).astype(np.float32)
        frame_table = {
            "observation.state": state8,
            "observation.states.ee_state": ee_states[idx],
            "observation.states.joint_state": joint_states[idx],
            "observation.states.gripper_state": gripper_states[idx],
            "action": actions[idx],
        }

        wrist_path = os.path.join(args.out, "wrist", f"{stem}.mp4")
        write_mp4(wrist_path, wrist_frames, args.fps)
        for theta in args.thetas:
            write_mp4(os.path.join(args.out, f"theta{theta:03d}", f"{stem}.mp4"), theta_frames[theta], args.fps)

        npz_path = os.path.join(args.out, "frames", f"{stem}.npz")
        os.makedirs(os.path.dirname(npz_path), exist_ok=True)
        np.savez(npz_path, **frame_table)

        os.makedirs(os.path.dirname(done_marker), exist_ok=True)
        with open(done_marker, "w") as fh:
            json.dump(
                {
                    "task": args.task,
                    "demo": demo_idx,
                    "stem": stem,
                    "language": language,
                    "length": int(len(keep)),
                    "raw_length": int(len(actions)),
                    "thetas": args.thetas,
                    "res": args.res,
                    "fps": args.fps,
                },
                fh,
            )
        print(f"[ok] {stem} kept={len(keep)}/{len(actions)} elapsed={time.time()-t_start:.0f}s", flush=True)

    f.close()
    env.close()
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
