"""
Script taken from robomimic scripts/dataset_states_to_obs.py
https://github.com/ARISE-Initiative/robomimic/blob/main/robomimic/scripts/dataset_states_to_obs.py

Script to extract observations from low-dimensional simulation states in a robosuite dataset.

Args:
    dataset (str): path to input hdf5 dataset

    output_name (str): name of output hdf5 dataset

    n (int): if provided, stop after n trajectories are processed

    shaped (bool): if flag is set, use dense rewards

    camera_names (str or [str]): camera name(s) to use for image observations. 
        Leave out to not use image observations.

    camera_height (int): height of image observation.

    camera_width (int): width of image observation

    done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a success state.
        If 1, done is 1 at the end of each trajectory. If 2, both.

    copy_rewards (bool): if provided, copy rewards from source file instead of inferring them

    copy_dones (bool): if provided, copy dones from source file instead of inferring them

Example usage:
    
    # extract low-dimensional observations
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name low_dim.hdf5 --done_mode 2
    
    # extract 84x84 image observations
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image.hdf5 \
        --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

    # extract 84x84 image and depth observations
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name depth.hdf5 \
        --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 --depth

    # (space saving option) extract 84x84 image observations with compression and without 
    # extracting next obs (not needed for pure imitation learning algos)
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image.hdf5 \
        --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 \
        --compress --exclude-next-obs

    # use dense rewards, and only annotate the end of trajectories with done signal
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image_dense_done_1.hdf5 \
        --done_mode 1 --dense --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84
"""
import os
import json
import h5py
import argparse
import numpy as np
from copy import deepcopy
from tqdm import tqdm

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase

import cpgen_envs

# Multi-view camera support
from demo_aug.configs.multi_view_config import (
    MultiViewCameraConfig,
    ThirdViewCameraConfig,
    add_multi_view_args,
    create_config_from_args,
)
from demo_aug.envs.wrapper.multi_view_wrapper import MultiViewEnvWrapper
from demo_aug.utils.robosuite_utils import (
    is_robosuite_v15_or_later,
    refactor_composite_controller_config,
)


def quaternion_to_rotation_matrix(quat):
    """
    Convert quaternion to rotation matrix.

    Args:
        quat: quaternion in format [qx, qy, qz, qw]

    Returns:
        3x3 rotation matrix
    """
    qx, qy, qz, qw = quat

    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])

    return R


def rotation_matrix_to_quaternion(R):
    """
    Convert rotation matrix to quaternion.

    Args:
        R: 3x3 rotation matrix

    Returns:
        quaternion in format [qw, qx, qy, qz]
    """
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    return np.array([qw, qx, qy, qz])


def compute_tcp_pixel_data(h, w, tcp_pose, camera_extrinsics, camera_intrinsics):
    """
    Compute TCP pixel coordinates and direction vectors

    Args:
        h: image height
        w: image width
        tcp_pose: TCP pose as [x, y, z, qw, qx, qy, qz]
        camera_extrinsics: 4x4 camera extrinsic matrix (world to camera)
        camera_intrinsics: 3x3 camera intrinsic matrix

    Returns:
        Dict containing:
        - tcp_pixel_coords: [u, v, depth]
        - tcp_dir_x: [dir_x, dir_y] for X-axis
        - tcp_dir_y: [dir_x, dir_y] for Y-axis
        - tcp_dir_z: [dir_x, dir_y] for Z-axis
        - tcp_pos: gripper position in camera frame
        - tcp_orn: gripper orientation (rotation matrix) in camera frame (flattened)
        - tcp_quat: gripper quaternion in camera frame [qw, qx, qy, qz]
        - camera_intrinsics: camera intrinsic matrix
        - camera_extrinsics: camera extrinsic matrix
    """
    try:
        # Extract position and quaternion from TCP pose
        gripper_pos = tcp_pose[:3]
        gripper_quat = tcp_pose[3:]  # [qx, qy, qz, qw]

        # Convert quaternion to rotation matrix
        gripper_rot = quaternion_to_rotation_matrix(gripper_quat)

        # Create gripper transformation matrix
        gripper_transform = np.eye(4)
        gripper_transform[:3, :3] = gripper_rot
        gripper_transform[:3, 3] = gripper_pos

        # Transform gripper pose to camera coordinates
        world_to_camera = camera_extrinsics
        gripper_in_camera = world_to_camera @ gripper_transform

        # Project gripper position to image coordinates
        gripper_pos_camera = gripper_in_camera[:3, 3]
        gripper_orn_camera = gripper_in_camera[:3, :3]
        # transform gripper_orn_camera to quaternion
        gripper_quat_camera = rotation_matrix_to_quaternion(gripper_orn_camera)
        depth = gripper_pos_camera[2]

        if depth <= 0:  # Behind camera
            return None

        # Project to image plane
        pixel_coords = camera_intrinsics @ gripper_pos_camera
        pixel_coords = pixel_coords / pixel_coords[2]

        u, v = int(pixel_coords[0]), int(pixel_coords[1])

        # Check if coordinates are within image bounds
        if not (0 <= u < w and 0 <= v < h):
            return None

        # Calculate direction vectors for X and Y axes
        axis_length = 1

        # Extract axes from rotation matrix in camera coordinates
        x_axis_camera = gripper_in_camera[:3, 0] * axis_length
        y_axis_camera = gripper_in_camera[:3, 1] * axis_length
        z_axis_camera = gripper_in_camera[:3, 2] * axis_length

        # Project axis endpoints
        x_axis_end_camera = gripper_pos_camera + x_axis_camera
        if x_axis_end_camera[2] <= 0:
            if (gripper_pos_camera[2] > 0) and (x_axis_camera[2] < 0):
                scale = gripper_pos_camera[2] / x_axis_camera[2]
                x_axis_camera = x_axis_camera * scale
                x_axis_end_camera = gripper_pos_camera + x_axis_camera * 0.99  # make sure in front of the camera
        y_axis_end_camera = gripper_pos_camera + y_axis_camera
        if y_axis_end_camera[2] <= 0:
            if (gripper_pos_camera[2] > 0) and (y_axis_camera[2] < 0):
                scale = gripper_pos_camera[2] / y_axis_camera[2]
                y_axis_camera = y_axis_camera * scale
                y_axis_end_camera = gripper_pos_camera + y_axis_camera * 0.99  # make sure in front of the camera
        z_axis_end_camera = gripper_pos_camera + z_axis_camera
        if z_axis_end_camera[2] <= 0:
            if (gripper_pos_camera[2] > 0) and (z_axis_camera[2] < 0):
                scale = gripper_pos_camera[2] / z_axis_camera[2]
                z_axis_camera = z_axis_camera * scale
                z_axis_end_camera = gripper_pos_camera + z_axis_camera * 0.99  # make sure in front of the camera

        # Calculate direction vectors in image space
        tcp_dir_x = np.array([0.0, 0.0])
        tcp_dir_y = np.array([0.0, 0.0])
        tcp_dir_z = np.array([0.0, 0.0])

        # X-axis direction
        if x_axis_end_camera[2] > 0:
            x_axis_end_homogeneous = camera_intrinsics @ x_axis_end_camera
            x_axis_end_homogeneous = x_axis_end_homogeneous / x_axis_end_homogeneous[2]

            dir_x = x_axis_end_homogeneous[0] - u
            dir_y = x_axis_end_homogeneous[1] - v

            tcp_dir_x = np.array([dir_x, dir_y])

        # Y-axis direction
        if y_axis_end_camera[2] > 0:
            y_axis_end_homogeneous = camera_intrinsics @ y_axis_end_camera
            y_axis_end_homogeneous = y_axis_end_homogeneous / y_axis_end_homogeneous[2]

            dir_x = y_axis_end_homogeneous[0] - u
            dir_y = y_axis_end_homogeneous[1] - v

            tcp_dir_y = np.array([dir_x, dir_y])

        # Z-axis direction
        if z_axis_end_camera[2] > 0:
            z_axis_end_homogeneous = camera_intrinsics @ z_axis_end_camera
            z_axis_end_homogeneous = z_axis_end_homogeneous / z_axis_end_homogeneous[2]

            dir_x = z_axis_end_homogeneous[0] - u
            dir_y = z_axis_end_homogeneous[1] - v

            tcp_dir_z = np.array([dir_x, dir_y])

        return {
            'tcp_pixel_coords': np.array([u, v, depth], dtype=np.float32),
            'tcp_dir_x': tcp_dir_x.astype(np.float32),
            'tcp_dir_y': tcp_dir_y.astype(np.float32),
            'tcp_dir_z': tcp_dir_z.astype(np.float32),
            'tcp_pos': gripper_pos_camera.astype(np.float32),
            'tcp_orn': gripper_orn_camera.reshape(-1).astype(np.float32),
            'tcp_quat': gripper_quat_camera.astype(np.float32),  # [qw, qx, qy, qz]
            # 'camera_intrinsics': camera_intrinsics.astype(np.float32),
            # 'camera_extrinsics': camera_extrinsics.astype(np.float32)
        }

    except Exception as e:
        return None


def extract_trajectory(
    env, 
    initial_state, 
    states, 
    actions,
    actions_abs,
    done_mode,
    camera_names=None, 
    camera_height=84, 
    camera_width=84,
):
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load to extract information
        actions (np.array): array of actions
        done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a 
            success state. If 1, done is 1 at the end of each trajectory.
            If 2, do both.
    """
    assert isinstance(env, (EnvBase, MultiViewEnvWrapper))
    assert states.shape[0] == actions.shape[0]

    # load the initial state
    obs = env.reset_to(initial_state)

    # maybe add in intrinsics and extrinsics for all cameras
    camera_info = None
    is_robosuite_env = EnvUtils.is_robosuite_env(env=env)
    if is_robosuite_env:
        camera_info = get_camera_info(
            env=env,
            camera_names=camera_names, 
            camera_height=camera_height, 
            camera_width=camera_width,
        )

    traj = dict(
        obs=[], 
        next_obs=[], 
        rewards=[], 
        dones=[], 
        actions=np.array(actions), 
        states=np.array(states), 
        initial_state_dict=initial_state,
    )
    if actions_abs is not None:
        traj["actions_abs"] = np.array(actions_abs)
    
    traj_len = states.shape[0]

    # Identify eye-in-hand cameras — these need fresh world-frame extrinsics
    # each timestep because the camera moves with the robot arm.
    # camera_info stores EEF-relative extrinsics for eye-in-hand cameras,
    # which is correct for saving to HDF5 but NOT for TCP projection.
    eye_in_hand_cameras = set()
    if camera_names is not None:
        for cam_name in camera_names:
            if "eye_in_hand" in cam_name:
                eye_in_hand_cameras.add(cam_name)

    def _compute_tcp_for_obs(observation):
        """Compute TCP pixel data for an observation using current sim state.

        For all cameras (fixed and eye-in-hand), TCP data is computed in camera
        frame using world-frame extrinsics. Eye-in-hand cameras get fresh
        extrinsics from the simulator each call (since the camera moves with
        the arm), while fixed cameras use cached extrinsics from camera_info.
        """
        if camera_info is None or camera_names is None:
            return

        # Extract TCP pose [x, y, z, qw, qx, qy, qz] from observation
        tcp_pose = None
        for key in observation.keys():
            if "eef_pos" in key:
                quat_key = key.replace("_pos", "_quat")
                if quat_key in observation:
                    tcp_pose = np.concatenate([observation[key], observation[quat_key]])
                    break

        if tcp_pose is None:
            return

        for cam_name in camera_names:
            if cam_name not in camera_info:
                continue
            K = np.array(camera_info[cam_name]["intrinsics"])

            if cam_name in eye_in_hand_cameras:
                # Get fresh world-frame extrinsics (camera-to-world) from sim
                R_world = env.get_camera_extrinsic_matrix(camera_name=cam_name)
            else:
                # Fixed camera: cached extrinsics are already world-frame
                R_world = np.array(camera_info[cam_name]["extrinsics"])

            # inv(R_world) = world-to-camera transform
            tcp_data = compute_tcp_pixel_data(
                h=camera_height,
                w=camera_width,
                tcp_pose=tcp_pose,
                camera_extrinsics=np.linalg.inv(R_world),
                camera_intrinsics=K,
            )

            if tcp_data is not None:
                for data_key, data_val in tcp_data.items():
                    observation[f"{cam_name}_{data_key}"] = data_val

    # iteration variable @t is over "next obs" indices
    for t in range(1, traj_len + 1):

        # Compute TCP for obs BEFORE advancing state so sim matches obs
        _compute_tcp_for_obs(obs)

        # get next observation
        if t == traj_len:
            # play final action to get next observation for last timestep
            next_obs, _, _, _ = env.step(actions[t - 1])
        else:
            # reset to simulator state to get observation
            next_obs = env.reset_to({"states" : states[t]})

        # infer reward signal
        # note: our tasks use reward r(s'), reward AFTER transition, so this is
        #       the reward for the current timestep
        r = env.get_reward()

        # infer done signal
        done = False
        if (done_mode == 1) or (done_mode == 2):
            # done = 1 at end of trajectory
            done = done or (t == traj_len)
        if (done_mode == 0) or (done_mode == 2):
            # done = 1 when s' is task success state
            done = done or env.is_success()["task"]
        done = int(done)

        # Compute TCP for next_obs AFTER advancing state so sim matches next_obs
        _compute_tcp_for_obs(next_obs)

        # collect transition
        traj["obs"].append(obs)
        traj["next_obs"].append(next_obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)

        # update for next iter
        obs = deepcopy(next_obs)

    # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj, camera_info


def get_camera_info(
    env,
    camera_names=None, 
    camera_height=84, 
    camera_width=84,
):
    """
    Helper function to get camera intrinsics and extrinsics for cameras being used for observations.
    """

    # TODO: make this function more general than just robosuite environments
    assert EnvUtils.is_robosuite_env(env=env)

    # check for v1.5+ robosuite
    import robosuite
    is_v15 = (robosuite.__version__.split(".")[0] == "1") and (robosuite.__version__.split(".")[1] >= "5")

    if camera_names is None:
        return None

    camera_info = dict()
    for cam_name in camera_names:
        K = env.get_camera_intrinsic_matrix(camera_name=cam_name, camera_height=camera_height, camera_width=camera_width)
        R = env.get_camera_extrinsic_matrix(camera_name=cam_name) # camera pose in world frame
        if "eye_in_hand" in cam_name:
            # convert extrinsic matrix to be relative to robot eef control frame
            assert cam_name.startswith("robot0") or cam_name.startswith("robot1")
            robot_ind = int(cam_name[5])
            if is_v15:
                eef_site_name = env.base_env.robots[robot_ind].composite_controller.part_controllers["right"].ref_name
            else:
                eef_site_name = env.base_env.robots[robot_ind].controller.eef_name
            eef_pos = np.array(env.base_env.sim.data.site_xpos[env.base_env.sim.model.site_name2id(eef_site_name)])
            eef_rot = np.array(env.base_env.sim.data.site_xmat[env.base_env.sim.model.site_name2id(eef_site_name)].reshape([3, 3]))
            eef_pose = np.zeros((4, 4)) # eef pose in world frame
            eef_pose[:3, :3] = eef_rot
            eef_pose[:3, 3] = eef_pos
            eef_pose[3, 3] = 1.0
            eef_pose_inv = np.zeros((4, 4))
            eef_pose_inv[:3, :3] = eef_pose[:3, :3].T
            eef_pose_inv[:3, 3] = -eef_pose_inv[:3, :3].dot(eef_pose[:3, 3])
            eef_pose_inv[3, 3] = 1.0
            R = R.dot(eef_pose_inv) # T_E^W * T_W^C = T_E^C
        camera_info[cam_name] = dict(
            intrinsics=K.tolist(),
            extrinsics=R.tolist(),
        )
    return camera_info


def dataset_states_to_obs(args):
    if args.depth:
        assert len(args.camera_names) > 0, "must specify camera names if using depth"

    # Check if multi-view is enabled and auto-enable depth for SAVLA compatibility
    multi_view_config = create_config_from_args(args, args.camera_height, args.camera_width)
    if multi_view_config is not None and not args.depth:
        print("[INFO] Multi-view enabled: automatically enabling depth for SAVLA compatibility")
        args.depth = True

    # create environment to use for data processing
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    
    # Handle robosuite 1.5+ controller configuration compatibility
    # Old datasets have controller_configs in robosuite 1.4 format which won't work with 1.5+
    if is_robosuite_v15_or_later():
        if "controller_configs" in env_meta.get("env_kwargs", {}):
            # Convert old controller config to new composite format
            env_meta["env_kwargs"]["controller_configs"] = refactor_composite_controller_config(
                env_meta["env_kwargs"]["controller_configs"],
                robot_type="panda",
                arms=["right"],
            )
        else:
            # Use default composite controller config for robosuite 1.5+
            from robosuite.controllers import load_composite_controller_config
            controller_config = load_composite_controller_config(robot="Panda")
            env_meta["env_kwargs"]["controller_configs"] = controller_config
    
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=args.camera_names, 
        camera_height=args.camera_height, 
        camera_width=args.camera_width, 
        reward_shaping=args.shaped,
        use_depth_obs=args.depth,
    )

    # Wrap with multi-view if enabled
    if multi_view_config is not None:
        env = MultiViewEnvWrapper(env, multi_view_config)
        # Update camera names to include all views
        args.camera_names = env.camera_names
        print(f"Multi-view enabled with cameras: {env.camera_names}")
        # Print configuration summary with validation warnings
        env.print_config_summary()

    print("==== Using environment with the following metadata ====")
    print(json.dumps(env.serialize(), indent=4))
    print("")

    # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(args.dataset, "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[:args.n]

    # output file in same directory as input file
    output_name = args.output_name
    if output_name is None:
        if len(args.camera_names) == 0:
            output_name = os.path.basename(args.dataset)[:-5] + "_ld.hdf5"
        else:
            output_name = os.path.basename(args.dataset)[:-5] + "_im{}.hdf5".format(args.camera_width)

    output_path = os.path.join(os.path.dirname(args.dataset), output_name)
    f_out = h5py.File(output_path, "w")
    data_grp = f_out.create_group("data")
    print("input file: {}".format(args.dataset))
    print("output file: {}".format(output_path))

    total_samples = 0
    for ind in tqdm(range(len(demos))):
        ep = demos[ind]

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
            initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get("ep_meta", None)

        # extract obs, rewards, dones
        actions = f["data/{}/actions".format(ep)][()]
        if "data/{}/actions_abs".format(ep) in f:
            actions_abs = f["data/{}/actions_abs".format(ep)][()]
        else:
            actions_abs = None
        traj, camera_info = extract_trajectory(
            env=env, 
            initial_state=initial_state, 
            states=states, 
            actions=actions,
            actions_abs=actions_abs,
            done_mode=args.done_mode,
            camera_names=args.camera_names, 
            camera_height=args.camera_height, 
            camera_width=args.camera_width,
        )

        # maybe copy reward or done signal from source file
        if args.copy_rewards:
            traj["rewards"] = f["data/{}/rewards".format(ep)][()]
        if args.copy_dones:
            traj["dones"] = f["data/{}/dones".format(ep)][()]

        # store transitions

        # IMPORTANT: keep name of group the same as source file, to make sure that filter keys are
        #            consistent as well
        ep_data_grp = data_grp.create_group(ep)
        ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
        ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
        ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
        ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
        if "actions_abs" in traj:
            ep_data_grp.create_dataset("actions_abs", data=np.array(traj["actions_abs"]))
        for k in traj["obs"]:
            if args.compress:
                ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]), compression="gzip")
            else:
                ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
            if not args.exclude_next_obs:
                if args.compress:
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]), compression="gzip")
                else:
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

        # copy action dict (if applicable)
        if "data/{}/action_dict".format(ep) in f:
            action_dict = f["data/{}/action_dict".format(ep)]
            for k in action_dict:
                ep_data_grp.create_dataset("action_dict/{}".format(k), data=np.array(action_dict[k][()]))

        # episode metadata
        if is_robosuite_env:
            ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
        if "ep_meta" in f["data/{}".format(ep)].attrs:
            ep_data_grp.attrs["ep_meta"] = f["data/{}".format(ep)].attrs["ep_meta"]
        ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode

        if camera_info is not None:
            assert is_robosuite_env
            ep_data_grp.attrs["camera_info"] = json.dumps(camera_info, indent=4)

        total_samples += traj["actions"].shape[0]


    # copy over all filter keys that exist in the original hdf5
    if "mask" in f:
        f.copy("mask", f_out)

    # global metadata
    data_grp.attrs["total"] = total_samples
    data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
    print("Wrote {} trajectories to {}".format(len(demos), output_path))

    f.close()
    f_out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to input hdf5 dataset",
    )
    # name of hdf5 to write - it will be in the same directory as @dataset
    parser.add_argument(
        "--output_name",
        type=str,
        help="name of output hdf5 dataset",
    )

    # specify number of demos to process - useful for debugging conversion with a handful
    # of trajectories
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )

    # flag for reward shaping
    parser.add_argument(
        "--shaped", 
        action='store_true',
        help="(optional) use shaped rewards",
    )

    # camera names to use for observations
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=[],
        help="(optional) camera name(s) to use for image observations. Leave out to not use image observations.",
    )

    parser.add_argument(
        "--camera_height",
        type=int,
        default=256,
        help="(optional) height of image observations",
    )

    parser.add_argument(
        "--camera_width",
        type=int,
        default=256,
        help="(optional) width of image observations",
    )

    # flag for including depth observations per camera
    parser.add_argument(
        "--depth", 
        action='store_true',
        help="(optional) use depth observations for each camera",
    )

    # specifies how the "done" signal is written. If "0", then the "done" signal is 1 wherever 
    # the transition (s, a, s') has s' in a task completion state. If "1", the "done" signal 
    # is one at the end of every trajectory. If "2", the "done" signal is 1 at task completion
    # states for successful trajectories and 1 at the end of all trajectories.
    parser.add_argument(
        "--done_mode",
        type=int,
        default=0,
        help="how to write done signal. If 0, done is 1 whenever s' is a success state.\
            If 1, done is 1 at the end of each trajectory. If 2, both.",
    )

    # flag for copying rewards from source file instead of re-writing them
    parser.add_argument(
        "--copy_rewards", 
        action='store_true',
        help="(optional) copy rewards from source file instead of inferring them",
    )

    # flag for copying dones from source file instead of re-writing them
    parser.add_argument(
        "--copy_dones", 
        action='store_true',
        help="(optional) copy dones from source file instead of inferring them",
    )

    # flag to exclude next obs in dataset
    parser.add_argument(
        "--exclude-next-obs", 
        action='store_true',
        help="(optional) exclude next obs in dataset",
    )

    # flag to compress observations with gzip option in hdf5
    parser.add_argument(
        "--compress", 
        action='store_true',
        help="(optional) compress observations with gzip option in hdf5",
    )

    # Multi-view camera options (centralized in multi_view_config.py)
    add_multi_view_args(parser)

    args = parser.parse_args()
    dataset_states_to_obs(args)