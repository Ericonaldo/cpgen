#!/usr/bin/env python
"""
Direct TCP-replay cross-embodiment (no MimicGen planning, no physics).

For each source demo (e.g. Panda MimicGen physics data):
  1. The scene is initialized identically: the object qpos block (both nuts)
     is copied per-step straight from the source states, so the nut starts at
     the same place, falls, is carried and lands on the peg exactly as
     recorded. No physics is stepped.
  2. The target robot's TCP -- the grip site between the fingertips, NOT the
     wrist/flange -- tracks the source robot's achieved TCP trajectory
     (datagen_info/eef_pose) pose-for-pose.
  3. Joints come from warm-started damped-least-squares IK with null-space
     anchoring, shortest-arc rotation errors and a per-step rate clamp, so
     consecutive frames stay in the same IK branch (no elbow/wrist jumps).
  4. Fingers are mapped per-gripper (open/closed-on-nut qpos, measured from
     physics demos) with a rate-limited closing profile.

Output demos are 1:1 paired with the source (demo_i <-> demo_i): same scene,
same TCP motion, different embodiment.

Self-contained apart from demo_aug.rs15_compat (robosuite 1.5 env compat).

Usage (from the cpgen repo root):
    MUJOCO_GL=egl python demo_aug/replay_tcp_cross_embodiment.py \
        --source datasets/square_d0/panda_physics_200/demo.hdf5 \
        --output datasets/square_d0/tcp_replay_sawyer/demo.hdf5 \
        --robot Sawyer --gripper RethinkGripper
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))  # cpgen repo root

from demo_aug.rs15_compat import apply_all

apply_all()

import mujoco
import robosuite.utils.transform_utils as T

import mimicgen.utils.robomimic_utils as RobomimicUtils
from robomimic.utils.file_utils import get_env_metadata_from_dataset


# --------------------------------------------------------------------------
# gripper finger mapping
# --------------------------------------------------------------------------

GRIPPER_QPOS = {
    # (open finger qpos, closed-on-nut-handle finger qpos) per gripper.
    # Closed values are measured from physics demos where the fingers rest on
    # the nut handle, so the closed gripper looks like an actual grasp.
    "PandaGripper": (np.array([0.020833, -0.020833]), np.array([0.011, -0.011])),
    # Measured from the rs1.4 Sawyer transfer dataset (states[:, 8:10] while
    # actions[:, 6] > 0): fingers cross zero and squeeze slightly inward.
    "RethinkGripper": (np.array([0.020833, -0.020833]), np.array([-0.0047, 0.0045])),
}

GRIPPER_CLOSE_STEPS = 8
"""Steps for a full open<->close finger travel (physics grippers take ~7-9
control steps; teleporting in one frame reads as 'the gripper never closes')."""


def gripper_action_to_qpos(gripper_action: np.ndarray, gripper_name: str) -> np.ndarray:
    """Map normalized gripper action (-1 open, 1 close) to target finger qpos."""
    action = float(np.asarray(gripper_action).reshape(-1)[0])
    open_qpos, closed_qpos = GRIPPER_QPOS[gripper_name]
    t = (action + 1.0) * 0.5  # 0=open, 1=close
    return open_qpos * (1.0 - t) + closed_qpos * t


def step_gripper_qpos(
    current: Optional[np.ndarray],
    gripper_action: np.ndarray,
    gripper_name: str,
) -> np.ndarray:
    """Advance fingers one control step toward the commanded target, rate-limited
    to mimic the physical closing motion."""
    target = gripper_action_to_qpos(gripper_action, gripper_name)
    if current is None:
        return target
    open_qpos, closed_qpos = GRIPPER_QPOS[gripper_name]
    rate = np.abs(closed_qpos - open_qpos) / GRIPPER_CLOSE_STEPS
    return current + np.clip(target - current, -rate, rate)


# --------------------------------------------------------------------------
# continuity-preserving IK
# --------------------------------------------------------------------------

ARM_QPOS_RATE = 0.35
"""Max arm joint change per control step (rad). Physics arms move <0.16
rad/joint/step on this task; the clamp is a last-resort guard so a bad IK
solve can never teleport the arm within one frame (warm-started IK recovers
from the clamped configuration on subsequent steps)."""


def _ik_descend(
    sim,
    site_id: int,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    qpos_start: np.ndarray,
    max_iters: int,
    pos_tol: float,
    rot_tol: float,
) -> Tuple[np.ndarray, float, bool]:
    """One damped-least-squares descent from a single seed.

    The redundant DOF is anchored to the seed configuration (null-space
    regularization toward @qpos_start), so consecutive solves starting from
    the previous solution stay in the same IK branch instead of drifting or
    flipping the elbow/wrist while the EE pose barely changes.

    Returns (best_arm_qpos, best_cost, converged).
    """
    model = sim.model
    jnt_range = model.jnt_range[:7]
    rot_weight = 0.5
    qpos = qpos_start.copy()
    q_anchor = qpos_start[:7].copy()

    best_qpos = qpos[:7].copy()
    best_cost = np.inf

    for _ in range(max_iters):
        sim.data.qpos[:] = qpos
        sim.forward()

        curr_pos = sim.data.site_xpos[site_id].copy()
        curr_quat = T.mat2quat(sim.data.site_xmat[site_id].reshape(3, 3))
        pos_err = target_pos - curr_pos
        quat_err = T.quat_multiply(target_quat, T.quat_inverse(curr_quat))
        if quat_err[3] < 0:
            # canonicalize double cover: always take the shortest arc,
            # otherwise the rotation error points the long way around and the
            # wrist spins toward a 2*pi-wrapped solution
            quat_err = -quat_err
        rot_err = T.quat2axisangle(quat_err)

        pos_norm = np.linalg.norm(pos_err)
        rot_norm = np.linalg.norm(rot_err)
        cost = pos_norm + rot_weight * rot_norm
        if cost < best_cost:
            best_cost = cost
            best_qpos = qpos[:7].copy()
        if pos_norm < pos_tol and rot_norm < rot_tol:
            return best_qpos, best_cost, True

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model._model, sim.data._data, jacp, jacr, site_id)
        J = np.vstack([jacp[:, :7], rot_weight * jacr[:, :7]])
        err = np.concatenate([pos_err, rot_weight * rot_err])

        damping = 1e-3
        continuity = 0.02  # null-space pull toward the seed configuration
        dq = np.linalg.solve(
            J.T @ J + (damping + continuity) * np.eye(7),
            J.T @ err + continuity * (q_anchor - qpos[:7]),
        )

        step = min(1.0, 0.3 / (np.linalg.norm(dq) + 1e-8))
        qpos[:7] += step * dq
        qpos[:7] = np.clip(qpos[:7], jnt_range[:, 0] + 1e-4, jnt_range[:, 1] - 1e-4)

    return best_qpos, best_cost, False


def ik_solve(
    sim,
    site_id: int,
    target_pose: np.ndarray,
    qpos_full: np.ndarray,
    restart_seeds: Optional[List[np.ndarray]] = None,
    max_iters: int = 100,
    pos_tol: float = 1e-3,
    rot_tol: float = 2e-2,
    restart_cost: float = 0.05,
) -> Tuple[np.ndarray, bool]:
    """
    Damped-least-squares IK, warm-started from the current configuration.

    Joint continuity matters more than the last millimetre: restart seeds are
    only tried when the warm-started descent is badly stuck (cost above
    @restart_cost), because a restart can land in a different IK branch and
    cause a visible one-frame arm jump.
    Returns (arm_qpos[7], converged).
    """
    target_pos = target_pose[:3, 3]
    target_quat = T.mat2quat(target_pose[:3, :3])

    q_prev = qpos_full[:7].copy()
    best_qpos, best_cost, converged = _ik_descend(
        sim, site_id, target_pos, target_quat, qpos_full, max_iters, pos_tol, rot_tol
    )
    if converged or best_cost < restart_cost:
        return best_qpos, converged

    # Warm start badly stuck: try restart seeds, but among acceptable
    # candidates prefer the one closest to the previous configuration so a
    # rescue does not flip the arm into a distant IK branch.
    candidates = [(best_cost, float(np.linalg.norm(best_qpos - q_prev)), best_qpos, converged)]
    for seed in restart_seeds or []:
        qpos_seeded = qpos_full.copy()
        qpos_seeded[:7] = seed
        cand_qpos, cand_cost, cand_conv = _ik_descend(
            sim, site_id, target_pos, target_quat, qpos_seeded, max_iters, pos_tol, rot_tol
        )
        candidates.append(
            (cand_cost, float(np.linalg.norm(cand_qpos - q_prev)), cand_qpos, cand_conv)
        )

    acceptable = [c for c in candidates if c[0] < restart_cost or c[3]]
    if acceptable:
        _, _, best_qpos, converged = min(acceptable, key=lambda c: c[1])
    else:
        _, _, best_qpos, converged = min(candidates, key=lambda c: c[0])
    return best_qpos, converged


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------

def replay_demo(env, base_env, src, gripper_name):
    """Replay one source demo kinematically; returns per-step records."""
    states_src = src["states"]          # (T, 45)
    eef_src = src["eef_pose"]           # (T, 4, 4)
    actions_src = src["actions"]        # (T, 7)
    T_len = len(actions_src)

    env.reset()
    # robosuite hard resets rebuild the sim object; grab references afterwards
    sim = base_env.sim
    site_id = sim.model.site_name2id(base_env.robots[0].controller.ref_name)
    template = env.get_state()["states"].copy()

    qpos_full = sim.data.qpos.copy()
    gripper_qpos = None
    ik_fail = 0

    states, observations = [], []
    for t in range(T_len):
        arm_qpos, ok = ik_solve(
            sim=sim,
            site_id=site_id,
            target_pose=eef_src[t],
            qpos_full=qpos_full,
            restart_seeds=[qpos_full[:7]],
            # t=0 places the arm at the source's start TCP from the target
            # robot's home pose (a long move); give the solver more budget
            max_iters=300 if t == 0 else 100,
        )
        if not ok:
            ik_fail += 1
        if t > 0:
            # continuity clamp only applies within the trajectory; the t=0
            # pose is scene initialization (place the arm at the source's
            # starting TCP directly, no catch-up transient)
            arm_qpos = qpos_full[:7] + np.clip(
                arm_qpos - qpos_full[:7], -ARM_QPOS_RATE, ARM_QPOS_RATE
            )
        gripper_qpos = step_gripper_qpos(gripper_qpos, actions_src[t, 6:7], gripper_name)

        state = template.copy()
        state[1:8] = arm_qpos
        state[8:10] = gripper_qpos
        state[10:24] = states_src[t, 10:24]  # both nuts, verbatim from source
        state[24:45] = 0.0                   # velocities
        sim.set_state_from_flattened(state)
        sim.forward()
        qpos_full = sim.data.qpos.copy()

        states.append(env.get_state()["states"])
        observations.append(env.get_observation())

    success = bool(env.is_success().get("task", False))
    return dict(
        states=np.array(states),
        observations=observations,
        actions=np.array(actions_src),
        success=success,
        ik_fail=ik_fail,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Source hdf5 (with datagen_info/eef_pose)")
    parser.add_argument("--output", required=True, help="Output demo.hdf5")
    parser.add_argument("--robot", default="Sawyer")
    parser.add_argument("--gripper", default="RethinkGripper")
    parser.add_argument("--n_demos", type=int, default=None, help="default: all")
    parser.add_argument("--camera_names", nargs="+", default=["agentview", "robot0_eye_in_hand"])
    parser.add_argument("--camera_height", type=int, default=84)
    parser.add_argument("--camera_width", type=int, default=84)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    assert args.gripper in GRIPPER_QPOS, f"add {args.gripper} to GRIPPER_QPOS first"

    env_meta = get_env_metadata_from_dataset(dataset_path=args.source)
    env = RobomimicUtils.create_env(
        env_meta=env_meta,
        env_class=None,
        env_name=env_meta["env_name"],
        robot=args.robot,
        gripper=args.gripper,
        env_meta_update_kwargs={"controller_configs": {"control_delta": True}},
        camera_names=args.camera_names,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        render=False,
        render_offscreen=True,
        use_image_obs=True,
        use_depth_obs=False,
    )
    base_env = env.base_env if hasattr(env, "base_env") else env.env

    f_src = h5py.File(args.source, "r")
    demo_keys = sorted(
        [k for k in f_src["data"].keys() if k.startswith("demo_")],
        key=lambda k: int(k.split("_")[1]),
    )
    if args.n_demos is not None:
        demo_keys = demo_keys[: args.n_demos]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    env_meta_out = json.loads(json.dumps(env_meta))
    env_meta_out["env_kwargs"]["robots"] = [args.robot]
    env_meta_out["env_kwargs"]["gripper_types"] = [args.gripper]

    t0 = time.time()
    n_success, total_ik_fail = 0, 0
    with h5py.File(args.output, "w") as f_out:
        data_grp = f_out.create_group("data")
        data_grp.attrs["env_args"] = json.dumps(env_meta_out)
        data_grp.attrs["type"] = 1
        data_grp.attrs["env"] = "robosuite"
        data_grp.attrs["generation_method"] = "tcp_replay_cross_embodiment"
        data_grp.attrs["source_dataset"] = str(args.source)

        total = 0
        for i, dk in enumerate(demo_keys):
            g_src = f_src[f"data/{dk}"]
            src = dict(
                states=g_src["states"][:],
                eef_pose=g_src["datagen_info"]["eef_pose"][:],
                actions=g_src["actions"][:],
            )
            result = replay_demo(env, base_env, src, args.gripper)
            n_success += int(result["success"])
            total_ik_fail += result["ik_fail"]

            ep = data_grp.create_group(dk)
            ep.attrs["num_samples"] = len(result["actions"])
            ep.attrs["success"] = result["success"]
            total += len(result["actions"])
            ep.create_dataset("states", data=result["states"])
            ep.create_dataset("actions", data=result["actions"])
            ep.create_dataset("rewards", data=np.ones(len(result["actions"])))
            ep.create_dataset("dones", data=np.zeros(len(result["actions"]), dtype=bool))
            obs_grp = ep.create_group("obs")
            for key in result["observations"][0].keys():
                obs_grp.create_dataset(
                    key, data=np.array([o[key] for o in result["observations"]])
                )

            if (i + 1) % 10 == 0:
                logging.info(
                    "%d/%d demos (success %d, ik non-converged steps %d)",
                    i + 1, len(demo_keys), n_success, total_ik_fail,
                )

        data_grp.attrs["total"] = total
        f_out.create_dataset(
            "mask/train", data=np.array([k.encode() for k in demo_keys])
        )
    f_src.close()

    stats = dict(
        output=args.output,
        n_demos=len(demo_keys),
        n_success=n_success,
        ik_nonconverged_steps=total_ik_fail,
        elapsed_sec=round(time.time() - t0, 1),
    )
    logging.info("TCP replay done: %s", stats)


if __name__ == "__main__":
    main()
