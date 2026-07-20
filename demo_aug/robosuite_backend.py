"""Backend-neutral helpers for the pinned robosuite 1.4 runtime."""

from contextlib import contextmanager
from copy import deepcopy

import numpy as np
from robosuite.controllers import load_controller_config
from scipy.spatial.transform import Rotation

_COMPOSITE_ARM_KEYS = (
    "input_max",
    "input_min",
    "output_max",
    "output_min",
    "kp",
    "impedance_mode",
    "kp_limits",
    "position_limits",
    "orientation_limits",
    "uncouple_pos_ori",
    "interpolation",
    "ramp_ratio",
)


@contextmanager
def managed_mujoco_renderer(factory, model):
    renderer = factory(model)
    try:
        yield renderer
    finally:
        close = getattr(renderer, "close", None)
        if close is not None:
            close()
        else:
            renderer._mjr_context.free()
            renderer._gl_context.free()


def _legacy_osc_config(controller_config):
    if "body_parts" not in controller_config:
        controller_type = controller_config.get("type")
        if controller_type != "OSC_POSE":
            raise ValueError(
                "only OSC_POSE controller metadata is supported, got {!r}".format(
                    controller_type
                )
            )
        legacy = deepcopy(controller_config)
        for old_key, canonical_key in (
            ("damping", "damping_ratio"),
            ("damping_limits", "damping_ratio_limits"),
        ):
            if old_key not in legacy:
                continue
            if canonical_key in legacy and legacy[canonical_key] != legacy[old_key]:
                raise ValueError(
                    "conflicting damping metadata for {} and {}".format(
                        old_key, canonical_key
                    )
                )
            legacy[canonical_key] = legacy.pop(old_key)
        return legacy
    body_parts = controller_config["body_parts"]
    if controller_config.get("type") != "BASIC":
        raise ValueError("composite controller type must be BASIC")
    if not isinstance(body_parts, dict) or not isinstance(
        body_parts.get("right"), dict
    ):
        raise ValueError("composite controller metadata must define body_parts.right")

    arm = body_parts["right"]
    controller_type = arm.get("type")
    if controller_type != "OSC_POSE":
        raise ValueError(
            "only OSC_POSE controller metadata is supported, got {!r}".format(
                controller_type
            )
        )
    input_type = arm.get("input_type")
    if input_type not in ("delta", "absolute"):
        raise ValueError(
            "OSC_POSE input_type must be 'delta' or 'absolute', got {!r}".format(
                input_type
            )
        )
    if arm.get("input_ref_frame") != "world":
        raise ValueError("OSC_POSE input_ref_frame must be 'world'")

    legacy = {"type": "OSC_POSE", "control_delta": input_type == "delta"}
    for key in _COMPOSITE_ARM_KEYS:
        if key in arm:
            legacy[key] = deepcopy(arm[key])
    if "damping_ratio" in arm:
        legacy["damping_ratio"] = deepcopy(arm["damping_ratio"])
    if "damping_ratio_limits" in arm:
        legacy["damping_ratio_limits"] = deepcopy(arm["damping_ratio_limits"])
    return legacy


def normalize_env_meta_for_runtime(env_meta, *, robot=None, gripper=None):
    """Return an independent robosuite 1.4-compatible metadata dictionary."""
    if not isinstance(env_meta, dict):
        raise ValueError("env_meta must be a dictionary")
    normalized = deepcopy(env_meta)
    env_kwargs = normalized.get("env_kwargs")
    if not isinstance(env_kwargs, dict):
        raise ValueError("env_meta.env_kwargs must be a dictionary")

    controller_config = env_kwargs.get("controller_configs")
    if controller_config is not None:
        if not isinstance(controller_config, dict):
            raise ValueError("controller_configs must be a dictionary")
        env_kwargs["controller_configs"] = _legacy_osc_config(controller_config)
    if robot is not None:
        if not isinstance(robot, str) or not robot:
            raise ValueError("robot must be a non-empty string")
        env_kwargs["robots"] = [robot]
    if gripper is not None:
        if not isinstance(gripper, str) or not gripper:
            raise ValueError("gripper must be a non-empty string")
        env_kwargs["gripper_types"] = [gripper]
    return normalized


def _load_controller_config(*, default_controller):
    return load_controller_config(default_controller=default_controller)


def controller_config_for_runtime(*, robot: str, mode: str):
    """Load the native legacy OSC controller for delta or absolute targets."""
    if not isinstance(robot, str) or not robot:
        raise ValueError("robot must be a non-empty string")
    if mode not in ("delta", "absolute"):
        raise ValueError("mode must be 'delta' or 'absolute'")
    loaded = _load_controller_config(default_controller="OSC_POSE")
    if not isinstance(loaded, dict) or loaded.get("type") != "OSC_POSE":
        raise ValueError("runtime controller must be OSC_POSE")
    config = deepcopy(loaded)
    config["control_delta"] = mode == "delta"
    return config


def resolve_cpgen_controller_for_rs14(*, robot: str, requested: str):
    """Resolve the CPGen CLI controller name for the rs1.4 runtime.

    CPGen emits absolute world-frame EEF poses. The historical ``ik`` CLI
    name selected rs1.5 WHOLE_BODY_IK, which rs1.4 cannot reproduce. It remains
    as an explicit compatibility alias for native absolute OSC; requests for
    the actual WHOLE_BODY_IK controller fail closed.
    """
    if requested not in ("default", "ik"):
        if requested == "WHOLE_BODY_IK":
            raise ValueError("WHOLE_BODY_IK is not supported by the rs1.4 runtime")
        raise ValueError("unsupported CPGen controller {!r}".format(requested))
    return controller_config_for_runtime(robot=robot, mode="absolute")


def convert_frame_name_to_rs14(frame_name):
    """Translate the known Panda frame names emitted by the rs1.5 source stack."""
    aliases = {
        "gripper0_right_rightfinger": "gripper0_rightfinger",
        "gripper0_right_leftfinger": "gripper0_leftfinger",
        "gripper0_right_right_gripper": "gripper0_right_gripper",
        "gripper0_right_grip_site": "gripper0_grip_site",
        "gripper0_right_eef": "gripper0_eef",
    }
    return aliases.get(frame_name, frame_name)


def site_rotation_to_xyzw_quaternion(site_rotation):
    rotation = np.asarray(site_rotation, dtype=float).reshape(3, 3)
    return Rotation.from_matrix(rotation).as_quat()


def eef_body_name_for_rs14(eef_name, arm=None):
    if isinstance(eef_name, str):
        return eef_name
    if arm is None or arm not in eef_name:
        raise ValueError("composite EEF names require a selected arm")
    return eef_name[arm]


def eef_site_name_for_rs14(robot):
    controller = getattr(robot, "controller", None)
    eef_name = getattr(controller, "eef_name", None)
    if not isinstance(eef_name, str) or not eef_name:
        raise ValueError("rs1.4 robot controller must define a non-empty eef_name")
    return eef_name


def resolve_mujoco_model(robot_model, live_model):
    return getattr(robot_model, "mujoco_model", live_model)
