"""
Utility functions for handling robosuite version compatibility.

This module provides functions to convert controller configurations between
robosuite 1.4 and 1.5+ formats, and safely access controllers across versions.
"""

import robosuite
from typing import Dict, Any, List, Optional


def get_robosuite_version_tuple() -> tuple:
    """Get robosuite version as a tuple of integers."""
    version_str = robosuite.__version__
    version_parts = version_str.split(".")
    return tuple(int(part) for part in version_parts[:2])


def is_robosuite_v15_or_later() -> bool:
    """Check if robosuite version is 1.5 or later."""
    major, minor = get_robosuite_version_tuple()
    return (major == 1 and minor >= 5) or major > 1


def refactor_composite_controller_config(
    controller_config: Dict[str, Any],
    robot_type: str = "panda",
    arms: List[str] = None,
) -> Dict[str, Any]:
    """
    Convert robosuite 1.4 controller config to robosuite 1.5+ composite controller format.

    Args:
        controller_config: Original controller configuration dict (may be old or new format)
        robot_type: Robot type (e.g., "panda", "ur5e")
        arms: List of arm names (e.g., ["right"], ["left", "right"]). Default: ["right"]

    Returns:
        Controller config in robosuite 1.5+ composite controller format

    Note:
        If the config is already in 1.5+ format (has 'body_parts' key), returns it unchanged.
        Otherwise, converts from 1.4 format to 1.5+ format.
    """
    if arms is None:
        arms = ["right"]

    # Check if already in new format (has body_parts)
    if "body_parts" in controller_config:
        # Already in 1.5+ format, return as-is
        return controller_config

    # Convert from robosuite 1.4 format to 1.5+ format
    # In 1.4, controller config was a flat dictionary with keys like:
    # - type: controller type (e.g., "OSC_POSE", "JOINT_POSITION")
    # - input_max, input_min, output_max, output_min
    # - control_delta: bool
    # - kp, kd, damping_ratio (for OSC)
    # - interpolation
    # - ramp_ratio

    old_config = controller_config.copy()
    controller_type = old_config.get("type", "OSC_POSE")

    # Build new composite controller format
    # robosuite 1.5 expects: body_parts -> right/left (flat, not nested under "arms")
    new_config = {
        "type": "BASIC",  # Add composite controller type
        "body_parts": {}
    }

    # Convert for each arm - put directly under body_parts, not under body_parts.arms
    for arm in arms:
        arm_config = {
            "type": controller_type,
            "input_max": old_config.get("input_max", 1),
            "input_min": old_config.get("input_min", -1),
            "output_max": old_config.get("output_max", [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]),
            "output_min": old_config.get("output_min", [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5]),
            "kp": old_config.get("kp", 150),
            "damping_ratio": old_config.get("damping_ratio", 1.0),
            "interpolation": old_config.get("interpolation", None),
            "ramp_ratio": old_config.get("ramp_ratio", 0.2),
        }

        # Add input_type based on control_delta
        control_delta = old_config.get("control_delta", True)
        arm_config["input_type"] = "delta" if control_delta else "absolute"

        # Add input_ref_frame if specified
        if "input_ref_frame" in old_config:
            arm_config["input_ref_frame"] = old_config["input_ref_frame"]

        # Add controller-type-specific parameters
        if controller_type in ["OSC_POSE", "OSC_POSITION"]:
            if "kd" in old_config:
                arm_config["kd"] = old_config["kd"]
            if "damping" in old_config:
                arm_config["damping"] = old_config["damping"]
        elif controller_type == "JOINT_POSITION":
            arm_config["kp"] = old_config.get("kp", 500)
            arm_config["kd"] = old_config.get("kd", 20)
            arm_config["velocity_limits"] = old_config.get("velocity_limits", [-1, 1])
            arm_config["kp_limits"] = old_config.get("kp_limits", [0, 300])

        # Add gripper config
        arm_config["gripper"] = {"type": "GRIP"}

        # Put arm config directly under body_parts (not nested under "arms")
        new_config["body_parts"][arm] = arm_config

    return new_config


def get_robot_controller(robot, arm: str = "right"):
    """
    Safely get robot controller, handling both robosuite 1.4 and 1.5+ APIs.

    Args:
        robot: Robosuite robot object
        arm: Arm name (only used for 1.5+)

    Returns:
        Controller object with goal_pos, goal_ori attributes
    """
    if is_robosuite_v15_or_later():
        # robosuite 1.5+: use composite_controller
        return robot.composite_controller.part_controllers[arm]
    else:
        # robosuite 1.4 and earlier: use simple controller
        return robot.controller


def get_robot_eef_site_name(robot, robot_ind: int = 0, arm: str = "right") -> str:
    """
    Get end-effector site name, handling both robosuite 1.4 and 1.5+ APIs.

    Args:
        robot: Robosuite robot object
        robot_ind: Robot index (for multi-robot setups)
        arm: Arm name (only used for 1.5+)

    Returns:
        End-effector site name string
    """
    if is_robosuite_v15_or_later():
        # robosuite 1.5+: use composite_controller
        return robot.composite_controller.part_controllers[arm].ref_name
    else:
        # robosuite 1.4 and earlier: use controller.eef_name
        return robot.controller.eef_name


def set_controller_config_absolute(controller_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Modify controller config to use absolute actions (not delta).
    Handles both robosuite 1.4 and 1.5+ formats.

    Args:
        controller_config: Controller configuration dict

    Returns:
        Modified controller config with absolute action control
    """
    config = controller_config.copy()

    if "body_parts" in config:
        # robosuite 1.5+ format
        body_parts = config["body_parts"]

        # Check if structure is nested (body_parts -> arms -> right/left)
        # or flat (body_parts -> right/left/torso/etc.)
        if "arms" in body_parts and isinstance(body_parts["arms"], dict):
            # Nested structure: body_parts -> arms -> right/left
            for arm_name, arm_config in body_parts["arms"].items():
                if isinstance(arm_config, dict):
                    arm_config["input_type"] = "absolute"
                    if "input_ref_frame" not in arm_config:
                        arm_config["input_ref_frame"] = "world"
        else:
            # Flat structure: body_parts -> right/left/torso/head/etc.
            # Set absolute for all controllable parts
            for part_name, part_config in body_parts.items():
                if isinstance(part_config, dict) and "input_type" in part_config:
                    part_config["input_type"] = "absolute"
                    if "input_ref_frame" not in part_config:
                        part_config["input_ref_frame"] = "world"
    else:
        # robosuite 1.4 format
        config["control_delta"] = False
        if "input_ref_frame" not in config:
            config["input_ref_frame"] = "world"

    return config
