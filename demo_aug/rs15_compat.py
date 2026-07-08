"""Compatibility layer for running MimicGen data generation on the cpgen stack
(robosuite 1.5.x + mujoco 3.2.6).

MimicGen itself targets robosuite 1.4; three gaps are bridged here:

1. Environment registration: mimicgen's env classes fail to import under
   robosuite 1.5. `cpgen_envs` ships 1.5-compatible re-implementations and
   registers them, but it also monkey-patches the Sawyer/RethinkGripper model
   paths to XML files that do not exist in the cpgen-envs repo. We restore the
   stock robosuite models (sanitized for mujoco 3.2.6 where needed).

2. Controller config: source datasets store robosuite<=1.4 style OSC_POSE
   configs, which robosuite 1.5 cannot load. We convert them to the 1.5
   composite-controller format on env creation (delta OSC in world frame,
   matching the 1.4 controller convention that mimicgen assumes).

3. Controller access: mimicgen's env interfaces use `robot.controller`
   (removed in 1.5) for the eef site name and action scaling bounds. We expose
   a `controller` property returning the right-arm part controller and point
   the eef pose lookup at its `ref_name` site.
"""

import os
import pathlib
import xml.etree.ElementTree as ET
from copy import deepcopy


def register_mimicgen_envs():
    """Register mimicgen task envs (Square_D1 etc.) for robosuite 1.5 via cpgen_envs."""
    import cpgen_envs  # noqa: F401


def _sanitize_robot_xml(robot_name):
    """Return a path to a mujoco-3.2.6-loadable copy of a stock robosuite robot XML.

    robosuite 1.5 ships some robot XMLs (sawyer, baxter) using the mesh attribute
    inertia="shell", which requires mujoco>=3.3. Strip it and absolutize asset
    paths so the copy can live outside the robosuite assets tree.
    """
    from robosuite.utils.mjcf_utils import xml_path_completion

    src = xml_path_completion(f"robots/{robot_name}/robot.xml")
    with open(src) as f:
        content = f.read()
    if 'inertia="shell"' not in content:
        return src

    src_dir = os.path.dirname(src)
    tree = ET.parse(src)
    for elem in tree.getroot().iter():
        if elem.tag == "mesh" and "inertia" in elem.attrib:
            del elem.attrib["inertia"]
        file_attr = elem.get("file")
        if file_attr and not os.path.isabs(file_attr):
            elem.set("file", os.path.join(src_dir, file_attr))

    out_dir = pathlib.Path("/tmp/rs15_compat_models") / robot_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "robot.xml"
    tree.write(out_path)
    return str(out_path)


def fix_robot_models():
    """Undo cpgen_envs' broken Sawyer/RethinkGripper model-path patches.

    cpgen_envs.environments.manipulation.nut_assembly points these models at
    XMLs missing from the cpgen-envs repo; restore stock robosuite models,
    sanitizing the Sawyer XML for mujoco 3.2.6.
    """
    import cpgen_envs  # noqa: F401  (must run first so we override its patches)
    from robosuite.models.grippers.gripper_model import GripperModel
    from robosuite.models.grippers.rethink_gripper import RethinkGripperBase
    from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
    from robosuite.models.robots.manipulators.sawyer_robot import Sawyer
    from robosuite.utils.mjcf_utils import xml_path_completion

    sawyer_xml = _sanitize_robot_xml("sawyer")

    def _sawyer_init(self, idn=0):
        ManipulatorModel.__init__(self, sawyer_xml, idn=idn)

    def _rethink_init(self, idn=None):
        GripperModel.__init__(self, xml_path_completion("grippers/rethink_gripper.xml"), idn=idn)

    Sawyer.__init__ = _sawyer_init
    RethinkGripperBase.__init__ = _rethink_init

    # cpgen_envs sets SquareNutObject.bottom_offset to [0, 0, 0.01], which makes
    # the nut spawn already settled on the table (z=0.83) at reset. MimicGen's
    # source datasets record the t=0 object pose mid-air (z=0.89, robosuite 1.4
    # convention: bottom_offset [0, 0, -0.05]), and its object-centric waypoint
    # transform `cur_obj_pose @ inv(src_obj_pose)` then shifts every waypoint
    # 6 cm into the table. Restore the stock offset for data generation.
    import numpy as np
    from robosuite.models.objects import SquareNutObject

    SquareNutObject.bottom_offset = np.array([0, 0, -0.05])


def convert_legacy_osc_config(legacy_cfg, robot):
    """Convert a robosuite<=1.4 OSC_POSE controller config dict to the 1.5
    composite-controller format, preserving the action scaling parameters."""
    from robosuite.controllers import load_composite_controller_config

    assert legacy_cfg.get("type", "OSC_POSE") == "OSC_POSE", (
        f"only OSC_POSE legacy configs are supported, got {legacy_cfg.get('type')}"
    )
    composite = load_composite_controller_config(robot=robot)
    arm = composite["body_parts"]["right"]
    carry_over = (
        "input_max", "input_min", "output_max", "output_min", "kp", "damping_ratio",
        "impedance_mode", "kp_limits", "damping_ratio_limits", "position_limits",
        "orientation_limits", "uncouple_pos_ori", "interpolation", "ramp_ratio",
    )
    for key in carry_over:
        if key in legacy_cfg:
            arm[key] = legacy_cfg[key]
    # very old robomimic datasets use pre-1.4 key names for damping
    renamed = {"damping": "damping_ratio", "damping_limits": "damping_ratio_limits"}
    for old_key, new_key in renamed.items():
        if old_key in legacy_cfg and new_key not in legacy_cfg:
            arm[new_key] = legacy_cfg[old_key]
    arm["input_type"] = "delta" if legacy_cfg.get("control_delta", True) else "absolute"
    # mimicgen's target_pose_to_action computes deltas in the world frame
    # (robosuite 1.4 convention); 1.5 defaults to the base frame.
    arm["input_ref_frame"] = "world"
    return composite


def _convert_env_meta_inplace(env_meta, robot=None):
    controller_cfg = env_meta.get("env_kwargs", {}).get("controller_configs")
    if controller_cfg is not None and "body_parts" not in controller_cfg:
        robots = env_meta["env_kwargs"].get("robots", "Panda")
        default_robot = robots[0] if isinstance(robots, (list, tuple)) else robots
        env_meta["env_kwargs"]["controller_configs"] = convert_legacy_osc_config(
            controller_cfg, robot or default_robot
        )


def patch_mimicgen_create_env():
    """Make env creation convert legacy controller configs on the fly.

    Patches both mimicgen's create_env (knows the robot override) and robomimic's
    create_env_for_data_processing (used directly by e.g. prepare_src_dataset.py
    and dataset_states_to_obs.py).
    """
    import mimicgen.utils.robomimic_utils as RobomimicUtils
    import robomimic.utils.env_utils as EnvUtils

    orig_create_env = RobomimicUtils.create_env

    def create_env(env_meta, robot=None, **kwargs):
        env_meta = deepcopy(env_meta)
        _convert_env_meta_inplace(env_meta, robot=robot)
        return orig_create_env(env_meta=env_meta, robot=robot, **kwargs)

    RobomimicUtils.create_env = create_env

    orig_cedp = EnvUtils.create_env_for_data_processing

    def create_env_for_data_processing(env_meta, **kwargs):
        env_meta = deepcopy(env_meta)
        _convert_env_meta_inplace(env_meta)
        return orig_cedp(env_meta=env_meta, **kwargs)

    EnvUtils.create_env_for_data_processing = create_env_for_data_processing


def patch_mimicgen_env_interfaces():
    """Adapt mimicgen env interfaces to the robosuite 1.5 controller API."""
    from robosuite.controllers.parts.arm.osc import OperationalSpaceController
    from robosuite.robots.robot import Robot

    # robosuite 1.5's composite controller reads the LAST action dimension as a
    # mobile-base mode flag and, when it is > 0, switches arm delta actions to
    # accumulate on the previous desired goal instead of the achieved pose. For
    # fixed-base single-arm robots the last dimension is the GRIPPER action, so
    # closing the gripper silently changes the delta-OSC convention away from
    # the robosuite 1.4 one that mimicgen's target_pose_to_action assumes.
    # Force the 1.4 behavior ("achieved") at all times.
    OperationalSpaceController.set_goal_update_mode = lambda self, goal_update_mode: None

    if not hasattr(Robot, "controller"):
        Robot.controller = property(
            lambda self: self.composite_controller.part_controllers["right"]
        )

    from mimicgen.env_interfaces.robosuite import RobosuiteInterface

    def get_robot_eef_pose(self):
        # 1.5 part controllers call the control frame site `ref_name`
        # (1.4 called it `eef_name`)
        return self.get_object_pose(
            obj_name=self.env.robots[0].controller.ref_name,
            obj_type="site",
        )

    RobosuiteInterface.get_robot_eef_pose = get_robot_eef_pose


def apply_all():
    register_mimicgen_envs()
    fix_robot_models()
    patch_mimicgen_create_env()
    patch_mimicgen_env_interfaces()
