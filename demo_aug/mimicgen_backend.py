"""MimicGen generation on the pinned robosuite 1.4 runtime."""

import json
import logging
import pathlib
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional

import h5py
import mimicgen
from mimicgen.configs import config_factory
from mimicgen.scripts.generate_dataset import generate_dataset
from mimicgen.utils import robomimic_utils
from mimicgen.utils.misc_utils import deep_update

from demo_aug.robosuite_backend import normalize_env_meta_for_runtime


@dataclass
class MimicgenConfig:
    name: Optional[str] = None
    robot: Optional[str] = None
    gripper: Optional[str] = None
    config_json: Optional[pathlib.Path] = None
    select_src_per_subtask: Optional[bool] = None
    action_noise: Optional[float] = None
    keep_failed: bool = True
    render_video: bool = True
    camera_names: List[str] = field(
        default_factory=lambda: ["agentview", "robot0_eye_in_hand"]
    )
    camera_height: int = 84
    camera_width: int = 84


_GENERIC_DATASET_STEMS = frozenset(
    {"demo", "merged_demos", "demo_failed", "demo_rs14_src"}
)
# MimicGen task templates are keyed by task family, not env variant (D0/D1).
_ENV_NAME_TO_TEMPLATE = {
    "Square_D0": "square",
    "Square_D1": "square",
}


def _read_env_name_from_dataset(demo_path):
    with h5py.File(demo_path, "r") as f:
        return json.loads(f["data"].attrs["env_args"]).get("env_name")


def _template_stem_for_dataset(demo_path, env_name=None):
    stem = pathlib.Path(demo_path).stem
    if stem not in _GENERIC_DATASET_STEMS:
        return stem
    env_name = env_name or _read_env_name_from_dataset(demo_path)
    if env_name in _ENV_NAME_TO_TEMPLATE:
        return _ENV_NAME_TO_TEMPLATE[env_name]
    raise FileNotFoundError(
        f"Cannot infer MimicGen task template for generic dataset {demo_path} "
        f"(env_name={env_name}). Pass --config-json with a full MimicGen config."
    )


def _find_task_template(demo_path, env_name=None):
    """Locate the MimicGen task template json for a source dataset."""
    template_dir = pathlib.Path(mimicgen.__path__[0]) / "exps/templates/robosuite"
    template_stem = _template_stem_for_dataset(demo_path, env_name=env_name)
    template = template_dir / (template_stem + ".json")
    if not template.exists():
        available = sorted(p.stem for p in template_dir.glob("*.json"))
        raise FileNotFoundError(
            f"No MimicGen task template named '{template.name}' for source dataset "
            f"{demo_path}. Available templates: {available}. Rename the dataset to "
            f"match a template, or pass --cfg.mimicgen.config-json with a full "
            f"MimicGen config."
        )
    return template


_ENV_TO_INTERFACE = {
    "Square_D0": ("MG_Square", "robosuite"),
    "Square_D1": ("MG_Square", "robosuite"),
}


def annotate_datagen_interface(dataset_path, interface=None, interface_type=None):
    """Stamp env-interface attrs onto datagen_info if missing (generated datasets)."""
    with h5py.File(dataset_path, "r+") as f:
        if interface is None:
            env_name = json.loads(f["data"].attrs["env_args"]).get("env_name")
            if env_name not in _ENV_TO_INTERFACE:
                raise ValueError(
                    f"Cannot infer interface for env_name={env_name}; "
                    f"set experiment.task.interface in the MimicGen config."
                )
            interface, interface_type = _ENV_TO_INTERFACE[env_name]

        for demo_key in sorted(k for k in f["data"].keys() if k.startswith("demo_")):
            dg = f[f"data/{demo_key}/datagen_info"]
            if dg.attrs.get("env_interface_name") != interface:
                dg.attrs["env_interface_name"] = interface
            if dg.attrs.get("env_interface_type") != interface_type:
                dg.attrs["env_interface_type"] = interface_type


@contextmanager
def _normalized_create_env_boundary():
    """Normalize composite metadata immediately before upstream env creation."""
    original = robomimic_utils.create_env

    def create_env(*args, **kwargs):
        if args:
            raise TypeError("MimicGen create_env boundary requires keyword arguments")
        env_meta = deepcopy(kwargs["env_meta"])
        env_name = kwargs.get("env_name")
        if env_name is not None:
            env_meta["env_name"] = env_name
        updates = kwargs.get("env_meta_update_kwargs")
        if updates:
            deep_update(env_meta, updates)
        kwargs["env_meta"] = normalize_env_meta_for_runtime(
            env_meta,
            robot=kwargs.get("robot"),
            gripper=kwargs.get("gripper"),
        )
        kwargs["env_name"] = None
        kwargs["robot"] = None
        kwargs["gripper"] = None
        kwargs["env_meta_update_kwargs"] = {}
        return original(**kwargs)

    robomimic_utils.create_env = create_env
    try:
        yield
    finally:
        robomimic_utils.create_env = original


def run_mimicgen_generation(cfg):
    """Run MimicGen data generation driven by a demo_aug.generate.Config."""
    mg = cfg.mimicgen
    base_json = mg.config_json or _find_task_template(
        cfg.demo_path, env_name=cfg.env_name
    )
    with open(base_json) as f:
        dic = json.load(f)

    exp = dic["experiment"]
    exp["source"]["dataset_path"] = str(cfg.demo_path)
    exp["source"]["start"] = cfg.load_demos_start_idx or None
    exp["source"]["n"] = (
        None
        if cfg.load_demos_end_idx is None
        else cfg.load_demos_end_idx - (cfg.load_demos_start_idx or 0)
    )
    exp["task"]["name"] = cfg.env_name
    exp["task"]["robot"] = mg.robot
    exp["task"]["gripper"] = mg.gripper
    exp["generation"]["path"] = str(cfg.save_dir)
    exp["generation"]["num_trials"] = cfg.n_demos
    exp["generation"]["guarantee"] = cfg.require_n_demos
    exp["generation"]["keep_failed"] = mg.keep_failed
    if mg.select_src_per_subtask is not None:
        exp["generation"]["select_src_per_subtask"] = mg.select_src_per_subtask
    exp["seed"] = cfg.seed
    exp["render_video"] = mg.render_video
    exp["name"] = mg.name or "_".join(
        filter(None, [cfg.env_name, mg.robot, mg.gripper, "mimicgen"])
    )

    dic["obs"]["camera_names"] = list(mg.camera_names)
    dic["obs"]["camera_height"] = mg.camera_height
    dic["obs"]["camera_width"] = mg.camera_width

    if mg.action_noise is not None:
        for subtask in dic["task"]["task_spec"].values():
            subtask["action_noise"] = mg.action_noise

    if cfg.debug:
        exp["source"]["n"] = 3
        exp["generation"]["num_trials"] = 2
        exp["generation"]["guarantee"] = False

    mg_config = config_factory(dic["name"], dic["type"], dic=dic)
    logging.info(
        f"Running MimicGen generation: task={cfg.env_name} robot={mg.robot} "
        f"gripper={mg.gripper} -> {cfg.save_dir}/{exp['name']}"
    )
    with _normalized_create_env_boundary():
        important_stats = generate_dataset(mg_config, auto_remove_exp=True)
    if important_stats is not None:
        logging.info(
            f"MimicGen generation stats: {json.dumps(important_stats, indent=2)}"
        )
        output_demo = pathlib.Path(important_stats["generation_path"]) / "demo.hdf5"
        if important_stats.get("num_success", 0) > 0 and output_demo.exists():
            annotate_datagen_interface(str(output_demo))
    return important_stats
