import pathlib
from dataclasses import dataclass, field
from typing import Optional

import cpgen_envs  # noqa: F401 - registers CPGen target environments
import tyro

from demo_aug.mimicgen_backend import MimicgenConfig, run_mimicgen_generation


@dataclass
class Config:
    demo_path: pathlib.Path
    save_dir: pathlib.Path
    env_name: str = "Square_D0"
    n_demos: int = 1
    require_n_demos: bool = False
    load_demos_start_idx: Optional[int] = None
    load_demos_end_idx: Optional[int] = None
    seed: int = 0
    debug: bool = False
    mimicgen: MimicgenConfig = field(default_factory=MimicgenConfig)


def main(cfg: Config):
    return run_mimicgen_generation(cfg)


if __name__ == "__main__":
    tyro.cli(main)
