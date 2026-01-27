# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CPGen (Constraint-Preserving Data Generation) is a research codebase for generating augmented robot demonstration data for visuomotor policy generalization. It takes source demonstrations and generates constraint-preserving augmented trajectories using motion planning and geometric transformations.

## Build and Development Commands

```bash
# Install (after cloning mimicgen and cpgen-envs dependencies)
pip install -e .

# Linting and formatting
make check          # Run ruff and black without changes
make autoformat     # Auto-fix with ruff and black

# Pre-commit runs ruff (import sorting + formatting) on staged files
```

## Core Commands

```bash
# Generate augmented data
MUJOCO_GL=egl python demo_aug/generate.py --cfg.demo-path <path/to/hdf5> --cfg.env-name <EnvName>

# Playback dataset with actions
python scripts/playback_dataset.py --dataset <path/to/hdf5> --use-actions --video_path playback_dataset.mp4 --n 1
```

## Architecture

### Main Entry Point
- `demo_aug/generate.py` - Main data generation script. Loads demos, segments into constraints, applies augmentations, runs motion planning, and saves augmented trajectories.

### Core Components

**Configs** (`demo_aug/configs/`):
- `base_config.py` - Central config definitions: `DemoAugConfig`, `ConstraintInfo`, `AugmentationConfig` (SE3, scale, shear, warp, EE noise augmentations)
- `env_configs.py` - Environment configs (sim/real robot setups)
- `robot_configs.py` - Robot-specific configurations

**Augmentor** (`demo_aug/augmentor/augmentor.py`):
- `Augmentor.generate_augmented_demos()` - Main augmentation pipeline
- `Augmentor.apply_augs()` - Applies geometric transformations (SE3, scale, shear) to trajectories and objects

**Environments** (`demo_aug/envs/`):
- `nerf_robomimic_env.py` - Primary environment combining NeRF object rendering with MuJoCo robot simulation
- `base_env.py` - Base environment class with motion planner type definitions
- `motion_planners/` - Motion planning implementations (CuRoBO, Drake trajopt, linear interpolation, sampling-based)

**Demo Representation** (`demo_aug/demo.py`):
- `Demo` dataclass - Contains timestep data, constraint infos, and reconstruction manager

### Data Flow
1. Load source demo (HDF5) with states, actions, observations
2. Segment demo into constraint time ranges (can use LLM for segmentation)
3. For each constraint: sample augmentation parameters, apply geometric transforms
4. Motion plan from random start to transformed constraint pose
5. Track original trajectory with applied transforms
6. Save augmented trajectories to HDF5

### Key Dependencies
- `robomimic` - Demo format and environment wrappers
- `mimicgen` - Additional env definitions
- `cpgen-envs` - Custom task environments
- `curobo` - GPU-accelerated motion planning
- `mujoco` - Physics simulation
- `tyro` - Config management with YAML serialization

## Troubleshooting

**Robot opens gripper after grasping**: `obj_to_parent_attachment_frame` not specified in constraint config.
