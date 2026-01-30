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

## Multi-View Random Camera Generation

CPGen supports generating demonstrations with diverse random camera viewpoints for training view-invariant policies.

### Presets

Four sampling presets control camera view diversity:

| Preset | Azimuth Range | Elevation Range | Use Case |
|--------|---------------|-----------------|----------|
| `conservative` | -45° to 45° | 25° to 55° | Frontal views only (safest, like agentview) |
| `wide` | -90° to 90° | 15° to 70° | Full front hemisphere, minimal occlusion risk |
| `hemisphere` | -135° to 135° | 10° to 80° | 270° coverage, some side/back views, moderate occlusion |
| `full_sphere` | -180° to 180° | 0° to 85° | Full 360° coverage, expect back-view occlusions |

**Recommendation**: Start with `wide` for good diversity without occlusions.

### Workflow (Post-Processing)

**Step 1: Generate demonstrations with states (no rendering)**

```bash
MUJOCO_GL=egl python demo_aug/generate.py \
    --cfg.demo-path datasets/source/coffee.hdf5 \
    --cfg.env-name Coffee
# Output: datasets/generated/coffee_TIMESTAMP/coffee.hdf5 (states only)
```

**Step 2: Render observations with random camera views**

```bash
# Example 1: Wide frontal views (recommended for training)
python scripts/dataset_states_to_obs.py \
    --dataset datasets/generated/coffee_TIMESTAMP/coffee.hdf5 \
    --enable_multi_view \
    --multi_view_preset wide \
    --num_third_views 6 \
    --multi_view_seed 42 \
    --camera_names agentview robot0_eye_in_hand

# Example 2: Hemisphere views (test generalization)
python scripts/dataset_states_to_obs.py \
    --dataset datasets/generated/coffee_TIMESTAMP/coffee.hdf5 \
    --enable_multi_view \
    --multi_view_preset hemisphere \
    --num_third_views 8 \
    --multi_view_seed 123

# Example 3: Manual range override
python scripts/dataset_states_to_obs.py \
    --dataset datasets/generated/coffee_TIMESTAMP/coffee.hdf5 \
    --enable_multi_view \
    --multi_view_preset conservative \
    --multi_view_azimuth_range -60 60 \
    --num_third_views 4
```

**Step 3 (Optional): Generate preview for validation**

```bash
python scripts/playback_dataset.py \
    --dataset datasets/generated/coffee_TIMESTAMP/coffee.hdf5 \
    --enable_multi_view \
    --multi_view_preset wide \
    --num_third_views 6 \
    --multi_view_seed 42 \
    --multi_view_preview_path preview_cameras.png \
    --video_path preview.mp4 \
    --n 1
# Check preview_cameras.png to validate camera quality before full rendering
```

### HDF5 Output Structure

One HDF5 file contains ALL camera views:

```
dataset.hdf5
├── demo_0/
│   ├── obs/
│   │   ├── agentview_image              # Original camera
│   │   ├── agentview_depth
│   │   ├── agentview_intrinsics
│   │   ├── agentview_extrinsics
│   │   ├── third_view_0_image           # Random view 1
│   │   ├── third_view_0_depth
│   │   ├── third_view_0_intrinsics
│   │   ├── third_view_0_extrinsics
│   │   ├── third_view_1_image           # Random view 2
│   │   ├── third_view_1_depth
│   │   ├── third_view_1_intrinsics
│   │   ├── third_view_1_extrinsics
│   │   └── ...
```

### SpatialAlignVLA Integration

Load different camera views from the same HDF5 file:

```python
from savla.dataset.robomimic_hdf5 import RobomimicHDF5Dataset
from torch.utils.data import ConcatDataset

# Load multiple camera views
train_datasets = [
    RobomimicHDF5Dataset(
        hdf5_path="cpgen_dataset.hdf5",
        camera_name='agentview',
        use_depth=True,
    ),
    RobomimicHDF5Dataset(
        hdf5_path="cpgen_dataset.hdf5",
        camera_name='third_view_0',
        use_depth=True,
    ),
    RobomimicHDF5Dataset(
        hdf5_path="cpgen_dataset.hdf5",
        camera_name='third_view_1',
        use_depth=True,
    ),
]
combined = ConcatDataset(train_datasets)
```

### Troubleshooting

**Issue: Severe occlusions in generated views**
- Use `--multi_view_preview_path` to check camera quality first
- Try a more conservative preset (e.g., switch from `hemisphere` to `wide`)
- Increase `--multi_view_min_separation` for more diverse angles
- Use different random seeds to resample bad configurations

**Issue: All views look similar**
- Increase `--num_third_views` for more cameras
- Use a more diverse preset (e.g., `hemisphere` instead of `conservative`)
- Check that `--multi_view_seed` is not reused across different datasets

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

## Robosuite Version Compatibility

CPGen supports both old datasets (robosuite 1.4) and new cpgen-envs (robosuite 1.5+). The controller API changed between versions:

- **Robosuite 1.4 and earlier**: Single `robot.controller` with `goal_pos`, `goal_ori`, `eef_name` attributes
- **Robosuite 1.5+**: Composite controllers with `robot.composite_controller.part_controllers["right"]`

### Controller Config Structure

**Robosuite 1.4 format** (old):
```python
{
    "type": "OSC_POSE",
    "control_delta": True,
    "kp": 150,
    ...
}
```

**Robosuite 1.5+ format** (new) - IMPORTANT: Flat structure:
```python
{
    "type": "BASIC",           # Composite controller type
    "body_parts": {
        "right": {             # ⚠️ Direct under body_parts (NOT under "arms")
            "type": "OSC_POSE",
            "input_type": "absolute",
            "gripper": {"type": "GRIP"},
            ...
        }
    }
}
```

**Common mistake**: Adding an extra `"arms"` nesting level:
```python
# ❌ WRONG - Don't do this:
{
    "body_parts": {
        "arms": {              # Extra nesting - robosuite 1.5 will fail
            "right": {...}
        }
    }
}
```

### Automatic Compatibility Handling

The codebase automatically handles version differences via `demo_aug/utils/robosuite_utils.py`:

- **`refactor_composite_controller_config()`** - Converts old (1.4) controller configs to new (1.5+) format
- **`get_robot_controller()`** - Safely accesses robot controller across versions
- **`get_robot_eef_site_name()`** - Gets end-effector site name across versions
- **`set_controller_config_absolute()`** - Sets absolute action mode for both formats

### Files with Version-Aware Code

- `demo_aug/utils/robomimic_utils.py` - Action conversion, camera info (lines 45, 235)
- `demo_aug/generate.py` - Controller config loading with auto-conversion (line 4423)
- `scripts/segment/demo_seg.py` - Controller config refactoring with fallback (line 267)

### Working with Old Datasets

When loading datasets recorded with robosuite 1.4:
1. Controller configs are automatically detected and converted to 1.5+ format
2. If auto-conversion fails, the code falls back to original configs (may cause issues)
3. Version metadata (`env_version`) is saved in newly generated datasets

## Troubleshooting

**Robot opens gripper after grasping**: `obj_to_parent_attachment_frame` not specified in constraint config.

**Controller attribute errors (e.g., `AttributeError: 'CompositeController' has no attribute 'goal_pos'`)**: Make sure you're using the compatibility functions from `robosuite_utils.py` instead of directly accessing `robot.controller`.

## Project Goal

Build flexible environment wrappers to support data collection and testing with multiple camera views.

**Challenge**: Original source demonstrations are recorded from fixed camera views (third-person view + wrist view). We need to generate demonstrations from more diverse viewpoints.

**Purpose**:
- Collect more diverse demonstrations simultaneously from different camera angles
- Enable flexible view selection at test time
- Support multi-view visuomotor policy generalization

## Code Style Guidelines

**Conciseness and Clarity**:
- Always write concise and clear code
- Prefer simple, direct solutions over complex abstractions
- Keep functions focused and single-purpose

**Minimal Changes**:
- Do NOT change original code unless absolutely necessary
- Do NOT change the project structure unless absolutely necessary
- If changes to existing code or structure are needed, **ask for permission first**

**When Making Changes**:
- Explain why the change is necessary
- Show what will be modified before making changes
- Prefer extending functionality over modifying existing code
- Add new files/modules rather than restructuring existing ones when possible