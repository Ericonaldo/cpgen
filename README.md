# Constraint-Preserving Data Generation for Visuomotor Policy Generalization

This repo contains the official data generation code for the paper *Constraint-Preserving Data Generation for Visuomotor Policy Generalization*.
See [adaflow-cpgen](https://github.com/kevin-thankyou-lin/adaflow-cpgen) for policy training and evaluation code.

## Installation

The supported generation stack uses Python 3.8.20, robosuite 1.4.1, MuJoCo
2.3.2, robomimic 0.3.1, Mink 0.0.7, and cuRobo 0.7.8. It requires Conda, an
NVIDIA CUDA installation with `nvcc`, and GCC/G++ 11. The checked-in
environment builds cuRobo for CUDA compute capability 8.6 (for example, RTX
30-series GPUs).

```bash
git clone https://github.com/xshenhan/cpgen.git
cd cpgen
python scripts/setup_rs14_env.py
conda activate cpgen-rs14-unified
```

The setup command initializes the locked submodules, applies the repository
patches in a cache outside the checkout, creates or updates the Conda
environment, installs the project, and verifies versions and environment
registrations. It is safe to run again. To validate an existing installation
without changing it:

```bash
python scripts/setup_rs14_env.py --verify-only
```

## Source datasets

The examples below expect three different datasets. They are not
interchangeable:

| Path | Contents | Used by |
| --- | --- | --- |
| `datasets/source/square.hdf5` plus `datasets/source/square/constraints.json` | One `NutAssemblySquare` demonstration and its CPGen constraints | CPGen cross-view |
| `datasets/mimicgen_source/square.hdf5` | `Square_D0` Panda demonstrations with `datagen_info` | Ordinary MimicGen |
| A successful ordinary MimicGen `demo.hdf5` | Panda/PandaGripper demonstrations with `datagen_info` | Panda-to-Sawyer cross-embodiment |

Download the public CPGen source data with Git LFS:

```bash
git lfs install
git clone https://huggingface.co/datasets/cpgen/datasets-src /tmp/cpgen-datasets-src
mkdir -p datasets
cp -a /tmp/cpgen-datasets-src/datasets/. datasets/
```

Download and prepare the public MimicGen Square source:

```bash
python -m mimicgen.scripts.download_datasets \
  --download_dir datasets/mimicgen_download \
  --dataset_type source \
  --tasks square
mkdir -p datasets/mimicgen_source
python -m mimicgen.scripts.prepare_src_dataset \
  --dataset datasets/mimicgen_download/source/square.hdf5 \
  --env_interface MG_Square \
  --env_interface_type robosuite \
  --output datasets/mimicgen_source/square.hdf5
```

A CPGen source trajectory does not contain MimicGen's per-step
`datagen_info`, so `datasets/source/square.hdf5` cannot be used directly for
either MimicGen command.

## Generate data

Run every command from the repository root after activating
`cpgen-rs14-unified`. Output paths below are repository-relative, avoiding
permission errors from accidental paths such as `/cpgen-output`.

### CPGen cross-view

This transfers the Panda `NutAssemblySquare` demonstration to `Square_D1`:

```bash
python demo_aug/generate.py \
  --cfg.demo-path datasets/source/square.hdf5 \
  --cfg.env-name Square_D1 \
  --cfg.seed 20260715 \
  --cfg.n-demos 1 \
  --cfg.no-require-n-demos \
  --cfg.save-dir outputs/cross-view-square-d1 \
  --cfg.merge-demo-save-path outputs/cross-view-square-d1/demo.hdf5
```

A successful trial writes `outputs/cross-view-square-d1/demo.hdf5`, an
observation dataset, and videos. A failed trial writes `demo_failures.hdf5` and
`failures/*.mp4` instead. A fixed-seed smoke result is not evidence of a
success rate.

### Ordinary MimicGen

This keeps both source and target on Panda/PandaGripper:

```bash
python scripts/generate_mimicgen.py \
  --cfg.demo-path datasets/mimicgen_source/square.hdf5 \
  --cfg.save-dir outputs/mimicgen-panda \
  --cfg.env-name Square_D0 \
  --cfg.n-demos 1 \
  --cfg.require-n-demos \
  --cfg.seed 20260715
```

The generated dataset and videos are written below
`outputs/mimicgen-panda/Square_D0_mimicgen/`.

### Cross-embodiment MimicGen

This uses the successful output of the ordinary Panda run as its legal
MimicGen source and generates a Sawyer/RethinkGripper trajectory:

```bash
python scripts/generate_mimicgen.py \
  --cfg.demo-path outputs/mimicgen-panda/Square_D0_mimicgen/demo.hdf5 \
  --cfg.save-dir outputs/mimicgen-sawyer \
  --cfg.env-name Square_D0 \
  --cfg.mimicgen.robot Sawyer \
  --cfg.mimicgen.gripper RethinkGripper \
  --cfg.n-demos 1 \
  --cfg.no-require-n-demos \
  --cfg.seed 20260815
```

The generated dataset and videos are written below
`outputs/mimicgen-sawyer/Square_D0_Sawyer_RethinkGripper_mimicgen/`.
Because this command makes one fixed-seed attempt, the directory can contain
only `demo_failed.hdf5` and a failure video. A nonempty `demo.hdf5` indicates a
successful smoke trajectory; one result does not establish a success rate.

Robosuite may print warnings about optional private macros or
`robosuite_task_zoo`. These do not affect the Square environments verified by
the installer. EGL shutdown can also print destructor warnings after output has
been flushed; validate the HDF5 and video files rather than treating that
shutdown-only warning as a failed generation.

## Playback

```bash
python scripts/playback_dataset.py \
  --dataset <path/to/hdf5> \
  --use-actions \
  --video_path playback_dataset.mp4 \
  --n 1
```

### Data-gen behavior fixing

1. Robot opens gripper after grasping object? Likely because `obj_to_parent_attachment_frame` is not specified.
