"""
Split a raw multi-view HDF5 into per-pair HDF5 files with savla-compatible camera names.

Each pair maps:
  - third_view_i_*  -> agentview_*
  - robot0_eye_in_hand_perturbed_i_* -> robot0_eye_in_hand_*

All non-camera obs (joint states, gripper, etc.), actions, states, rewards, dones,
and HDF5 attributes are copied to each output file.

Example usage:
    python scripts/split_view_pairs.py \
        --input datasets/raw/square_8view_raw.hdf5 \
        --output_dir datasets/square_8view/ \
        --num_pairs 4
"""

import argparse
import os

import h5py
import numpy as np


# Camera name prefixes that get renamed
THIRD_VIEW_PREFIX = "third_view"
WRIST_PERTURBED_PREFIX = "robot0_eye_in_hand_perturbed"

# Target savla-compatible names
TARGET_AGENTVIEW = "agentview"
TARGET_WRIST = "robot0_eye_in_hand"

# Original camera prefixes to skip (not copied to output)
ORIGINAL_CAMERA_PREFIXES = ("agentview_", "robot0_eye_in_hand_")


def is_original_camera_key(key: str) -> bool:
    """Check if an obs key belongs to an original (non-paired) camera."""
    # Skip original agentview and robot0_eye_in_hand keys,
    # but NOT robot0_eye_in_hand_perturbed_* keys
    if key.startswith("robot0_eye_in_hand_perturbed_"):
        return False
    if key.startswith("robot0_eye_in_hand_"):
        return True
    if key.startswith("agentview_"):
        return True
    return False


def is_paired_camera_key(key: str, pair_idx: int) -> bool:
    """Check if an obs key belongs to a specific pair index."""
    third_prefix = f"{THIRD_VIEW_PREFIX}_{pair_idx}_"
    wrist_prefix = f"{WRIST_PERTURBED_PREFIX}_{pair_idx}_"
    return key.startswith(third_prefix) or key.startswith(wrist_prefix)


def is_any_camera_key(key: str) -> bool:
    """Check if an obs key belongs to any camera (original or generated)."""
    camera_prefixes = (
        "agentview_",
        "robot0_eye_in_hand_",
        f"{THIRD_VIEW_PREFIX}_",
    )
    return any(key.startswith(p) for p in camera_prefixes)


def rename_camera_key(key: str, pair_idx: int) -> str:
    """Rename a paired camera obs key to savla-compatible name.

    third_view_{pair_idx}_{suffix} -> agentview_{suffix}
    robot0_eye_in_hand_perturbed_{pair_idx}_{suffix} -> robot0_eye_in_hand_{suffix}
    """
    third_prefix = f"{THIRD_VIEW_PREFIX}_{pair_idx}_"
    wrist_prefix = f"{WRIST_PERTURBED_PREFIX}_{pair_idx}_"

    if key.startswith(third_prefix):
        suffix = key[len(third_prefix):]
        return f"{TARGET_AGENTVIEW}_{suffix}"
    elif key.startswith(wrist_prefix):
        suffix = key[len(wrist_prefix):]
        return f"{TARGET_WRIST}_{suffix}"
    else:
        raise ValueError(f"Key '{key}' does not match pair index {pair_idx}")


def split_view_pairs(input_path: str, output_dir: str, num_pairs: int):
    """Split a raw multi-view HDF5 into per-pair files.

    Args:
        input_path: Path to raw HDF5 with all camera views
        output_dir: Directory to write pair_0.hdf5, pair_1.hdf5, ...
        num_pairs: Number of pairs to extract
    """
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(input_path, "r") as f_in:
        data_grp = f_in["data"]
        demos = sorted(
            [k for k in data_grp.keys() if k.startswith("demo")],
            key=lambda x: int(x.split("_")[-1]),
        )

        if not demos:
            print("No demos found in input file")
            return

        # Verify expected camera keys exist in first demo
        first_obs_keys = list(data_grp[demos[0]]["obs"].keys())
        for i in range(num_pairs):
            third_keys = [k for k in first_obs_keys if k.startswith(f"{THIRD_VIEW_PREFIX}_{i}_")]
            wrist_keys = [k for k in first_obs_keys if k.startswith(f"{WRIST_PERTURBED_PREFIX}_{i}_")]
            if not third_keys:
                print(f"WARNING: No third_view_{i}_* keys found in obs. Available: {first_obs_keys}")
            if not wrist_keys:
                print(f"WARNING: No robot0_eye_in_hand_perturbed_{i}_* keys found in obs. Available: {first_obs_keys}")

        # Identify non-camera obs keys (copied to all output files)
        non_camera_keys = [k for k in first_obs_keys if not is_any_camera_key(k)]
        print(f"Non-camera obs keys: {non_camera_keys}")

        for pair_idx in range(num_pairs):
            output_path = os.path.join(output_dir, f"pair_{pair_idx}.hdf5")
            print(f"\nCreating {output_path} (pair {pair_idx})...")

            # Find camera keys for this pair
            pair_camera_keys = [k for k in first_obs_keys if is_paired_camera_key(k, pair_idx)]
            print(f"  Camera keys for pair {pair_idx}: {pair_camera_keys}")

            with h5py.File(output_path, "w") as f_out:
                out_data_grp = f_out.create_group("data")

                # Copy top-level data attributes
                for attr_name, attr_val in data_grp.attrs.items():
                    out_data_grp.attrs[attr_name] = attr_val

                total_samples = 0

                for ep in demos:
                    ep_grp = data_grp[ep]
                    out_ep_grp = out_data_grp.create_group(ep)

                    # Copy non-obs datasets (actions, states, rewards, dones, etc.)
                    for ds_name in ep_grp.keys():
                        if ds_name == "obs" or ds_name == "next_obs":
                            continue
                        ep_grp.copy(ds_name, out_ep_grp)

                    # Copy episode attributes
                    for attr_name, attr_val in ep_grp.attrs.items():
                        out_ep_grp.attrs[attr_name] = attr_val

                    # Create obs group
                    out_obs_grp = out_ep_grp.create_group("obs")

                    # Copy non-camera obs keys
                    for key in non_camera_keys:
                        if key in ep_grp["obs"]:
                            ep_grp["obs"].copy(key, out_obs_grp)

                    # Copy and rename paired camera keys
                    for key in pair_camera_keys:
                        if key in ep_grp["obs"]:
                            new_key = rename_camera_key(key, pair_idx)
                            data = ep_grp["obs"][key][()]
                            out_obs_grp.create_dataset(
                                new_key, data=data, compression="gzip"
                            )

                    # Handle next_obs if present
                    if "next_obs" in ep_grp:
                        out_next_obs_grp = out_ep_grp.create_group("next_obs")
                        next_obs_keys = list(ep_grp["next_obs"].keys())

                        for key in next_obs_keys:
                            if not is_any_camera_key(key):
                                # Non-camera key, copy as-is
                                ep_grp["next_obs"].copy(key, out_next_obs_grp)
                            elif is_paired_camera_key(key, pair_idx):
                                # Paired camera key, rename
                                new_key = rename_camera_key(key, pair_idx)
                                data = ep_grp["next_obs"][key][()]
                                out_next_obs_grp.create_dataset(
                                    new_key, data=data, compression="gzip"
                                )

                    total_samples += ep_grp.attrs.get("num_samples", 0)

                out_data_grp.attrs["total"] = total_samples

            print(f"  Wrote {len(demos)} demos, {total_samples} samples to {output_path}")

    print(f"\nDone. Created {num_pairs} pair files in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Split raw multi-view HDF5 into per-pair files with savla-compatible camera names"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to raw HDF5 with all camera views",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write pair_0.hdf5, pair_1.hdf5, ...",
    )
    parser.add_argument(
        "--num_pairs",
        type=int,
        required=True,
        help="Number of (third_view, wrist_perturbed) pairs to extract",
    )
    args = parser.parse_args()

    split_view_pairs(args.input, args.output_dir, args.num_pairs)


if __name__ == "__main__":
    main()
