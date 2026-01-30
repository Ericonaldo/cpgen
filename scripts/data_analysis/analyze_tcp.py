"""Analyze TCP (Tool Center Point) data from HDF5 datasets.

Generates 4 plots:
1. TCP pixel coords (u, v, depth) over time for all cameras
2. Wrist camera TCP stability (should be nearly constant)
3. Third-person camera TCP 2D tracks
4. EEF 3D trajectory

Usage:
    python scripts/data_analysis/analyze_tcp.py \
        --dataset <path/to/dataset.hdf5> \
        --demo_idx 0 \
        --output_dir tcp_analysis
"""

import argparse
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def analyze_tcp(dataset_path, demo_idx=0, output_dir="tcp_analysis"):
    os.makedirs(output_dir, exist_ok=True)

    f = h5py.File(dataset_path, "r")
    demo_key = f"data/demo_{demo_idx}"
    if demo_key not in f:
        available = [k for k in f["data"].keys()]
        raise ValueError(f"Demo {demo_idx} not found. Available: {available}")

    obs = f[demo_key]["obs"]
    traj_len = next(iter(obs.values())).shape[0]

    # Detect cameras
    cam_names = sorted(
        set(k.rsplit("_image", 1)[0] for k in obs.keys() if k.endswith("_image"))
    )
    tcp_cams = [c for c in cam_names if f"{c}_tcp_pixel_coords" in obs]

    print(f"Dataset: {dataset_path}")
    print(f"Demo {demo_idx}: {traj_len} timesteps")
    print(f"Cameras with TCP: {len(tcp_cams)}/{len(cam_names)}")

    # ---- Plot 1: TCP pixel coords over time ----
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for cam in tcp_cams:
        coords = obs[f"{cam}_tcp_pixel_coords"][:]
        is_wrist = "eye_in_hand" in cam
        ls = "--" if is_wrist else "-"
        alpha = 0.7 if is_wrist else 1.0
        axes[0].plot(coords[:, 0], linestyle=ls, alpha=alpha, label=cam)
        axes[1].plot(coords[:, 1], linestyle=ls, alpha=alpha, label=cam)
        axes[2].plot(coords[:, 2], linestyle=ls, alpha=alpha, label=cam)

    axes[0].set_ylabel("u (pixel)")
    axes[0].set_title("TCP Pixel Coordinate: u")
    axes[1].set_ylabel("v (pixel)")
    axes[1].set_title("TCP Pixel Coordinate: v")
    axes[2].set_ylabel("depth (m)")
    axes[2].set_title("TCP Depth")
    axes[2].set_xlabel("Timestep")

    for ax in axes:
        ax.legend(fontsize=5, ncol=3, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "tcp_coords_over_time.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")

    # ---- Plot 2: Wrist camera TCP stability ----
    wrist_cams = [c for c in tcp_cams if "eye_in_hand" in c]
    if wrist_cams:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        print()
        print("=== Wrist Camera TCP Stability (should be nearly constant) ===")
        for cam in wrist_cams:
            coords = obs[f"{cam}_tcp_pixel_coords"][:]
            u_std, v_std, d_std = np.std(coords, axis=0)
            u_mean, v_mean, d_mean = np.mean(coords, axis=0)
            print(
                f"  {cam}: u={u_mean:.1f}+-{u_std:.2f}, "
                f"v={v_mean:.1f}+-{v_std:.2f}, "
                f"d={d_mean:.4f}+-{d_std:.4f}"
            )
            ax.plot(coords[:, 0], coords[:, 1], "o-", markersize=1, alpha=0.5, label=cam)

        ax.set_xlabel("u (pixel)")
        ax.set_ylabel("v (pixel)")
        ax.set_title("Wrist Camera TCP (u,v) Over Time (should cluster tightly)")
        ax.legend(fontsize=6)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, "wrist_tcp_stability.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved {path}")

    # ---- Plot 3: Third-person camera TCP tracks ----
    third_cams = [c for c in tcp_cams if "third_view" in c or c == "agentview"]
    if third_cams:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        print()
        print("=== Third-Person Camera TCP Range ===")
        for cam in third_cams:
            coords = obs[f"{cam}_tcp_pixel_coords"][:]
            u_range = coords[:, 0].max() - coords[:, 0].min()
            v_range = coords[:, 1].max() - coords[:, 1].min()
            print(f"  {cam}: u_range={u_range:.1f}px, v_range={v_range:.1f}px")
            ax.plot(coords[:, 0], coords[:, 1], "o-", markersize=1, alpha=0.7, label=cam)

        ax.set_xlabel("u (pixel)")
        ax.set_ylabel("v (pixel)")
        ax.set_title("Third-Person Camera TCP (u,v) Tracks")
        ax.legend(fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, 256)
        ax.set_ylim(256, 0)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, "third_person_tcp_tracks.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved {path}")

    # ---- Plot 4: EEF 3D trajectory ----
    if "robot0_eef_pos" in obs:
        eef_pos = obs["robot0_eef_pos"][:]
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        cmap = plt.get_cmap("viridis")
        for i in range(len(eef_pos) - 1):
            ax.plot(
                [eef_pos[i, 0], eef_pos[i + 1, 0]],
                [eef_pos[i, 1], eef_pos[i + 1, 1]],
                [eef_pos[i, 2], eef_pos[i + 1, 2]],
                color=cmap(i / len(eef_pos)),
            )
        ax.scatter(*eef_pos[0], color="green", s=100, label="Start", zorder=5)
        ax.scatter(*eef_pos[-1], color="red", s=100, label="End", zorder=5)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("EEF 3D Trajectory")
        ax.legend()
        plt.tight_layout()
        path = os.path.join(output_dir, "eef_3d_trajectory.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved {path}")

    f.close()
    print(f"\nDone! All plots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Analyze TCP data from HDF5 datasets")
    parser.add_argument("--dataset", required=True, help="Path to HDF5 dataset")
    parser.add_argument("--demo_idx", type=int, default=0, help="Demo index")
    parser.add_argument("--output_dir", default="tcp_analysis", help="Output directory for plots")
    args = parser.parse_args()

    analyze_tcp(args.dataset, args.demo_idx, args.output_dir)


if __name__ == "__main__":
    main()
