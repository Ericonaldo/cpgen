"""Visualize TCP (Tool Center Point) overlaid on camera images from HDF5 datasets.

Draws TCP pixel location (green dot) and 3-axis orientation arrows on each camera
view that has TCP data. Outputs a grid image or video.

Usage:
    python scripts/visualize_tcp.py \
        --dataset <path/to/dataset.hdf5> \
        --demo_idx 0 \
        --frame_idx 0 \
        --output_path tcp_vis.png

    # Visualize multiple frames as video
    python scripts/visualize_tcp.py \
        --dataset <path/to/dataset.hdf5> \
        --demo_idx 0 \
        --output_path tcp_vis.mp4 \
        --num_frames 50
"""

import argparse
import cv2
import h5py
import numpy as np
from pathlib import Path


def draw_tcp_on_image(image, tcp_pixel_coords, tcp_dir_x=None, tcp_dir_y=None, tcp_dir_z=None):
    """Draw TCP marker and orientation axes on an image.

    Args:
        image: HxWx3 uint8 BGR image
        tcp_pixel_coords: [u, v, depth] — u,v in pixel coords or normalized [0,1]
        tcp_dir_x/y/z: [dx, dy] 2D direction vectors for each axis
    """
    result = image.copy()
    h, w = result.shape[:2]

    u, v = float(tcp_pixel_coords[0]), float(tcp_pixel_coords[1])

    # Detect if coords are normalized (0-1) or pixel values
    if u <= 1.0 and v <= 1.0 and u >= 0.0 and v >= 0.0:
        u = int(u * (w - 1))
        v = int(v * (h - 1))
    else:
        u, v = int(u), int(v)

    depth = float(tcp_pixel_coords[2])

    # Draw TCP center
    if 0 <= u < w and 0 <= v < h:
        cv2.circle(result, (u, v), 6, (0, 255, 0), -1)   # Green filled
        cv2.circle(result, (u, v), 7, (0, 0, 0), 2)       # Black border

        # Draw depth text
        cv2.putText(result, f"d={depth:.2f}", (u + 10, v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw orientation axes
        axis_length = 30
        axes = [
            (tcp_dir_x, (0, 0, 255), "X"),    # Red
            (tcp_dir_y, (0, 255, 0), "Y"),     # Green
            (tcp_dir_z, (255, 0, 0), "Z"),     # Blue
        ]
        for dir_vec, color, label in axes:
            if dir_vec is None:
                continue
            dx, dy = float(dir_vec[0]), float(dir_vec[1])
            norm = np.sqrt(dx**2 + dy**2)
            if norm > 1e-6:
                u_end = int(u + (dx / norm) * axis_length)
                v_end = int(v + (dy / norm) * axis_length)
                if 0 <= u_end < w and 0 <= v_end < h:
                    cv2.arrowedLine(result, (u, v), (u_end, v_end), color, 2, tipLength=0.3)
    else:
        # TCP out of view — draw label in corner
        cv2.putText(result, f"TCP out of view ({u},{v})", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return result


def get_camera_tcp_data(obs_group, camera_name, frame_idx):
    """Extract TCP data for a camera at a given frame index.

    Returns dict with keys: pixel_coords, dir_x, dir_y, dir_z (or None if missing).
    """
    prefix = f"{camera_name}_tcp"
    coords_key = f"{prefix}_pixel_coords"
    if coords_key not in obs_group:
        return None

    data = {
        "pixel_coords": obs_group[coords_key][frame_idx],
    }
    for suffix in ["dir_x", "dir_y", "dir_z"]:
        key = f"{prefix}_{suffix}"
        if key in obs_group:
            data[suffix] = obs_group[key][frame_idx]
        else:
            data[suffix] = None
    return data


def visualize_frame(obs_group, frame_idx, camera_names=None):
    """Create a grid visualization of TCP on all cameras for one frame.

    Args:
        obs_group: HDF5 obs group for a demo
        frame_idx: which timestep to visualize
        camera_names: list of camera names, or None to auto-detect

    Returns:
        Grid image (numpy array, BGR)
    """
    if camera_names is None:
        # Auto-detect cameras from image keys
        camera_names = []
        for key in sorted(obs_group.keys()):
            if key.endswith("_image"):
                cam = key[: -len("_image")]
                camera_names.append(cam)

    panels = []
    for cam in camera_names:
        img_key = f"{cam}_image"
        if img_key not in obs_group:
            continue

        img = obs_group[img_key][frame_idx]  # HxWx3, RGB
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        tcp_data = get_camera_tcp_data(obs_group, cam, frame_idx)

        if tcp_data is not None:
            img_bgr = draw_tcp_on_image(
                img_bgr,
                tcp_data["pixel_coords"],
                tcp_data.get("dir_x"),
                tcp_data.get("dir_y"),
                tcp_data.get("dir_z"),
            )
            label = cam
        else:
            label = f"{cam} (no TCP)"

        # Add camera name label
        cv2.putText(img_bgr, label, (5, img_bgr.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        panels.append(img_bgr)

    if not panels:
        raise ValueError("No camera images found in obs group")

    # Arrange in grid: up to 4 per row
    cols = min(len(panels), 4)
    rows = (len(panels) + cols - 1) // cols

    h, w = panels[0].shape[:2]
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

    for i, panel in enumerate(panels):
        r, c = divmod(i, cols)
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = panel

    return grid


def main():
    parser = argparse.ArgumentParser(description="Visualize TCP on camera images from HDF5")
    parser.add_argument("--dataset", required=True, help="Path to HDF5 dataset")
    parser.add_argument("--demo_idx", type=int, default=0, help="Demo index")
    parser.add_argument("--frame_idx", type=int, default=0, help="Frame index for single image output")
    parser.add_argument("--output_path", required=True, help="Output path (.png for image, .mp4 for video)")
    parser.add_argument("--num_frames", type=int, default=0,
                        help="Number of frames for video (0 = single frame image)")
    parser.add_argument("--camera_names", nargs="+", default=None,
                        help="Camera names to visualize (default: all)")
    parser.add_argument("--fps", type=int, default=20, help="FPS for video output")
    args = parser.parse_args()

    f = h5py.File(args.dataset, "r")
    demo_key = f"data/demo_{args.demo_idx}"
    if demo_key not in f:
        available = [k for k in f["data"].keys()]
        raise ValueError(f"Demo {args.demo_idx} not found. Available: {available}")

    obs = f[demo_key]["obs"]
    traj_len = next(iter(obs.values())).shape[0]
    print(f"Demo {args.demo_idx}: {traj_len} timesteps")

    # List cameras and TCP status
    cam_names = args.camera_names
    if cam_names is None:
        cam_names = [k[:-len("_image")] for k in sorted(obs.keys()) if k.endswith("_image")]
    print(f"Cameras: {cam_names}")
    for cam in cam_names:
        has_tcp = f"{cam}_tcp_pixel_coords" in obs
        print(f"  {cam}: TCP={'YES' if has_tcp else 'NO'}")

    output_path = Path(args.output_path)

    if args.num_frames > 0 and output_path.suffix == ".mp4":
        # Video mode
        num_frames = min(args.num_frames, traj_len)
        first_frame = visualize_frame(obs, 0, cam_names)
        h, w = first_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (w, h))
        for t in range(num_frames):
            frame = visualize_frame(obs, t, cam_names)
            writer.write(frame)
            if (t + 1) % 50 == 0:
                print(f"  Frame {t + 1}/{num_frames}")
        writer.release()
        print(f"Video saved to {output_path} ({num_frames} frames)")
    else:
        # Single frame mode
        frame_idx = min(args.frame_idx, traj_len - 1)
        grid = visualize_frame(obs, frame_idx, cam_names)
        cv2.imwrite(str(output_path), grid)
        print(f"Image saved to {output_path} (frame {frame_idx})")

    f.close()


if __name__ == "__main__":
    main()
