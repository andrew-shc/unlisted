#!/usr/bin/env python3
"""Average per-image envmap EXRs into a single world-space envmap.

Each per-image envmap is in camera-local space. This script rotates each one
into world space using its camera-to-world pose before averaging.

Usage:
    python average_envmaps.py <env_map_dir> <transforms_test.json> [--output <out.exr>]

Example:
    python average_envmaps.py \\
        Stanford-ORB/ground_truth/teapot_scene001/env_map \\
        Stanford-ORB/blender_HDR/teapot_scene001/transforms_test.json
"""

import argparse
import json
import os
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np


def align_to_world(envmap, c2w):
    """Rotate a camera-local equirectangular envmap to world space (ORB convention)."""
    R = np.array(c2w)[:3, :3]
    H, W = envmap.shape[:2]
    theta, phi = np.meshgrid(
        np.linspace(-0.5 * np.pi, 1.5 * np.pi, W),
        np.linspace(0.0, np.pi, H),
    )
    viewdirs = np.stack(
        [-np.cos(theta) * np.sin(phi), np.cos(phi), -np.sin(theta) * np.sin(phi)],
        axis=-1,
    ).reshape(H * W, 3)
    viewdirs = (R.T @ viewdirs.T).T.reshape(H, W, 3)
    coord_y = ((np.arccos(viewdirs[..., 1]) / np.pi * (H - 1) + H) % H).astype(np.float32)
    coord_x = (
        ((np.arctan2(viewdirs[..., 0], -viewdirs[..., 2]) + np.pi) / (2 * np.pi) * (W - 1) + W) % W
    ).astype(np.float32)
    return cv2.remap(envmap, coord_x, coord_y, cv2.INTER_LINEAR)


def main():
    parser = argparse.ArgumentParser(description="Average pose-aligned per-image envmap EXRs")
    parser.add_argument("env_map_dir", type=Path, help="directory containing per-image envmap .exr files")
    parser.add_argument("transforms_json", type=Path, help="transforms_test.json with per-frame c2w poses")
    parser.add_argument("--output", type=Path, default=None, help="output path (default: <env_map_dir>/envmap_avg.exr)")
    args = parser.parse_args()

    with open(args.transforms_json) as f:
        frames = json.load(f)["frames"]

    # build stem -> c2w mapping
    pose_by_stem = {Path(fr["file_path"]).stem: fr["transform_matrix"] for fr in frames}

    paths = sorted(args.env_map_dir.glob("*.exr"))
    if not paths:
        raise FileNotFoundError(f"No .exr files found in {args.env_map_dir}")

    acc = None
    count = 0
    for p in paths:
        stem = p.stem
        if stem not in pose_by_stem:
            print(f"  WARNING: no pose for {p.name}, skipping")
            continue

        env = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)  # (H, W, 3) BGR float32
        if env is None:
            raise IOError(f"Failed to read {p}")

        aligned = align_to_world(env, pose_by_stem[stem])
        aligned = np.roll(aligned, aligned.shape[1] // 2, axis=1)  # world → NPBIR convention

        if acc is None:
            acc = aligned.astype(np.float64)
        else:
            if aligned.shape != acc.shape:
                raise ValueError(f"{p.name} shape {aligned.shape} != expected {acc.shape}")
            acc += aligned

        count += 1
        print(f"  aligned {p.name}")

    if count == 0:
        raise RuntimeError("No envmaps were processed")

    avg = (acc / count).astype(np.float32)
    out_path = args.output or (args.env_map_dir / "envmap_avg.exr")
    cv2.imwrite(str(out_path), avg)
    print(f"Averaged {count} envmap(s) ({avg.shape[0]}x{avg.shape[1]}) -> {out_path}")


if __name__ == "__main__":
    main()
