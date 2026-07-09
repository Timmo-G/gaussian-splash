#!/usr/bin/env python3
"""Create a tiny NeRF synthetic dataset for the original 3DGS implementation."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def camera_pose(theta_degrees: float, radius: float = 4.0) -> list[list[float]]:
    theta = math.radians(theta_degrees)
    phi = math.radians(-30.0)

    translate = np.eye(4, dtype=np.float32)
    translate[2, 3] = radius

    rotate_phi = np.eye(4, dtype=np.float32)
    rotate_phi[1, 1] = math.cos(phi)
    rotate_phi[1, 2] = -math.sin(phi)
    rotate_phi[2, 1] = math.sin(phi)
    rotate_phi[2, 2] = math.cos(phi)

    rotate_theta = np.eye(4, dtype=np.float32)
    rotate_theta[0, 0] = math.cos(theta)
    rotate_theta[0, 2] = -math.sin(theta)
    rotate_theta[2, 0] = math.sin(theta)
    rotate_theta[2, 2] = math.cos(theta)

    blender_fix = np.array(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return (blender_fix @ rotate_theta @ rotate_phi @ translate).tolist()


def write_image(path: Path, frame_index: int, frame_count: int) -> None:
    image = Image.new("RGBA", (512, 512), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)

    angle = 2.0 * math.pi * frame_index / max(frame_count, 1)
    offset_x = int(35 * math.cos(angle))
    offset_y = int(25 * math.sin(angle))

    draw.ellipse((150 + offset_x, 125 + offset_y, 360 + offset_x, 335 + offset_y), fill=(230, 60, 50, 255))
    draw.rectangle((215 - offset_x, 185, 335 - offset_x, 365), fill=(40, 120, 230, 255))
    draw.polygon([(256, 80), (420, 420), (92, 420)], outline=(20, 20, 20, 255), width=8)
    image.save(path)


def write_split(root: Path, split_name: str, frame_count: int) -> None:
    frames = []
    for index in range(frame_count):
        stem = f"{split_name}_{index:03d}"
        write_image(root / f"{stem}.png", index, frame_count)
        frames.append(
            {
                "file_path": stem,
                "transform_matrix": camera_pose(360.0 * index / frame_count),
            }
        )

    metadata = {"camera_angle_x": 0.6911112070083618, "frames": frames}
    (root / f"transforms_{split_name}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/official_gs/ns_smoke"))
    parser.add_argument("--train-frames", type=int, default=16)
    parser.add_argument("--test-frames", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and args.overwrite:
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    write_split(args.output, "train", args.train_frames)
    write_split(args.output, "test", args.test_frames)
    print(f"created {args.output}")


if __name__ == "__main__":
    main()
