#!/usr/bin/env python3
"""Create a tiny Blender-format dataset for Nerfstudio smoke tests."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def pose(theta_degrees: float, radius: float = 4.0) -> list[list[float]]:
    theta = math.radians(theta_degrees)
    camera_to_world = np.eye(4, dtype=np.float32)
    camera_to_world[0, 3] = radius * math.sin(theta)
    camera_to_world[1, 3] = 0.0
    camera_to_world[2, 3] = radius * math.cos(theta)
    return camera_to_world.tolist()


def write_image(path: Path, frame_index: int) -> None:
    image = Image.new("RGBA", (512, 512), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)

    offset = int(18 * math.sin(frame_index * 0.7))
    draw.ellipse((150 + offset, 120, 360 + offset, 330), fill=(230, 60, 50, 255))
    draw.rectangle((215, 185 + offset, 335, 365 + offset), fill=(40, 120, 230, 255))
    draw.line((90, 420, 420, 90), fill=(20, 20, 20, 255), width=8)

    image.save(path)


def write_split(root: Path, split_name: str, frame_count: int) -> None:
    split_dir = root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for index in range(frame_count):
        stem = f"{split_name}/r_{index:03d}"
        write_image(root / f"{stem}.png", index)
        frames.append(
            {
                "file_path": stem,
                "transform_matrix": pose(360.0 * index / frame_count),
            }
        )

    metadata = {
        "camera_angle_x": 0.6911112070083618,
        "frames": frames,
    }
    (root / f"transforms_{split_name}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/blender/ns_smoke"))
    parser.add_argument("--train-frames", type=int, default=16)
    parser.add_argument("--val-frames", type=int, default=4)
    parser.add_argument("--test-frames", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and args.overwrite:
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    write_split(args.output, "train", args.train_frames)
    write_split(args.output, "val", args.val_frames)
    write_split(args.output, "test", args.test_frames)

    print(f"created {args.output}")


if __name__ == "__main__":
    main()
