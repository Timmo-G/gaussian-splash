#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from nerfstudio.data.utils.colmap_parsing_utils import Camera, read_cameras_binary, write_cameras_binary


def scaled_camera(camera: Camera, width: int, height: int) -> Camera:
    sx = width / camera.width
    sy = height / camera.height
    params = np.array(camera.params, dtype=np.float64).copy()

    if camera.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}:
        params[0] *= (sx + sy) * 0.5
        params[1] *= sx
        params[2] *= sy
    elif camera.model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
        params[0] *= sx
        params[1] *= sy
        params[2] *= sx
        params[3] *= sy
    elif camera.model == "FOV":
        params[0] *= sx
        params[1] *= sy
        params[2] *= sx
        params[3] *= sy
    else:
        raise ValueError(f"Unsupported camera model: {camera.model}")

    return Camera(id=camera.id, model=camera.model, width=width, height=height, params=params)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Nerfstudio-compatible COLMAP dataset copy.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    images_src = source / "images"
    sparse_src = source / "sparse" / "0"
    sparse_out = output / "sparse" / "0"

    if not images_src.is_dir():
        raise FileNotFoundError(images_src)
    if not sparse_src.is_dir():
        raise FileNotFoundError(sparse_src)

    first_image = sorted(images_src.glob("*"))[0]
    width, height = Image.open(first_image).size

    output.mkdir(parents=True, exist_ok=True)
    sparse_out.mkdir(parents=True, exist_ok=True)

    images_out = output / "images"
    if images_out.exists() or images_out.is_symlink():
        if images_out.resolve() != images_src:
            raise FileExistsError(f"{images_out} already exists and does not point at {images_src}")
    else:
        images_out.symlink_to(images_src, target_is_directory=True)

    cameras = read_cameras_binary(sparse_src / "cameras.bin")
    cameras = {camera_id: scaled_camera(camera, width, height) for camera_id, camera in cameras.items()}
    write_cameras_binary(cameras, sparse_out / "cameras.bin")

    for name in ("images.bin", "points3D.bin", "points3D.ply", "project.ini"):
        src = sparse_src / name
        dst = sparse_out / name
        if src.exists():
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src)

    print(f"created {output}")
    print(f"image size {width}x{height}")


if __name__ == "__main__":
    sys.exit(main())
