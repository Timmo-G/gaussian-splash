#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image


def iter_images(path: Path):
    suffixes = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix in suffixes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Nerfstudio-style images_N downscaled image folders.")
    parser.add_argument("--images", type=Path, required=True, help="Source images directory.")
    parser.add_argument("--factor", type=int, required=True, help="Integer downscale factor.")
    parser.add_argument("--output", type=Path, required=True, help="Output images_N directory.")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality.")
    args = parser.parse_args()

    if args.factor <= 1:
        raise ValueError("--factor must be greater than 1")

    images = args.images.resolve()
    output = args.output.resolve()
    files = iter_images(images)
    if not files:
        raise FileNotFoundError(f"No images found in {images}")

    for src in files:
        rel = src.relative_to(images)
        dst = output / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as image:
            width, height = image.size
            resized = image.resize((math.floor(width / args.factor), math.floor(height / args.factor)), Image.LANCZOS)
            save_kwargs = {"quality": args.quality} if dst.suffix.lower() in {".jpg", ".jpeg"} else {}
            resized.save(dst, **save_kwargs)

    print(f"created {len(files)} images in {output}")


if __name__ == "__main__":
    main()
