#!/usr/bin/env python3
"""Render lightweight turntable preview videos for exported 3DGS assets."""

from __future__ import annotations

import argparse
import math
import struct
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def read_opensplat_ply(path: Path, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        header = bytearray()
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY header ended unexpectedly")
            header.extend(line)
            if line.strip() == b"end_header":
                break
        header_text = header.decode("latin1")
        vertex_count = 0
        properties: list[tuple[str, str]] = []
        in_vertex = False
        for line in header_text.splitlines():
            parts = line.split()
            if not parts:
                continue
            if parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
                in_vertex = True
                continue
            if parts[0] == "element" and parts[1] != "vertex":
                in_vertex = False
            if in_vertex and parts[0] == "property":
                properties.append((parts[1], parts[2]))

        if not properties or any(kind != "float" for kind, _name in properties):
            raise ValueError("Only binary little-endian float PLY files are supported")

        prop_names = [name for _kind, name in properties]
        stride = 4 * len(prop_names)
        indices = np.linspace(0, max(vertex_count - 1, 0), min(max_points, vertex_count), dtype=np.int64)
        rng = np.random.default_rng(seed)
        if vertex_count > max_points:
            indices = np.sort(rng.choice(vertex_count, size=max_points, replace=False))

        positions = np.empty((len(indices), 3), dtype=np.float32)
        colors = np.empty((len(indices), 3), dtype=np.float32)
        name_to_idx = {name: idx for idx, name in enumerate(prop_names)}
        sh0 = 0.28209479177387814
        for out_i, src_i in enumerate(indices):
            f.seek(len(header) + int(src_i) * stride)
            row = struct.unpack("<" + "f" * len(prop_names), f.read(stride))
            positions[out_i] = [row[name_to_idx["x"]], row[name_to_idx["y"]], row[name_to_idx["z"]]]
            dc = np.array(
                [row[name_to_idx["f_dc_0"]], row[name_to_idx["f_dc_1"]], row[name_to_idx["f_dc_2"]]],
                dtype=np.float32,
            )
            colors[out_i] = np.clip(dc * sh0 + 0.5, 0.0, 1.0)

    return normalize_points(positions), colors


def read_sugar_obj_points(obj_path: Path, texture_path: Path, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    faces: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []

    with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                _tag, x, y, z = line.split(maxsplit=3)
                vertices.append((float(x), float(y), float(z)))
            elif line.startswith("vt "):
                parts = line.split()
                uvs.append((float(parts[1]), float(parts[2])))
            elif line.startswith("f "):
                refs = []
                for token in line.split()[1:4]:
                    items = token.split("/")
                    refs.append((int(items[0]) - 1, int(items[1]) - 1 if len(items) > 1 and items[1] else -1))
                faces.append((refs[0], refs[1], refs[2]))

    verts = np.asarray(vertices, dtype=np.float32)
    tex_uvs = np.asarray(uvs, dtype=np.float32)
    texture = np.asarray(Image.open(texture_path).convert("RGB"), dtype=np.float32) / 255.0
    tex_h, tex_w = texture.shape[:2]

    rng = np.random.default_rng(seed)
    face_indices = np.linspace(0, max(len(faces) - 1, 0), min(max_points, len(faces)), dtype=np.int64)
    if len(faces) > max_points:
        face_indices = rng.choice(len(faces), size=max_points, replace=False)

    points = np.empty((len(face_indices), 3), dtype=np.float32)
    colors = np.empty((len(face_indices), 3), dtype=np.float32)
    bary = rng.random((len(face_indices), 2), dtype=np.float32)
    over = bary.sum(axis=1) > 1.0
    bary[over] = 1.0 - bary[over]
    w0 = 1.0 - bary[:, 0] - bary[:, 1]
    w1 = bary[:, 0]
    w2 = bary[:, 1]

    for out_i, face_i in enumerate(face_indices):
        face = faces[int(face_i)]
        vi = [face[j][0] for j in range(3)]
        ti = [face[j][1] for j in range(3)]
        points[out_i] = verts[vi[0]] * w0[out_i] + verts[vi[1]] * w1[out_i] + verts[vi[2]] * w2[out_i]
        if all(idx >= 0 for idx in ti):
            uv = tex_uvs[ti[0]] * w0[out_i] + tex_uvs[ti[1]] * w1[out_i] + tex_uvs[ti[2]] * w2[out_i]
            x = int(np.clip(uv[0] * (tex_w - 1), 0, tex_w - 1))
            y = int(np.clip((1.0 - uv[1]) * (tex_h - 1), 0, tex_h - 1))
            colors[out_i] = texture[y, x]
        else:
            colors[out_i] = (0.7, 0.7, 0.7)

    return normalize_points(points), colors


def read_colored_mesh_ply(path: Path, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Open3D is required for colored mesh PLY previews") from exc

    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    if vertices.size == 0 or faces.size == 0:
        raise ValueError(f"{path} does not contain a triangle mesh")

    vertex_colors = np.asarray(mesh.vertex_colors, dtype=np.float32)
    if len(vertex_colors) != len(vertices):
        vertex_colors = np.full((len(vertices), 3), 0.68, dtype=np.float32)

    rng = np.random.default_rng(seed)
    face_vertices = vertices[faces]
    areas = np.linalg.norm(
        np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0]),
        axis=1,
    )
    if areas.sum() > 0:
        probabilities = areas / areas.sum()
        face_indices = rng.choice(len(faces), size=min(max_points, len(faces)), replace=len(faces) < max_points, p=probabilities)
    else:
        face_indices = rng.choice(len(faces), size=min(max_points, len(faces)), replace=len(faces) < max_points)

    bary = rng.random((len(face_indices), 2), dtype=np.float32)
    over = bary.sum(axis=1) > 1.0
    bary[over] = 1.0 - bary[over]
    w0 = (1.0 - bary[:, 0] - bary[:, 1])[:, None]
    w1 = bary[:, 0:1]
    w2 = bary[:, 1:2]

    sample_faces = faces[face_indices]
    sample_vertices = vertices[sample_faces]
    sample_colors = vertex_colors[sample_faces]
    points = sample_vertices[:, 0] * w0 + sample_vertices[:, 1] * w1 + sample_vertices[:, 2] * w2
    colors = sample_colors[:, 0] * w0 + sample_colors[:, 1] * w1 + sample_colors[:, 2] * w2
    return normalize_points(points.astype(np.float32)), np.clip(colors.astype(np.float32), 0.0, 1.0)


def normalize_points(points: np.ndarray) -> np.ndarray:
    center = np.median(points, axis=0, keepdims=True)
    shifted = points - center
    radius = np.percentile(np.linalg.norm(shifted, axis=1), 98)
    if radius <= 1e-6:
        radius = np.max(np.linalg.norm(shifted, axis=1)) + 1e-6
    return shifted / radius


def render_points(
    points: np.ndarray,
    colors: np.ndarray,
    out_dir: Path,
    title: str,
    frames: int,
    width: int,
    height: int,
    point_size: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    draw_order_base = np.arange(len(points))

    for frame in range(frames):
        theta = 2.0 * math.pi * frame / frames
        c, s = math.cos(theta), math.sin(theta)
        rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
        p = points @ rot.T
        depth = p[:, 2] + 3.2
        valid = depth > 0.1
        p = p[valid]
        d = depth[valid]
        col = colors[valid]

        focal = 0.72 * min(width, height)
        x = (width * 0.5 + focal * p[:, 0] / d).astype(np.int32)
        y = (height * 0.52 - focal * p[:, 1] / d).astype(np.int32)
        inside = (x >= -point_size) & (x < width + point_size) & (y >= -point_size) & (y < height + point_size)
        x, y, d, col = x[inside], y[inside], d[inside], col[inside]
        shade = np.clip(1.15 - 0.22 * d[:, None], 0.62, 1.0)
        col = np.clip(col * shade + 0.035, 0.0, 1.0)
        order = np.argsort(d)[::-1]

        img = Image.new("RGB", (width, height), (239, 242, 245))
        draw = ImageDraw.Draw(img, "RGBA")
        draw.rectangle((0, 0, width, 48), fill=(255, 255, 255, 224))
        draw.text((18, 15), title, fill=(32, 38, 45))
        draw.ellipse((width - 86, 14, width - 62, 38), outline=(15, 118, 110), width=2)
        draw.line((width - 74, 26, width - 74 + 9 * math.cos(theta), 26 + 9 * math.sin(theta)), fill=(15, 118, 110), width=2)

        for idx in order:
            rgba = tuple((col[idx] * 255).astype(np.uint8).tolist()) + (225,)
            px, py = int(x[idx]), int(y[idx])
            draw.rectangle((px - point_size, py - point_size, px + point_size, py + point_size), fill=rgba)

        # Subtle ground reference.
        gy = int(height * 0.78)
        draw.line((int(width * 0.18), gy, int(width * 0.82), gy), fill=(190, 198, 207, 140), width=1)
        img.save(out_dir / f"frame_{frame:04d}.png", optimize=False)

        # Keep linters quiet for the intentionally precomputed array.
        _ = draw_order_base


def encode_video(frames_dir: Path, output: Path, fps: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", choices=("sugar", "opensplat", "gof-mesh", "all"), default="all")
    parser.add_argument("--root", type=Path, default=Path("/home/aa/cases/3DGS"))
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--max-points", type=int, default=90000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.asset in ("sugar", "all"):
        base = args.root / "outputs/sugar/truck"
        obj = base / "sugarfine_3Dgs7000_densityestim02_sdfnorm02_level03_decim200000_normalconsistency01_gaussperface6.obj"
        tex = base / "sugarfine_3Dgs7000_densityestim02_sdfnorm02_level03_decim200000_normalconsistency01_gaussperface6.png"
        points, colors = read_sugar_obj_points(obj, tex, args.max_points, args.seed)
        frames_dir = base / "preview_frames"
        render_points(points, colors, frames_dir, "SuGaR textured OBJ mesh preview", args.frames, args.width, args.height, 1)
        encode_video(frames_dir, base / "sugar_textured_obj_turntable.mp4", args.fps)

    if args.asset in ("opensplat", "all"):
        base = args.root / "outputs/opensplat"
        points, colors = read_opensplat_ply(base / "truck_30000.ply", args.max_points, args.seed)
        frames_dir = base / "preview_frames"
        render_points(points, colors, frames_dir, "OpenSplat Gaussian PLY point preview", args.frames, args.width, args.height, 1)
        encode_video(frames_dir, base / "opensplat_truck_30000_turntable.mp4", args.fps)

    if args.asset in ("gof-mesh", "all"):
        base = args.root / "outputs/gof/truck"
        points, colors = read_colored_mesh_ply(base / "gof_truck_tsdf_7000.ply", args.max_points, args.seed)
        frames_dir = base / "tsdf_preview_frames"
        render_points(points, colors, frames_dir, "GOF TSDF mesh preview", args.frames, args.width, args.height, 1)
        encode_video(frames_dir, base / "gof_truck_tsdf_7000_turntable.mp4", args.fps)


if __name__ == "__main__":
    main()
