#!/usr/bin/env python3
"""Build compact point-cloud assets for the offline showcase viewer."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_asset_turntables import (  # noqa: E402
    read_colored_mesh_ply,
    read_opensplat_ply,
    read_sugar_obj_points,
)


def quantize_scene(points: np.ndarray, colors: np.ndarray) -> dict:
    points = np.clip(points, -1.35, 1.35)
    q_points = np.round((points + 1.35) / 2.7 * 65535.0).astype("<u2")
    q_colors = np.clip(np.round(colors * 255.0), 0, 255).astype("u1")
    return {
        "count": int(points.shape[0]),
        "points": base64.b64encode(q_points.tobytes()).decode("ascii"),
        "colors": base64.b64encode(q_colors.tobytes()).decode("ascii"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/aa/cases/3DGS"))
    parser.add_argument("--out", type=Path, default=Path("/home/aa/cases/3DGS/showcase/assets/interactive_points.json"))
    parser.add_argument("--out-js", type=Path, default=Path("/home/aa/cases/3DGS/showcase/assets/interactive_points.js"))
    parser.add_argument("--max-points", type=int, default=45000)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    root = args.root
    scenes: dict[str, dict] = {}

    sugar_dir = root / "outputs/sugar/truck"
    sugar_obj = sugar_dir / "sugarfine_3Dgs7000_densityestim02_sdfnorm02_level03_decim200000_normalconsistency01_gaussperface6.obj"
    sugar_tex = sugar_dir / "sugarfine_3Dgs7000_densityestim02_sdfnorm02_level03_decim200000_normalconsistency01_gaussperface6.png"
    points, colors = read_sugar_obj_points(sugar_obj, sugar_tex, args.max_points, args.seed)
    scenes["sugar"] = {
        "label": "SuGaR Textured Mesh",
        "kind": "OBJ mesh sample",
        **quantize_scene(points, colors),
    }

    points, colors = read_opensplat_ply(root / "outputs/opensplat/truck_30000.ply", args.max_points, args.seed)
    scenes["opensplat"] = {
        "label": "OpenSplat Gaussian PLY",
        "kind": "Gaussian center sample",
        **quantize_scene(points, colors),
    }

    points, colors = read_colored_mesh_ply(root / "outputs/gof/truck/gof_truck_tsdf_7000.ply", args.max_points, args.seed)
    scenes["gof"] = {
        "label": "GOF TSDF Mesh",
        "kind": "PLY mesh sample",
        **quantize_scene(points, colors),
    }

    points, colors = read_colored_mesh_ply(root / "outputs/2dgs/truck/train/ours_30000/fuse_unbounded_post.ply", args.max_points, args.seed)
    scenes["2dgs"] = {
        "label": "2DGS Unbounded Mesh",
        "kind": "PLY mesh sample",
        **quantize_scene(points, colors),
    }

    payload = json.dumps({"version": 1, "range": 1.35, "scenes": scenes}, separators=(",", ":"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload)
    args.out_js.parent.mkdir(parents=True, exist_ok=True)
    args.out_js.write_text("window.INTERACTIVE_POINTS=" + payload + ";\n")
    print(f"Wrote {args.out}")
    print(f"Wrote {args.out_js}")
    for key, scene in scenes.items():
        print(key, scene["count"], scene["label"])


if __name__ == "__main__":
    main()
