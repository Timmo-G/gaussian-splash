# 3DGS Demo Bundle

Packaged on 2026-07-09.

## Open The Showcase

From this bundle directory, run:

```bash
scripts/start_showcase.sh
```

Then open:

```text
http://127.0.0.1:8765/showcase/index.html
```

You can pass a different port if needed:

```bash
scripts/start_showcase.sh 8766
```

The page embeds final MP4 demos, links exported geometry assets, and includes an offline vendored Three.js interactive viewer. Open it through HTTP instead of `file://` because the viewer uses ES modules.

## Contents

- `showcase/index.html`: unified local dashboard for all packaged demos.
- `showcase/interactive-viewer.js`: Three.js viewer for SuGaR OBJ, GOF TSDF mesh, and OpenSplat colored sampled points.
- `showcase/assets/interactive_points.js`: compact OpenSplat point sample used by the interactive viewer.
- `showcase/vendor/three/`: vendored Three.js `0.185.1`, OrbitControls, PLYLoader, and OBJLoader.
- `outputs/official_gs/`: Official Graphdeco 3DGS videos for truck, train, drjohnson, and playroom.
- `outputs/nerfstudio_showcase/truck/`: Nerfstudio splatfacto truck video, exported PLY, and config.
- `outputs/gsplat/truck/`: gsplat truck trajectory video and PLY.
- `outputs/gof/truck/`: Gaussian Opacity Fields truck video, Gaussian PLY, TSDF mesh PLY, and TSDF mesh preview MP4.
- `outputs/2dgs/truck/`: 2DGS truck train-view video, trajectory RGB/depth videos, and unbounded mesh PLY.
- `outputs/sugar/truck/`: SuGaR coarse mesh, refined textured OBJ/MTL/PNG, and OBJ turntable preview MP4.
- `outputs/opensplat/`: OpenSplat truck Gaussian PLY and point-preview turntable MP4.
- `scripts/`: helper scripts used during reproduction plus `start_showcase.sh`.

## Viewing Notes

- MP4 files open directly in any video player or browser.
- OBJ/MTL/PNG must stay in the same folder for textured mesh viewing in Blender, MeshLab, or Open3D tools.
- Gaussian PLY files are best opened in a Gaussian splat viewer such as SuperSplat or GaussianSplats3D-compatible tools.
- The 2DGS raw unbounded mesh is intentionally not loaded in the interactive viewer because it is large and visually noisy; it remains linked as an exported artifact.
- The interactive viewer shows a clear WebGL-unavailable status if the browser cannot create a WebGL context.
- Raw render frame folders and large training checkpoints are not included.

## Key Metrics

- Official 3DGS truck: 30000 iterations.
- Nerfstudio splatfacto truck: 7000 iterations.
- gsplat truck: checkpoint 6999, PSNR 25.564, SSIM 0.9188, LPIPS 0.036.
- GOF truck: 7000 iterations, L1 0.02829, PSNR 27.0993.
- GOF TSDF mesh: 390725 vertices, 721938 faces.
- 2DGS truck: 30000 iterations, train PSNR around 26.44.
