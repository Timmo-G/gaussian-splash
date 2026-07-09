#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/aa/cases/3DGS}"
REPO_DIR="${REPO_DIR:-$ROOT_DIR/repos/2d-gaussian-splatting}"
SOURCE_PATH="${1:-$ROOT_DIR/data/official_gs/tandt/truck}"
SCENE_NAME="${2:-$(basename "$SOURCE_PATH")}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/2dgs}"
MODEL_PATH="${3:-$OUTPUT_DIR/$SCENE_NAME}"
GPU="${GPU:-1}"
ITERATIONS="${ITERATIONS:-30000}"
MESH_RES="${MESH_RES:-512}"
CONDA_ENV="${CONDA_ENV:-gs-official}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

source /home/aa/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export CUDA_HOME
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "$OUTPUT_DIR" "$MODEL_PATH" "$MPLCONFIGDIR"
cd "$REPO_DIR"

echo "==> Verifying 2DGS dependencies"
python -c "import torch, open3d, trimesh, mediapy; import diff_surfel_rasterization; from simple_knn._C import distCUDA2; print('2dgs deps ok', torch.__version__, torch.version.cuda)"

echo "==> Training $SCENE_NAME"
CUDA_VISIBLE_DEVICES="$GPU" python train.py -s "$SOURCE_PATH" -m "$MODEL_PATH" --iterations "$ITERATIONS" --test_iterations 7000 "$ITERATIONS" --save_iterations 7000 "$ITERATIONS" --data_device cpu

echo "==> Rendering images and trajectory videos for $SCENE_NAME"
CUDA_VISIBLE_DEVICES="$GPU" python render.py -s "$SOURCE_PATH" -m "$MODEL_PATH" --iteration "$ITERATIONS" --render_path --skip_mesh

echo "==> Exporting unbounded mesh for $SCENE_NAME"
CUDA_VISIBLE_DEVICES="$GPU" python render.py -s "$SOURCE_PATH" -m "$MODEL_PATH" --iteration "$ITERATIONS" --skip_train --skip_test --unbounded --mesh_res "$MESH_RES"

TRAIN_VIDEO="$MODEL_PATH/${SCENE_NAME}_train_${ITERATIONS}.mp4"
RENDER_GLOB="$MODEL_PATH/train/ours_${ITERATIONS}/renders/*.png"

echo "==> Creating training-view video $TRAIN_VIDEO"
ffmpeg -y -framerate 30 -pattern_type glob -i "$RENDER_GLOB" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p "$TRAIN_VIDEO"

echo "==> Done $SCENE_NAME"
echo "    model: $MODEL_PATH"
echo "    train video: $TRAIN_VIDEO"
echo "    trajectory color: $MODEL_PATH/traj/ours_${ITERATIONS}/render_traj_color.mp4"
echo "    trajectory depth: $MODEL_PATH/traj/ours_${ITERATIONS}/render_traj_depth.mp4"
echo "    mesh: $MODEL_PATH/train/ours_${ITERATIONS}/fuse_unbounded_post.ply"
