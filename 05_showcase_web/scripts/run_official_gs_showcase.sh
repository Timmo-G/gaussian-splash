#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/aa/cases/3DGS}"
REPO_DIR="${REPO_DIR:-$ROOT_DIR/repos/gaussian-splatting}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/official_gs}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/official_gs}"
GPU="${GPU:-1}"
ITERATIONS="${ITERATIONS:-30000}"
CONDA_ENV="${CONDA_ENV:-gs-official}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

source /home/aa/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "$OUTPUT_DIR"
cd "$REPO_DIR"

run_scene() {
  local name="$1"
  local source_path="$2"
  local model_path="$OUTPUT_DIR/$name"
  local video_path="$model_path/${name}_${ITERATIONS}.mp4"
  local render_glob="$model_path/train/ours_${ITERATIONS}/renders/*.png"

  echo "==> Training $name"
  CUDA_VISIBLE_DEVICES="$GPU" python train.py -s "$source_path" -m "$model_path" --iterations "$ITERATIONS" --data_device cpu

  echo "==> Rendering $name"
  CUDA_VISIBLE_DEVICES="$GPU" python render.py -s "$source_path" -m "$model_path"

  echo "==> Creating video $video_path"
  ffmpeg -y -framerate 30 -pattern_type glob -i "$render_glob" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p "$video_path"

  echo "==> Done $name"
  echo "    model: $model_path"
  echo "    video: $video_path"
}

run_scene "tandt_train" "$DATA_DIR/tandt/train"
run_scene "drjohnson" "$DATA_DIR/db/drjohnson"
