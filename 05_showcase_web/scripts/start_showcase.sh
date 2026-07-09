#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-8765}"

echo "Serving 3DGS showcase from ${ROOT_DIR}"
echo "Open http://127.0.0.1:${PORT}/showcase/index.html"
python3 -m http.server "${PORT}" --directory "${ROOT_DIR}"
