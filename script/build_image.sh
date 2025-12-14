#!/usr/bin/env bash
set -euo pipefail

# Build script for VMEvalKit Docker image
# Usage: ./scripts/build_image.sh

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "Building vmevalkit:latest from Dockerfile..."
docker build -t vmevalkit:latest -f Dockerfile .

echo "Build finished."

