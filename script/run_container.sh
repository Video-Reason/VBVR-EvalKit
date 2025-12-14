#!/usr/bin/env bash
set -euo pipefail

# Run VMEvalKit container with sensible defaults
# Usage: ./scripts/run_container.sh [image] [container_name]
# Example: ./scripts/run_container.sh vmevalkit:latest vmevalkit_dev

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${1:-vmevalkit:latest}"
NAME="${2:-vmevalkit_run}"
HOST_WORKDIR="${3:-$ROOT_DIR}"
WORKDIR="/workspace"

# Optional host dirs (customize by env or edit script)
WEIGHTS_DIR="${WEIGHTS_DIR:-$HOME/vmeval_weights}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/vmevalkit}"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"
SHM_SIZE="${SHM_SIZE:-8g}"

# Create host cache directories if missing
mkdir -p "$WEIGHTS_DIR" "$CACHE_DIR"

# Stop & remove existing container with same name
if docker ps -a --format '{{.Names}}' | grep -xq "$NAME"; then
  echo "Removing existing container $NAME..."
  docker rm -f "$NAME" >/dev/null || true
fi

echo "Running container $NAME from image $IMAGE"
docker run -d \
  --name "$NAME" \
  --gpus all \
  --network host \
  --shm-size="$SHM_SIZE" \
  -v "$HOST_WORKDIR":"$WORKDIR" \
  -v "$WEIGHTS_DIR":/workspace/weights \
  -v "$CACHE_DIR":/workspace/.cache \
  --env-file "$ENV_FILE" \
  "$IMAGE" tail -f /dev/null

echo "Container $NAME started. Exec into it with: docker exec -it $NAME bash"
