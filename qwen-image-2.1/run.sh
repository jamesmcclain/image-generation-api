#!/usr/bin/env bash
#
# run.sh — Run the qwen-image-api container so that:
#   1. It can read the Qwen-Image-2.1 weights from the HOST, by mounting a
#      local diffusers snapshot of the model into the container read-only.
#   2. Its REST API (same as the Krea 2 image's) is reachable from the host
#      and other machines on the same network, via a port mapping.
#
# Usage:
#   ./qwen-image-2.1/run.sh                          # uses default MODEL_DIR
#   ./qwen-image-2.1/run.sh /path/to/Qwen-Image-2.1  # overrides MODEL_DIR
#   HOST_PORT=5003 ./qwen-image-2.1/run.sh           # run beside krea2-api
#   OFFLOAD=none ./qwen-image-2.1/run.sh             # 40+ GB GPU: no offload
#
# MODEL_DIR defaults to $HOME/.cache/huggingface/Qwen-Image-2.1 and must be
# the model's diffusers directory (the one containing model_index.json).
# Download it once with:
#   hf download Qwen/Qwen-Image-2.1 --local-dir ~/.cache/huggingface/Qwen-Image-2.1
#
# HOST_PORT defaults to 5002 — the same port as krea2-api, so this is a
# drop-in replacement for existing clients. Set HOST_PORT to something else
# (e.g. 5003) to run both containers at once.
#
# Then, from the host (or anywhere that can reach the host on HOST_PORT):
#   curl -X POST http://localhost:5002/generate \
#     -H "Content-Type: application/json" \
#     -d '{"prompt": "a red fox sitting in fresh snow, golden hour, photorealistic"}' \
#     -o output.png
#
# The first request after startup waits for the weights to load (GET
# /health returns {"status": "loading"} until then).

set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-qwen-image-api}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen-image-api}"
HOST_PORT="${HOST_PORT:-5002}"
CONTAINER_PORT=5002

MODEL_DIR="${1:-${MODEL_DIR:-${HOME}/.cache/huggingface/Qwen-Image-2.1}}"
OFFLOAD="${OFFLOAD:-model}"

if [ ! -f "${MODEL_DIR}/model_index.json" ]; then
    echo "Error: ${MODEL_DIR}/model_index.json not found." >&2
    echo "Set MODEL_DIR (or pass it as the first argument) to a local snapshot of" >&2
    echo "Qwen/Qwen-Image-2.1, e.g.:" >&2
    echo "  hf download Qwen/Qwen-Image-2.1 --local-dir ${MODEL_DIR}" >&2
    exit 1
fi

# Build the image if it doesn't already exist locally. The build context is
# this script's own directory, so it works from any working directory.
if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    echo "Image ${IMAGE_NAME} not found locally — building it now..."
    docker build -t "${IMAGE_NAME}" "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

docker run -it --rm \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    -e OFFLOAD="${OFFLOAD}" \
    -v "${MODEL_DIR}:/models/Qwen-Image-2.1:ro" \
    --user "$(id -u):$(id -g)" \
    -p "${HOST_PORT}:${CONTAINER_PORT}" \
    "${IMAGE_NAME}"
