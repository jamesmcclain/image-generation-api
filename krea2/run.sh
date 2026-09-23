#!/usr/bin/env bash
#
# run.sh — Build (if needed) and run the krea2-api container so that:
#   1. It can read the GGUF model weights from the HOST, by mounting a
#      host directory of quantized models into the container read-only.
#   2. Its REST API (port 5002) is reachable from the host and other
#      machines on the same network, via a port mapping.
#
# Usage:
#   ./krea2/run.sh                          # uses default MODEL_DIR
#   ./krea2/run.sh /path/to/gguf-models     # overrides MODEL_DIR for this run
#   MODEL_FILE=other-quant.gguf ./krea2/run.sh
#
# (MODEL_DIR defaults to $HOME/.cache/huggingface/gguf if not set and no
# argument is given. Krea2 needs three files present in MODEL_DIR:
# MODEL_FILE (the diffusion GGUF), VAE_FILE (Wan2.1 VAE), and LLM_FILE
# (Qwen3-VL 4B text encoder GGUF) — each overridable via env var, with
# defaults shown below.)
#
# Then, from the host (or anywhere that can reach the host on port 5002):
#   curl -X POST http://localhost:5002/generate \
#     -H "Content-Type: application/json" \
#     -d '{"prompt": "a red fox sitting in fresh snow, golden hour, photorealistic"}' \
#     -o output.png
#
# output.png will be saved locally once generation finishes.

set -euo pipefail

IMAGE_NAME="krea2-api"
CONTAINER_NAME="krea2-api"
HOST_PORT=5002
CONTAINER_PORT=5002

MODEL_DIR="${1:-${MODEL_DIR:-${HOME}/.cache/huggingface/gguf}}"
MODEL_FILE="${2:-${MODEL_FILE:-krea2_turbo-Q4_K_M.gguf}}"
VAE_FILE="${VAE_FILE:-wan_2.1_vae.safetensors}"
LLM_FILE="${LLM_FILE:-Qwen3VL-4B-Instruct-Q4_K_M.gguf}"

for f in "${MODEL_FILE}" "${VAE_FILE}" "${LLM_FILE}"; do
    if [ ! -f "${MODEL_DIR}/${f}" ]; then
        echo "Error: file not found at ${MODEL_DIR}/${f}." >&2
        echo "Set MODEL_DIR/MODEL_FILE/VAE_FILE/LLM_FILE to point at your downloaded weights." >&2
        exit 1
    fi
done

# # Build the image if it doesn't already exist locally.
# if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
#     echo "Image ${IMAGE_NAME} not found locally — building it now..."
#     docker build -t "${IMAGE_NAME}" "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# fi

# # Remove any previous container with the same name so re-running is idempotent.
# docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

MOUNT_ARGS=(
    -v "${MODEL_DIR}:/models:ro"
)

# echo "Starting container '${CONTAINER_NAME}' in the foreground (Ctrl-C to stop)."
# echo "REST API will be available at: http://localhost:${HOST_PORT}"
# echo
# echo "Try it from another terminal:"
# echo "  curl -X POST http://localhost:${HOST_PORT}/generate \\"
# echo "    -H 'Content-Type: application/json' \\"
# echo "    -d '{\"prompt\": \"a red fox sitting in fresh snow, golden hour, photorealistic\"}' \\"
# echo "    -o output.png"
# echo

docker run -it --rm \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    -e MODEL_PATH="/models/${MODEL_FILE}" \
    -e VAE_PATH="/models/${VAE_FILE}" \
    -e LLM_PATH="/models/${LLM_FILE}" \
    "${MOUNT_ARGS[@]}" \
    --user "$(id -u):$(id -g)" \
    -p "${HOST_PORT}:${CONTAINER_PORT}" \
    "${IMAGE_NAME}"
