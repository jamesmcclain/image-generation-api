#!/usr/bin/env bash
#
# run.sh — Run the qwen-image-api container in API mode: the REST API (same
# as krea2-api's) on HOST_PORT, default 5002. For ComfyUI, use
# run-comfyui.sh instead (same image, same model directory).
#
# The container mounts a host model directory (MODEL_DIR) read-only at
# /models, in ComfyUI's folder layout. API mode needs three files; download
# them once with:
#
#   D=~/.cache/huggingface/qwen-image-2.1
#   hf download Comfy-Org/Qwen-Image-2.1 \
#     diffusion_models/qwen_image_2.1_int8_convrot.safetensors \
#     vae/qwen_image_2.1_vae_bf16.safetensors --local-dir "$D"
#   hf download unsloth/Qwen3-VL-8B-Instruct-GGUF Qwen3-VL-8B-Instruct-Q4_K_M.gguf \
#     --local-dir "$D/text_encoders"
#
# The text encoder must be a GGUF (or Comfy-Org's bf16 safetensors), not
# Comfy-Org's int8_convrot one, which stable-diffusion.cpp mis-loads (see the
# Dockerfile). It must also be Qwen3-VL-**8B**: the 4B one that Krea 2 uses
# loads without complaint but produces text features of the wrong width for
# Qwen-Image-2.1.
#
# Usage:
#   ./qwen-image-2.1/run.sh                          # default MODEL_DIR
#   ./qwen-image-2.1/run.sh /path/to/models          # overrides MODEL_DIR
#   HOST_PORT=5003 ./qwen-image-2.1/run.sh           # run beside krea2-api
#   OFFLOAD_TO_CPU=0 ./qwen-image-2.1/run.sh         # keep weights on the GPU
#
# MODEL_FILE / VAE_FILE / LLM_FILE pick the files, relative to MODEL_DIR.
# GGUF diffusion models (e.g. to fit a smaller card) mix freely with the
# other files, e.g.
#   MODEL_FILE=diffusion_models/qwen-image-2.1-Q4_0.gguf ./qwen-image-2.1/run.sh
# and a larger text encoder from the same unsloth repo works the same way:
#   LLM_FILE=text_encoders/Qwen3-VL-8B-Instruct-Q8_0.gguf ./qwen-image-2.1/run.sh
#
# API example, from the host (or anywhere that can reach it):
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

MODEL_DIR="${1:-${MODEL_DIR:-${HOME}/.cache/huggingface/qwen-image-2.1}}"
MODEL_FILE="${MODEL_FILE:-diffusion_models/qwen_image_2.1_int8_convrot.safetensors}"
VAE_FILE="${VAE_FILE:-vae/qwen_image_2.1_vae_bf16.safetensors}"
LLM_FILE="${LLM_FILE:-text_encoders/Qwen3-VL-8B-Instruct-Q4_K_M.gguf}"

for f in "${MODEL_FILE}" "${VAE_FILE}" "${LLM_FILE}"; do
    if [ ! -f "${MODEL_DIR}/${f}" ]; then
        echo "Error: file not found at ${MODEL_DIR}/${f}." >&2
        echo "Set MODEL_DIR/MODEL_FILE/VAE_FILE/LLM_FILE to point at your downloaded weights." >&2
        exit 1
    fi
done

# Build the image if it doesn't already exist locally. The build context is
# this script's own directory, so it works from any working directory.
if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    echo "Image ${IMAGE_NAME} not found locally — building it now..."
    docker build -t "${IMAGE_NAME}" "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

docker run -it --rm \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    -v "${MODEL_DIR}:/models:ro" \
    --user "$(id -u):$(id -g)" \
    -e MODEL_PATH="/models/${MODEL_FILE}" \
    -e VAE_PATH="/models/${VAE_FILE}" \
    -e LLM_PATH="/models/${LLM_FILE}" \
    -e OFFLOAD_TO_CPU="${OFFLOAD_TO_CPU:-1}" \
    -e SD_EXTRA_ARGS="${SD_EXTRA_ARGS:-}" \
    -p "${HOST_PORT}:5002" \
    "${IMAGE_NAME}" api
