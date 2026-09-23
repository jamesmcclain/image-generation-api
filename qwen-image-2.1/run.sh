#!/usr/bin/env bash
#
# run.sh — Run the qwen-image-api container in one of its two modes:
#
#   api      (default) the REST API (same as krea2-api's) on HOST_PORT,
#            default 5002
#   comfyui  ComfyUI on HOST_PORT, default 8188
#
# Both modes mount the same host model directory (MODEL_DIR) read-only at
# /models. It uses ComfyUI's folder layout; download the recommended files
# (Comfy-Org's INT8 convrot set, ~17 GB, used by both modes) once with:
#
#   hf download Comfy-Org/Qwen-Image-2.1 \
#     diffusion_models/qwen_image_2.1_int8_convrot.safetensors \
#     text_encoders/qwen3vl_8b_int8_convrot.safetensors \
#     vae/qwen_image_2.1_vae_bf16.safetensors \
#     --local-dir ~/.cache/huggingface/qwen-image-2.1
#
# Usage:
#   ./qwen-image-2.1/run.sh                          # api mode, default MODEL_DIR
#   ./qwen-image-2.1/run.sh comfyui                  # ComfyUI mode
#   ./qwen-image-2.1/run.sh api /path/to/models      # overrides MODEL_DIR
#   ./qwen-image-2.1/run.sh comfyui --lowvram        # extra args go to ComfyUI
#   HOST_PORT=5003 ./qwen-image-2.1/run.sh           # run beside krea2-api
#   OFFLOAD_TO_CPU=0 ./qwen-image-2.1/run.sh         # keep weights on the GPU
#
# GGUF diffusion models (e.g. to fit a smaller card) work in both modes and
# mix freely with the safetensors text encoder and VAE. Put the .gguf in
# MODEL_DIR/diffusion_models/, then:
#   - api mode: name it with MODEL_FILE (relative to MODEL_DIR), e.g.
#       MODEL_FILE=diffusion_models/qwen-image-2.1-Q4_K_M.gguf ./qwen-image-2.1/run.sh
#     (LLM_FILE likewise accepts a Qwen3-VL-8B-Instruct GGUF text encoder.)
#   - comfyui mode: in the workflow, replace the "Load Diffusion Model"
#     (UNETLoader) node with "Unet Loader (GGUF)" and pick the file there.
#
# ComfyUI mode keeps what it writes (outputs, uploaded inputs, workflows,
# settings, extra custom nodes) in COMFYUI_DATA_DIR on the host, default
# ~/.local/share/qwen-image-comfyui. Open http://localhost:8188 and load the
# "Qwen Image 2.1" template from the workflow templates browser. With a GGUF
# diffusion model on a 24 GB card, you can also set the text encoder's
# "Load CLIP" node to device "cpu": it runs once per prompt, so this frees
# ~9 GB of VRAM at little cost in speed.
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

MODE="api"
if [ "$#" -gt 0 ] && { [ "$1" = "api" ] || [ "$1" = "comfyui" ]; }; then
    MODE="$1"
    shift
fi

IMAGE_NAME="${IMAGE_NAME:-qwen-image-api}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen-image-${MODE}}"
MODEL_DIR="${MODEL_DIR:-${HOME}/.cache/huggingface/qwen-image-2.1}"
# In api mode, a leading non-flag argument is the model directory (like
# krea2's run.sh). In comfyui mode, all remaining arguments go to ComfyUI.
if [ "${MODE}" = "api" ] && [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; then
    MODEL_DIR="$1"
    shift
fi

MODEL_FILE="${MODEL_FILE:-diffusion_models/qwen-image-2.1-Q4_0.gguf}"
VAE_FILE="${VAE_FILE:-vae/qwen_image_2.1_vae_bf16.safetensors}"
LLM_FILE="${LLM_FILE:-text_encoders/Qwen3-VL-8B-Instruct-Q4_K_M.gguf}"

if [ ! -d "${MODEL_DIR}" ]; then
    echo "Error: model directory ${MODEL_DIR} not found." >&2
    echo "Set MODEL_DIR to where you downloaded the model files (see the top of this script)." >&2
    exit 1
fi

if [ "${MODE}" = "api" ]; then
    for f in "${MODEL_FILE}" "${VAE_FILE}" "${LLM_FILE}"; do
        if [ ! -f "${MODEL_DIR}/${f}" ]; then
            echo "Error: file not found at ${MODEL_DIR}/${f}." >&2
            echo "Set MODEL_DIR/MODEL_FILE/VAE_FILE/LLM_FILE to point at your downloaded weights." >&2
            exit 1
        fi
    done
fi

# Build the image if it doesn't already exist locally. The build context is
# this script's own directory, so it works from any working directory.
if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    echo "Image ${IMAGE_NAME} not found locally — building it now..."
    docker build -t "${IMAGE_NAME}" "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

RUN_ARGS=(
    -it --rm
    --name "${CONTAINER_NAME}"
    --gpus all
    -v "${MODEL_DIR}:/models:ro"
    --user "$(id -u):$(id -g)"
)

if [ "${MODE}" = "api" ]; then
    HOST_PORT="${HOST_PORT:-5002}"
    RUN_ARGS+=(
        -e MODEL_PATH="/models/${MODEL_FILE}"
        -e VAE_PATH="/models/${VAE_FILE}"
        -e LLM_PATH="/models/${LLM_FILE}"
        -e OFFLOAD_TO_CPU="${OFFLOAD_TO_CPU:-1}"
        -e SD_EXTRA_ARGS="${SD_EXTRA_ARGS:-}"
        -p "${HOST_PORT}:5002"
    )
else
    HOST_PORT="${HOST_PORT:-8188}"
    COMFYUI_DATA_DIR="${COMFYUI_DATA_DIR:-${HOME}/.local/share/qwen-image-comfyui}"
    mkdir -p "${COMFYUI_DATA_DIR}"
    RUN_ARGS+=(
        -v "${COMFYUI_DATA_DIR}:/data"
        -p "${HOST_PORT}:8188"
    )
fi

docker run "${RUN_ARGS[@]}" "${IMAGE_NAME}" "${MODE}" "$@"
