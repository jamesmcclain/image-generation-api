#!/usr/bin/env bash
#
# run-comfyui.sh — Run the qwen-image-api container in ComfyUI mode, on
# HOST_PORT (default 8188). For the REST API, use run.sh instead (same image,
# same model directory).
#
# The container mounts a host model directory (MODEL_DIR) read-only at
# /models, in ComfyUI's folder layout. Download the files once with:
#
#   D=~/.cache/huggingface/qwen-image-2.1
#   hf download Comfy-Org/Qwen-Image-2.1 \
#     diffusion_models/qwen_image_2.1_int8_convrot.safetensors \
#     text_encoders/qwen3vl_8b_int8_convrot.safetensors \
#     vae/qwen_image_2.1_vae_bf16.safetensors --local-dir "$D"
#
# The text encoder must be a .safetensors file (Comfy-Org's int8_convrot or
# bf16 one): ComfyUI only recognizes Qwen-Image-2.1's Qwen3-VL-8B encoder by
# its vision weights, which a GGUF text encoder doesn't carry, so a GGUF gets
# wired up as a different model's encoder and produces wrong conditioning.
# (That's the opposite of run.sh, whose stable-diffusion.cpp backend needs the
# GGUF text encoder instead.)
#
# The diffusion model, on the other hand, can be a GGUF (e.g. to fit a
# smaller card): put it in $D/diffusion_models/ and, in the workflow, replace
# the "Load Diffusion Model" (UNETLoader) node with "Unet Loader (GGUF)".
#
# Usage:
#   ./qwen-image-2.1/run-comfyui.sh                    # default MODEL_DIR
#   ./qwen-image-2.1/run-comfyui.sh /path/to/models    # overrides MODEL_DIR
#   ./qwen-image-2.1/run-comfyui.sh --lowvram          # extra args go to ComfyUI
#   HOST_PORT=8189 ./qwen-image-2.1/run-comfyui.sh
#
# Open http://localhost:8188 and load a "Qwen Image 2.1" template from the
# Templates browser (left sidebar). With a GGUF diffusion model on a 24 GB
# card, you can also set the text encoder's "Load CLIP" node to device "cpu":
# it runs once per prompt, so this frees ~9 GB of VRAM at little cost in speed.
#
# ComfyUI keeps what it writes (outputs, uploaded inputs, workflows,
# settings, extra custom nodes) in COMFYUI_DATA_DIR on the host, default
# ~/.local/share/qwen-image-comfyui.

set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-qwen-image-api}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen-image-comfyui}"
HOST_PORT="${HOST_PORT:-8188}"
COMFYUI_DATA_DIR="${COMFYUI_DATA_DIR:-${HOME}/.local/share/qwen-image-comfyui}"

MODEL_DIR="${MODEL_DIR:-${HOME}/.cache/huggingface/qwen-image-2.1}"
# A leading non-flag argument is the model directory; everything else goes
# to ComfyUI's command line.
if [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; then
    MODEL_DIR="$1"
    shift
fi
VAE_FILE="${VAE_FILE:-vae/qwen_image_2.1_vae_bf16.safetensors}"
LLM_FILE="${LLM_FILE:-text_encoders/qwen3vl_8b_int8_convrot.safetensors}"

# ComfyUI picks the actual files in the workflow; these checks just catch a
# missing download (or a wrong MODEL_DIR) before the container starts.
for f in "${VAE_FILE}" "${LLM_FILE}"; do
    if [ ! -f "${MODEL_DIR}/${f}" ]; then
        echo "Error: file not found at ${MODEL_DIR}/${f}." >&2
        echo "Set MODEL_DIR/VAE_FILE/LLM_FILE to point at your downloaded weights." >&2
        exit 1
    fi
done
if ! compgen -G "${MODEL_DIR}/diffusion_models/*.safetensors" >/dev/null &&
   ! compgen -G "${MODEL_DIR}/diffusion_models/*.gguf" >/dev/null; then
    echo "Error: no .safetensors or .gguf diffusion model in ${MODEL_DIR}/diffusion_models/." >&2
    exit 1
fi

# Build the image if it doesn't already exist locally. The build context is
# this script's own directory, so it works from any working directory.
if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    echo "Image ${IMAGE_NAME} not found locally — building it now..."
    docker build -t "${IMAGE_NAME}" "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

mkdir -p "${COMFYUI_DATA_DIR}"

docker run -it --rm \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    -v "${MODEL_DIR}:/models:ro" \
    -v "${COMFYUI_DATA_DIR}:/data" \
    --user "$(id -u):$(id -g)" \
    -p "${HOST_PORT}:8188" \
    "${IMAGE_NAME}" comfyui "$@"
