# Dockerfile — Qwen-Image-2.1 text-to-image REST API via Hugging Face diffusers
#
# Serves the SAME REST API as the Krea 2 image in ../krea2/ (POST
# /generate, GET /health on port 5002), so clients can point at either one.
#
# Build (from the repo root):
#   docker build -t qwen-image-api qwen-image-2.1/
#
# Run (model weights are NOT baked in — the bf16 diffusers checkpoint is
# ~33 GB — so mount a local snapshot of the Hugging Face repo instead):
#   hf download Qwen/Qwen-Image-2.1 --local-dir /path/to/Qwen-Image-2.1
#   docker run --gpus all -p 5002:5002 \
#     -v /path/to/Qwen-Image-2.1:/models/Qwen-Image-2.1:ro \
#     qwen-image-api
# (or just use qwen-image-2.1/run.sh, which does the above.)
#
# https://huggingface.co/Qwen/Qwen-Image-2.1
#
# Why not stable-diffusion.cpp like the Krea 2 image? Qwen-Image-2.1 is a
# new architecture (32-layer single-stream 7B DiT with a Qwen3-VL 8B text
# encoder and its own VAE) that sd.cpp has no support for, and there are no
# GGUF quants of it. Its reference implementation is diffusers'
# QwenImage21Pipeline, which has not shipped in a diffusers release yet, so
# diffusers is installed from a pinned commit on main.

# The PyTorch wheels below bundle their own CUDA 12.8 user-space libraries,
# so only the small "base" CUDA image is needed: it provides the
# NVIDIA_* env vars the container toolkit uses to inject the host driver for
# `docker run --gpus all` (and makes it refuse to start on a host driver too
# old for CUDA 12.8, instead of failing cryptically at first request).
FROM nvidia/cuda:12.8.1-base-ubuntu22.04

# Pinned so a rebuild next month produces the same stack. DIFFUSERS_REF is a
# commit on huggingface/diffusers main that contains QwenImage21Pipeline;
# bump it (and TRANSFORMERS_VERSION, which must be >= 5.17 for this model's
# Qwen3-VL text encoder per the model card) together.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
ARG TRANSFORMERS_VERSION=5.17.0
ARG ACCELERATE_VERSION=1.15.0
ARG DIFFUSERS_REF=8b3c707ebd3ec4881f4190cf42931da07eaf3b65

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade pip

# torch/torchvision first, from the CUDA 12.8 index (PyPI's default Linux
# wheels are CUDA 13 now, which would require a >= 580 host driver).
# torchvision is needed by transformers' Qwen3-VL processor.
RUN python3 -m pip install \
    --index-url "${TORCH_INDEX_URL}" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}"

# diffusers is installed from a GitHub source archive of the pinned commit,
# so git doesn't need to be in the image. torch is already satisfied above,
# so none of these pulls in a second (CUDA 13) torch from PyPI.
RUN python3 -m pip install \
    "transformers==${TRANSFORMERS_VERSION}" \
    "accelerate==${ACCELERATE_VERSION}" \
    "https://github.com/huggingface/diffusers/archive/${DIFFUSERS_REF}.zip" \
    flask \
    pillow

WORKDIR /app

COPY server.py /app/server.py

# MODEL_PATH is a local diffusers-format snapshot of Qwen/Qwen-Image-2.1 (the
# directory containing model_index.json and the transformer/, text_encoder/,
# vae/, processor/, scheduler/ subfolders). Nothing is fetched at runtime:
# HF_HUB_OFFLINE makes sure a missing mount fails fast with a clear /health
# error instead of silently starting a 33 GB download inside the container.
#
# OFFLOAD picks the VRAM/speed trade-off. In bf16 the transformer is ~14 GB,
# the text encoder ~17.5 GB and the VAE ~1.4 GB:
#   model       (default) keeps whole components in system RAM and moves each
#               onto the GPU only while it runs. Fits a 24 GB card (e.g. RTX
#               4090); needs ~40 GB of free host RAM.
#   sequential  streams individual layers onto the GPU. Works with far less
#               VRAM, but is much slower.
#   none        everything resident on the GPU. Fastest; needs ~40+ GB VRAM
#               (e.g. A6000 48 GB, A100/H100).
#
# HF_HOME/HOME point somewhere writable because run.sh starts the container
# as the invoking host user (--user), who has no home directory in here.
ENV MODEL_PATH=/models/Qwen-Image-2.1 \
    OFFLOAD=model \
    GENERATION_TIMEOUT=600 \
    HF_HUB_OFFLINE=1 \
    HF_HOME=/tmp/huggingface \
    HOME=/tmp

EXPOSE 5002

CMD ["python3", "/app/server.py"]

# docker build -t qwen-image-api qwen-image-2.1/
