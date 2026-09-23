# Dockerfile — Qwen-Image-2.1 in one image, two launch-time modes:
#
#   api      (default) the same REST API as the Krea 2 image in ../krea2/
#            (POST /generate, GET /health on port 5002), backed by
#            stable-diffusion.cpp's sd-server, which keeps the model loaded
#            between requests
#   comfyui  ComfyUI on port 8188, with Qwen-Image-2.1 support built in
#
# Both modes read the SAME model files from a single /models mount, laid out
# the way ComfyUI expects. The recommended set is Comfy-Org's INT8 convrot
# files (~17 GB total; both stable-diffusion.cpp and ComfyUI run them natively):
#
#   hf download Comfy-Org/Qwen-Image-2.1 \
#     diffusion_models/qwen_image_2.1_int8_convrot.safetensors \
#     text_encoders/qwen3vl_8b_int8_convrot.safetensors \
#     vae/qwen_image_2.1_vae_bf16.safetensors \
#     --local-dir /path/to/qwen-image-models
#
# Build (from the repo root):
#   docker build -t qwen-image-api qwen-image-2.1/
#   docker build -t qwen-image-api --build-arg WITH_COMFYUI=0 qwen-image-2.1/   # API only, much smaller
#
# Run (or use qwen-image-2.1/run.sh, which does this for you):
#   docker run --gpus all -p 5002:5002 -v /path/to/qwen-image-models:/models:ro qwen-image-api
#   docker run --gpus all -p 8188:8188 -v /path/to/qwen-image-models:/models:ro \
#     -v /path/to/comfyui-data:/data qwen-image-api comfyui
#
# https://huggingface.co/Qwen/Qwen-Image-2.1
# https://huggingface.co/Comfy-Org/Qwen-Image-2.1
# https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF   (API mode only, see below)
# https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/qwen_image_2.1.md

# CUDA 12.8 is the oldest toolkit that can target RTX 50-series (sm_120);
# sd-server needs a host driver >= 570.
ARG CUDA_VERSION=12.8.1

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Qwen-Image-2.1 support landed in stable-diffusion.cpp on 2026-09-20
# (leejet/stable-diffusion.cpp#1994, alpha-channel fixes in #2021), so pin a
# commit at or after that rather than tracking master.
ARG SD_CPP_REF=c92d73c408515c94beef32161bb5960764fde7a0
RUN git clone https://github.com/leejet/stable-diffusion.cpp.git . \
    && git checkout "${SD_CPP_REF}" \
    && git submodule update --init --recursive

# - SD_CUDA: build the CUDA backend.
# - The bundled (patched) GGML is kept (SD_USE_UPSTREAM_GGML stays OFF): the
#   INT8 convrot kernels that run Comfy-Org's *_int8_convrot.safetensors
#   natively only exist there. Their CUDA path needs compute capability 7.5+.
# - CUDA_ARCHITECTURES is explicit because there's no GPU at `docker build`
#   time to auto-detect: Turing, Ampere (A100 / RTX 30), Ada (RTX 40, e.g.
#   the 4090), Hopper, Blackwell (RTX 50). Trim it to your card for a faster
#   build, e.g. --build-arg CUDA_ARCHITECTURES=89.
# - GGML_NATIVE=OFF: don't tune the CPU code for the build machine's CPU.
# - SD_SERVER_BUILD_FRONTEND=OFF: sd-server's web UI needs Node/pnpm to build,
#   and it isn't needed here (use `comfyui` mode for a UI).
ARG CUDA_ARCHITECTURES="75;80;86;89;90;120"
RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DSD_CUDA=ON \
        -DGGML_NATIVE=OFF \
        "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" \
        -DSD_SERVER_BUILD_FRONTEND=OFF \
    && cmake --build build --config Release --target sd-server sd-cli -j"$(nproc)"

# ---------------------------------------------------------------------------

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04

# python3-venv: Ubuntu 24.04's system Python is "externally managed", so all
# Python packages go in a venv. libgomp1: OpenMP runtime for sd-server/sd-cli.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libgomp1 \
    python3 \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN python3 -m venv /opt/venv && pip install flask

# --- ComfyUI (optional: --build-arg WITH_COMFYUI=0 skips it) ---------------
# Qwen-Image-2.1 (and its INT8 convrot / KV-cache / prefetch support) landed in
# ComfyUI v0.37.0 with fixes after it, so pin a commit past those.
#
# PyTorch comes from PyPI, whose Linux wheels bundle CUDA 13 — ComfyUI
# requires a cu130+ PyTorch on RTX 20-series and newer. That means `comfyui`
# mode needs a host driver >= 580 (`api` mode doesn't use PyTorch at all). On
# an older driver, build with
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
#   --build-arg TORCH_VERSION=2.11.0 --build-arg TORCHVISION_VERSION=0.26.0
# at the cost of some of ComfyUI's newer optimizations.
ARG WITH_COMFYUI=1
ARG COMFYUI_REF=912fca4f39b875a0360f2c5170568176ea813ded
ARG TORCH_VERSION=2.14.0
ARG TORCHVISION_VERSION=0.29.0
ARG TORCH_INDEX_URL=https://pypi.org/simple
RUN if [ "${WITH_COMFYUI}" = "1" ]; then \
        set -e; \
        mkdir -p /opt/ComfyUI; \
        curl -fsSL "https://github.com/comfyanonymous/ComfyUI/archive/${COMFYUI_REF}.tar.gz" \
            | tar -xz --strip-components=1 -C /opt/ComfyUI; \
        pip install --index-url "${TORCH_INDEX_URL}" \
            "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"; \
        pip install -r /opt/ComfyUI/requirements.txt; \
    fi

# ---------------------------------------------------------------------------

WORKDIR /app

COPY --from=builder /build/build/bin/sd-server /build/build/bin/sd-cli /app/
COPY server.py entrypoint.sh extra_model_paths.yaml /app/
RUN chmod +x /app/entrypoint.sh \
    && mkdir -p /data && chmod 1777 /data

# API mode: the three model files. Defaults are Comfy-Org's INT8 convrot set
# (see the top of this file). stable-diffusion.cpp also takes the bf16
# safetensors, or GGUFs — e.g. unsloth's qwen-image-2.1-Q4_K_M.gguf as
# MODEL_PATH plus a Qwen3-VL-8B-Instruct GGUF as LLM_PATH, for smaller cards.
# (ComfyUI mode can't load GGUFs without a custom node, so keep the INT8 files
# around if you use both modes.)
#
# OFFLOAD_TO_CPU=1 keeps weights in system RAM and moves them onto the GPU as
# they're needed; set it to 0 to keep everything resident on a 24 GB+ card.
# SD_EXTRA_ARGS is appended verbatim to the sd-server command line (e.g.
# "--vae-tiling" for very large images).
#
# COMFYUI_DATA is where comfyui mode keeps everything it writes (outputs,
# inputs, workflows, settings); mount a host directory there to keep them.
# COMFYUI_ARGS is appended to ComfyUI's command line (e.g. "--lowvram").
#
# HOME points somewhere writable because run.sh starts the container as the
# invoking host user (--user), who has no home directory in here.
ENV MODEL_PATH=/models/diffusion_models/qwen_image_2.1_int8_convrot.safetensors \
    VAE_PATH=/models/vae/qwen_image_2.1_vae_bf16.safetensors \
    LLM_PATH=/models/text_encoders/qwen3vl_8b_int8_convrot.safetensors \
    SD_SERVER=/app/sd-server \
    OFFLOAD_TO_CPU=1 \
    DIFFUSION_FA=1 \
    SD_EXTRA_ARGS="" \
    DEFAULT_STEPS=40 \
    DEFAULT_CFG=1.0 \
    GENERATION_TIMEOUT=600 \
    COMFYUI_DATA=/data \
    COMFYUI_ARGS="" \
    HOME=/tmp

EXPOSE 5002 8188

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["api"]

# docker build -t qwen-image-api qwen-image-2.1/
