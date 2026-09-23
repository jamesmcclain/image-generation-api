# Dockerfile — Krea 2 Turbo (GGUF) text-to-image REST API via stable-diffusion.cpp
#
# Build (from the repo root):
#   docker build -t krea2-api krea2/
#
# Run (model weights are NOT baked in — they're large, multi-GB quantized
# GGUF/safetensors files — so mount them from the host instead). Krea2
# needs three separate files: the diffusion GGUF, a Wan2.1 VAE, and a
# Qwen3-VL 4B GGUF text encoder:
#   docker run -p 5002:5002 \
#     -v /path/to/gguf-models:/models:ro \
#     -e MODEL_PATH=/models/krea2_turbo-Q4_K_M.gguf \
#     -e VAE_PATH=/models/wan_2.1_vae.safetensors \
#     -e LLM_PATH=/models/Qwen3-VL-4B-Instruct-Q4_K_M.gguf \
#     krea2-api

# https://huggingface.co/vantagewithai/Krea-2-Turbo-GGUF
# https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF
# https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

# Build toolchain for stable-diffusion.cpp itself. cmake/g++ compile the
# GGML backend; git pulls the source (and its ggml submodule). The CUDA
# devel image already provides nvcc and the CUDA libraries SD_CUDA needs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# --recursive: stable-diffusion.cpp vendors ggml as a submodule; without
# --recursive the build fails with missing ggml headers.
RUN git clone --recursive https://github.com/leejet/stable-diffusion.cpp.git .

# -DSD_CUDA=ON builds the CUDA backend so generation actually runs on the
# GPU passed in via `docker run --gpus all`, instead of falling back to
# CPU (workably slow for a 12B DiT, but much slower than GPU).
RUN mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON \
    && cmake --build . --config Release -j"$(nproc)"

# ---------------------------------------------------------------------------

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# CUDA runtime libraries (no compiler/devel headers needed here) plus
# python3/pip for the Flask API, and libgomp1 which the compiled sd binary
# needs at runtime for its OpenMP-based threading.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir flask

WORKDIR /app

# Pull just the compiled binary out of the builder stage — none of the
# build toolchain or source tree needs to ship in the final image.
# Note: upstream's CLI binary is named `sd-cli` (renamed from the earlier
# `sd`, which collided with the unrelated, commonly preinstalled `sd`
# find-and-replace tool — see leejet/stable-diffusion.cpp#750/#1037).
COPY --from=builder /build/build/bin/sd-cli /app/sd-cli

COPY server.py /app/server.py

# Where generated images are written before being read back and returned
# in the HTTP response. Not baked read-only, since every request writes
# a new file here.
RUN mkdir -p /app/output && chmod 777 /app/output

# Krea2 needs three separate model files under /models (see run.sh):
#   - the Krea 2 Turbo diffusion GGUF (MODEL_PATH)
#   - the Wan2.1 VAE, e.g. wan_2.1_vae.safetensors (VAE_PATH)
#   - a Qwen3-VL 4B GGUF used as the text encoder (LLM_PATH)
# None are baked in — there's no model in the image (see the run command
# above). server.py fails fast with a clear error if any of these still
# point nowhere at request time.
ENV MODEL_PATH=/models/krea2_turbo-Q4_K_M.gguf \
    VAE_PATH=/models/wan_2.1_vae.safetensors \
    LLM_PATH=/models/Qwen3-VL-4B-Instruct-Q4_K_M.gguf \
    SD_BINARY=/app/sd-cli \
    OUTPUT_DIR=/app/output

EXPOSE 5002

CMD ["python3", "/app/server.py"]

# docker build -t krea2-api krea2/
