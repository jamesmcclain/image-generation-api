"""REST API wrapping stable-diffusion.cpp's `sd-cli` CLI for Krea 2 Turbo GGUF.

Krea2 is not a self-contained single-file checkpoint: it requires the
diffusion transformer GGUF plus two companion models, loaded via separate
flags:
  --diffusion-model  the Krea 2 Turbo GGUF itself
  --vae              the Wan2.1 VAE (not Krea2-specific; shared with Wan2.1)
  --llm              Qwen3-VL 4B, used as Krea2's text encoder

POST /generate
  JSON body:
    prompt          (str, required)
    steps           (int, default 8)     -- Turbo is distilled for ~8 steps
    cfg             (float, default 0.0) -- Turbo runs with CFG=0/1, not the
                                            7-12 range typical of SD1.5/SDXL
    width           (int, default 1024)
    height          (int, default 1024)
    seed            (int, optional)      -- omitted -> random seed each call

  Returns: image/png bytes on success, JSON error on failure.

GET /health
  Returns: {"status": "ok"} once all three model paths are confirmed to exist.
"""

import os
import subprocess
import uuid

from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

MODEL_PATH = os.environ["MODEL_PATH"]
VAE_PATH = os.environ["VAE_PATH"]
LLM_PATH = os.environ["LLM_PATH"]
SD_BINARY = os.environ["SD_BINARY"]
OUTPUT_DIR = os.environ["OUTPUT_DIR"]


def _missing_paths():
    paths = {"MODEL_PATH": MODEL_PATH, "VAE_PATH": VAE_PATH, "LLM_PATH": LLM_PATH}
    return {name: path for name, path in paths.items() if not os.path.isfile(path)}


@app.get("/health")
def health():
    missing = _missing_paths()
    if missing:
        return jsonify(status="error", missing=missing), 503
    return jsonify(status="ok")


@app.post("/generate")
def generate():
    body = request.get_json(silent=True) or {}

    prompt = body.get("prompt")
    if not prompt:
        return jsonify(error="'prompt' is required"), 400

    steps = int(body.get("steps", 8))
    cfg = float(body.get("cfg", 0.0))
    width = int(body.get("width", 1024))
    height = int(body.get("height", 1024))
    seed = body.get("seed")

    missing = _missing_paths()
    if missing:
        return jsonify(error="model file(s) not found", missing=missing), 503

    # Unique filename per request so concurrent requests never collide on
    # the same output path.
    out_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4().hex}.png")

    cmd = [
        SD_BINARY,
        "--diffusion-model", MODEL_PATH,
        "--vae", VAE_PATH,
        "--llm", LLM_PATH,
        "-p", prompt,
        "--steps", str(steps),
        "--cfg-scale", str(cfg),
        "-W", str(width),
        "-H", str(height),
        "-o", out_path,
        "--diffusion-fa",
    ]
    if seed is not None:
        cmd += ["-s", str(int(seed))]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        return jsonify(error="generation timed out"), 504

    if result.returncode != 0 or not os.path.isfile(out_path):
        return jsonify(
            error="generation failed",
            stderr=result.stderr[-4000:],
        ), 500

    try:
        return send_file(out_path, mimetype="image/png")
    finally:
        # The file's been read into the response by this point; no need
        # to keep it around on disk afterward.
        os.remove(out_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
