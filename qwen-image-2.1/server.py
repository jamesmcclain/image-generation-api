"""REST API for Qwen-Image-2.1 text-to-image, backed by stable-diffusion.cpp.

Same HTTP contract as the Krea 2 image (../krea2/server.py), so existing
clients (including the krea2-api skill's scripts) work unchanged against
either one.

Unlike the Krea 2 image, which runs `sd-cli` once per request (reloading every
model file each time), this one starts stable-diffusion.cpp's `sd-server` once,
at container startup, and forwards each request to its A1111-compatible
`/sdapi/v1/txt2img` endpoint on 127.0.0.1. The ~17 GB of weights therefore stay
loaded between requests. sd-server serializes generations internally.

Qwen-Image-2.1 needs three files (loaded via separate sd-server flags):
  --diffusion-model  the Qwen-Image-2.1 DiT: Comfy-Org's int8_convrot /
                     bf16 safetensors, or an unsloth GGUF
  --vae              qwen_image_2.1_vae_bf16.safetensors (NOT the Qwen-Image
                     1.x or Wan VAE: they aren't interchangeable)
  --llm              Qwen3-VL-8B text encoder: a GGUF (default: Qwen's
                     Qwen3VL-8B-Instruct-Q4_K_M.gguf) or Comfy-Org's bf16
                     safetensors. NOT Comfy-Org's int8_convrot one, which
                     sd.cpp mis-loads (see the Dockerfile), and NOT the 4B
                     model Krea 2 uses (wrong feature width for this model).

POST /generate
  JSON body:
    prompt          (str, required)
    steps           (int, default 40)     -- DEFAULT_STEPS env var. The
                                            official pipeline uses 40-50 euler
                                            steps; ComfyUI's template uses 25.
    cfg             (float, default 1.0)  -- DEFAULT_CFG env var. 1.0 means
                                            "guidance off", which is the
                                            official way to sample this model
                                            (and half the work per step).
                                            Values > 1 enable real CFG against
                                            `negative_prompt`; sd.cpp's and
                                            unsloth's examples use 6.0.
                                            Any value <= 1 (including 0.0) is
                                            treated as 1.0: stable-diffusion.cpp
                                            would otherwise read exactly 0.0 as
                                            "unconditioned" and silently ignore
                                            the prompt.
    width           (int, default 1024)   -- rounded DOWN to a multiple of 32;
    height          (int, default 1024)      the model is native up to 2048x2048
    seed            (int, optional)       -- omitted -> random seed each call
    negative_prompt (str, optional)       -- extension; only used when cfg > 1

  Returns: image/png bytes on success (an `X-Seed` header carries the seed
  that was used), JSON error on failure. The PNG is RGBA when the model
  produces transparency, which it does when the prompt asks for it, e.g.
  "This is an RGBA image with transparency. <subject>. The image has alpha
  channel and the background is transparent."

GET /health
  Returns: {"status": "ok"} once sd-server has loaded the model and is
  accepting requests. 503 with {"status": "error", "missing": {...}} if a
  model path points nowhere, {"status": "loading"} while sd-server is still
  loading weights, or {"status": "error", "error": "...", "stderr": "..."} if
  sd-server exited (the tail of its log explains why).
"""

import base64
import collections
import json
import os
import secrets
import shlex
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from flask import Flask, Response, jsonify, request

app = Flask(__name__)

MODEL_PATH = os.environ["MODEL_PATH"]
VAE_PATH = os.environ["VAE_PATH"]
LLM_PATH = os.environ["LLM_PATH"]
SD_SERVER = os.environ.get("SD_SERVER", "/app/sd-server")
SD_SERVER_PORT = int(os.environ.get("SD_SERVER_PORT", "5010"))
SAMPLING_METHOD = os.environ.get("SAMPLING_METHOD", "euler")
OFFLOAD_TO_CPU = os.environ.get("OFFLOAD_TO_CPU", "1") == "1"
DIFFUSION_FA = os.environ.get("DIFFUSION_FA", "1") == "1"
SD_EXTRA_ARGS = os.environ.get("SD_EXTRA_ARGS", "")
DEFAULT_STEPS = int(os.environ.get("DEFAULT_STEPS", "40"))
DEFAULT_CFG = float(os.environ.get("DEFAULT_CFG", "1.0"))
GENERATION_TIMEOUT = float(os.environ.get("GENERATION_TIMEOUT", "600"))

BACKEND_URL = f"http://127.0.0.1:{SD_SERVER_PORT}"
# stable-diffusion.cpp requires dimensions divisible by 32 for this model.
DIM_MULTIPLE = 32

_proc = None
_proc_ready = False  # has the current sd-server process ever answered?
_proc_lock = threading.Lock()
_log = collections.deque(maxlen=200)


def _missing_paths():
    paths = {"MODEL_PATH": MODEL_PATH, "VAE_PATH": VAE_PATH, "LLM_PATH": LLM_PATH}
    return {name: path for name, path in paths.items() if not os.path.isfile(path)}


def _log_tail(chars=4000):
    return "".join(_log)[-chars:]


def _pump_output(proc):
    # Keep a tail of sd-server's output for error responses, and mirror it to
    # our own stdout so it shows up in `docker logs`.
    for line in proc.stdout:
        _log.append(line)
        sys.stdout.write(f"[sd-server] {line}")
        sys.stdout.flush()
    code = proc.wait()
    _log.append(f"sd-server exited with code {code}\n")
    sys.stdout.write(f"[sd-server] exited with code {code}\n")
    sys.stdout.flush()


def _start_backend():
    """Launch a new sd-server process. Caller must hold _proc_lock."""
    global _proc, _proc_ready
    cmd = [
        SD_SERVER,
        "--diffusion-model", MODEL_PATH,
        "--vae", VAE_PATH,
        "--llm", LLM_PATH,
        "--listen-ip", "127.0.0.1",
        "--listen-port", str(SD_SERVER_PORT),
        "--sampling-method", SAMPLING_METHOD,
    ]
    if DIFFUSION_FA:
        cmd.append("--diffusion-fa")
    if OFFLOAD_TO_CPU:
        cmd.append("--offload-to-cpu")
    cmd += shlex.split(SD_EXTRA_ARGS)

    _log.clear()
    _log.append("$ " + shlex.join(cmd) + "\n")
    _proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )
    _proc_ready = False
    threading.Thread(target=_pump_output, args=(_proc,), daemon=True).start()


def _backend_listening():
    # sd-server loads every model before it starts listening, so an open port
    # means it's ready to generate.
    try:
        with socket.create_connection(("127.0.0.1", SD_SERVER_PORT), timeout=1):
            return True
    except OSError:
        return False


def _check_backend():
    """Return (payload_dict, 503) if the backend can't serve yet, else None.

    Also (re)starts sd-server when appropriate: on first use once the model
    files exist, and again if a previously healthy sd-server died (e.g. it
    crashed mid-generation). One that dies before ever becoming ready is left
    down, since restarting it would just fail the same way; the error and log
    tail are reported instead.
    """
    global _proc_ready
    missing = _missing_paths()
    if missing:
        return dict(status="error", error="model file(s) not found", missing=missing), 503

    with _proc_lock:
        try:
            if _proc is None:
                _start_backend()
            elif _proc.poll() is not None and _proc_ready:
                app.logger.warning("sd-server died after being healthy; restarting it")
                _start_backend()
        except OSError as e:  # e.g. the sd-server binary is missing
            return dict(status="error", error=f"could not start sd-server: {e}"), 503
        if _proc.poll() is not None:
            return dict(
                status="error",
                error=f"sd-server exited with code {_proc.returncode} while loading",
                stderr=_log_tail(),
            ), 503

        if _backend_listening():
            _proc_ready = True
            return None
    return dict(status="loading", error="model is still loading"), 503


@app.get("/health")
def health():
    not_ready = _check_backend()
    if not_ready is not None:
        return jsonify(not_ready[0]), not_ready[1]
    return jsonify(status="ok")


@app.post("/generate")
def generate():
    body = request.get_json(silent=True) or {}

    prompt = body.get("prompt")
    if not prompt:
        return jsonify(error="'prompt' is required"), 400

    try:
        steps = int(body.get("steps", DEFAULT_STEPS))
        cfg = float(body.get("cfg", DEFAULT_CFG))
        width = int(body.get("width", 1024))
        height = int(body.get("height", 1024))
        seed = body.get("seed")
        seed = secrets.randbelow(2**31) if seed is None else int(seed)
    except (TypeError, ValueError) as e:
        return jsonify(error=f"invalid parameter: {e}"), 400
    negative_prompt = body.get("negative_prompt") or ""

    if steps < 1:
        return jsonify(error="'steps' must be >= 1"), 400
    width -= width % DIM_MULTIPLE
    height -= height % DIM_MULTIPLE
    if width < DIM_MULTIPLE or height < DIM_MULTIPLE:
        return jsonify(error=f"'width' and 'height' must be >= {DIM_MULTIPLE}"), 400
    # See the module docstring: never let 0.0 reach sd.cpp (unconditioned
    # mode); everything <= 1 means "no guidance" here.
    cfg = max(cfg, 1.0)

    not_ready = _check_backend()
    if not_ready is not None:
        return jsonify(not_ready[0]), not_ready[1]

    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg_scale": cfg,
        "seed": seed,
        "batch_size": 1,
    }
    req = urllib.request.Request(
        f"{BACKEND_URL}/sdapi/v1/txt2img",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=GENERATION_TIMEOUT) as r:
            result = json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        return jsonify(error="generation failed", stderr=(detail + "\n" + _log_tail())[-4000:]), 500
    except (TimeoutError, socket.timeout):
        # sd-server can't be interrupted mid-generation; it finishes this job
        # in the background and later requests queue behind it.
        return jsonify(error="generation timed out"), 504
    except (urllib.error.URLError, ConnectionError) as e:
        return jsonify(error=f"generation failed: {e}", stderr=_log_tail()), 500

    try:
        png = base64.b64decode(result["images"][0])
    except (KeyError, IndexError, TypeError, ValueError):
        return jsonify(error="generation failed", stderr=json.dumps(result)[:2000] + "\n" + _log_tail()), 500
    if not png.startswith(b"\x89PNG"):
        return jsonify(error="generation failed: backend did not return a PNG", stderr=_log_tail()), 500

    app.logger.info("generated %dx%d, %d steps, cfg %.2f in %.1fs", width, height, steps, cfg,
                    time.monotonic() - started)
    return Response(png, mimetype="image/png", headers={"X-Seed": str(seed)})


if __name__ == "__main__":
    # Start loading the weights right away so the first /generate doesn't
    # have to wait for it; /health reports "loading" until sd-server is up.
    _check_backend()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5002")), threaded=True)
