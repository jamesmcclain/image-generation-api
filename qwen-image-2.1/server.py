"""REST API for Qwen-Image-2.1 text-to-image, served via Hugging Face diffusers.

Same HTTP contract as the Krea 2 image (../server.py), so existing clients
(including the krea2-api skill's scripts) work unchanged against either one.

Qwen-Image-2.1 is a new architecture (single-stream 7B DiT + Qwen3-VL 8B text
encoder + its own VAE) that stable-diffusion.cpp does not support, so unlike
the Krea 2 image this one runs the official `QwenImage21Pipeline` in-process.
The pipeline is loaded once, at startup, and then reused for every request;
requests are serialized on the GPU with a lock.

POST /generate
  JSON body:
    prompt          (str, required)
    steps           (int, default 40)     -- the model card's recommended count
    cfg             (float, default 1.0)  -- maps to diffusers' true_cfg_scale.
                                            Qwen-Image-2.1 is meant to be
                                            sampled WITHOUT guidance, so 1.0
                                            (= guidance off) is the right
                                            default. Values > 1 enable true
                                            CFG against `negative_prompt`
                                            (and roughly double the time per
                                            step). Unlike the sd.cpp image,
                                            cfg=0.0 is NOT a trap here: any
                                            value <= 1 just means "no guidance"
                                            and the prompt is still followed.
    width           (int, default 1024)   -- rounded DOWN to a multiple of 32
    height          (int, default 1024)   -- rounded DOWN to a multiple of 32
    seed            (int, optional)       -- omitted -> random seed each call
    negative_prompt (str, optional)       -- extension; only used when cfg > 1
                                            (defaults to " " in that case)

  Returns: image/png bytes on success (an `X-Seed` header carries the seed
  that was used), JSON error on failure. The PNG may be RGBA: the model
  natively produces transparency when the prompt asks for it.

GET /health
  Returns: {"status": "ok"} once the pipeline is loaded and ready.
  503 with {"status": "error", "missing": {...}} if MODEL_PATH isn't a
  diffusers model directory, {"status": "loading"} while weights are still
  being loaded, or {"status": "error", "error": "..."} if loading failed.

Environment:
  MODEL_PATH          local diffusers snapshot of Qwen/Qwen-Image-2.1
                      (the directory containing model_index.json)
  OFFLOAD             model (default) | sequential | none -- see Dockerfile
  GENERATION_TIMEOUT  seconds, default 600 (matches the Krea 2 image)
"""

import io
import os
import secrets
import threading
import time
import traceback

import torch
from flask import Flask, jsonify, request, send_file

app = Flask(__name__)

MODEL_PATH = os.environ["MODEL_PATH"]
OFFLOAD = os.environ.get("OFFLOAD", "model").strip().lower()
GENERATION_TIMEOUT = float(os.environ.get("GENERATION_TIMEOUT", "600"))
DEVICE = os.environ.get("DEVICE", "cuda")

# The pipeline floors width/height to this multiple itself (vae_scale_factor
# 16 x 2); doing it here too means the value we log/validate is the value used.
DIM_MULTIPLE = 32

_pipe = None
_load_error = None
_loading = False
_load_lock = threading.Lock()
# One generation at a time: the pipeline object isn't safe to call
# concurrently, and two jobs would just fight over VRAM anyway.
_gen_lock = threading.Lock()


def _missing_paths():
    index = os.path.join(MODEL_PATH, "model_index.json")
    if os.path.isfile(index):
        return {}
    return {"MODEL_PATH": index}


def _load_pipeline():
    """Load the pipeline once. Safe to call repeatedly / from several threads."""
    global _pipe, _load_error, _loading
    with _load_lock:
        if _pipe is not None or _missing_paths():
            return
        _loading = True
        _load_error = None
        try:
            # Imported here, not at module top, so /health can answer
            # immediately while the (slow) diffusers import + load runs.
            from diffusers import QwenImage21Pipeline

            pipe = QwenImage21Pipeline.from_pretrained(
                MODEL_PATH,
                dtype=torch.bfloat16,
                local_files_only=True,
            )
            if OFFLOAD == "none":
                pipe.to(DEVICE)
            elif OFFLOAD == "sequential":
                pipe.enable_sequential_cpu_offload(device=DEVICE)
            elif OFFLOAD == "model":
                pipe.enable_model_cpu_offload(device=DEVICE)
            else:
                raise ValueError(
                    f"OFFLOAD must be one of model|sequential|none, got {OFFLOAD!r}"
                )
            pipe.set_progress_bar_config(disable=True)
            _pipe = pipe
            app.logger.warning("Qwen-Image-2.1 loaded from %s (offload=%s)", MODEL_PATH, OFFLOAD)
        except Exception as e:  # surfaced via /health and /generate
            _load_error = f"{type(e).__name__}: {e}"
            app.logger.error("failed to load pipeline:\n%s", traceback.format_exc())
        finally:
            _loading = False


def _not_ready_response():
    """Return a (json, 503) tuple if the pipeline isn't usable yet, else None."""
    missing = _missing_paths()
    if missing:
        return jsonify(status="error", error="model file(s) not found", missing=missing), 503
    if _pipe is not None:
        return None
    if _loading or _load_lock.locked():
        return jsonify(status="loading", error="model is still loading"), 503
    if _load_error:
        return jsonify(status="error", error=_load_error), 503
    # Files exist but nothing has tried loading them yet (e.g. they were
    # mounted after startup): kick off a load in the background.
    threading.Thread(target=_load_pipeline, daemon=True).start()
    return jsonify(status="loading", error="model is still loading"), 503


@app.get("/health")
def health():
    not_ready = _not_ready_response()
    if not_ready is not None:
        return not_ready
    return jsonify(status="ok")


@app.post("/generate")
def generate():
    started = time.monotonic()
    body = request.get_json(silent=True) or {}

    prompt = body.get("prompt")
    if not prompt:
        return jsonify(error="'prompt' is required"), 400

    try:
        steps = int(body.get("steps", 40))
        cfg = float(body.get("cfg", 1.0))
        width = int(body.get("width", 1024))
        height = int(body.get("height", 1024))
        seed = body.get("seed")
        seed = secrets.randbits(63) if seed is None else int(seed)
    except (TypeError, ValueError) as e:
        return jsonify(error=f"invalid parameter: {e}"), 400
    negative_prompt = body.get("negative_prompt")

    if steps < 1:
        return jsonify(error="'steps' must be >= 1"), 400
    width -= width % DIM_MULTIPLE
    height -= height % DIM_MULTIPLE
    if width < DIM_MULTIPLE or height < DIM_MULTIPLE:
        return jsonify(error=f"'width' and 'height' must be >= {DIM_MULTIPLE}"), 400

    not_ready = _not_ready_response()
    if not_ready is not None:
        return not_ready

    # Guidance only kicks in with cfg > 1 AND a negative prompt; " " is the
    # conventional "empty" negative prompt for the Qwen-Image family.
    if cfg > 1.0 and not negative_prompt:
        negative_prompt = " "
    elif cfg <= 1.0:
        negative_prompt = None  # avoid a pointless second text-encoder pass

    deadline = started + GENERATION_TIMEOUT
    if not _gen_lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
        return jsonify(error="generation timed out (waiting for the GPU)"), 504

    timed_out = False

    def _stop_at_deadline(pipe, step, timestep, callback_kwargs):
        # Cooperative timeout: the pipeline checks `interrupt` before every
        # step, so this ends the denoising loop cleanly on the GPU.
        nonlocal timed_out
        if time.monotonic() > deadline:
            timed_out = True
            pipe._interrupt = True
        return callback_kwargs

    try:
        # CPU generator: identical noise for a given seed regardless of the
        # OFFLOAD mode / which device the pipeline happens to execute on.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        image = _pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            true_cfg_scale=cfg,
            width=width,
            height=height,
            num_inference_steps=steps,
            generator=generator,
            callback_on_step_end=_stop_at_deadline,
        ).images[0]
    except Exception as e:
        tb = traceback.format_exc()
        app.logger.error("generation failed:\n%s", tb)
        # Keep the same error shape as the sd.cpp image: the tail of the
        # underlying error output lives in `stderr`.
        return jsonify(error="generation failed", stderr=tb[-4000:]), 500
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _gen_lock.release()

    if timed_out:
        return jsonify(error="generation timed out"), 504

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    response = send_file(buf, mimetype="image/png")
    response.headers["X-Seed"] = str(seed)
    return response


if __name__ == "__main__":
    # Start loading the multi-GB weights right away so the first /generate
    # doesn't pay for it; /health reports "loading" until this finishes.
    threading.Thread(target=_load_pipeline, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5002")), threaded=True)
