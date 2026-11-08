"""REST API for Qwen-Image-2.1 text-to-image and image editing, backed by
stable-diffusion.cpp.

Same HTTP contract as the Krea 2 image (../krea2/server.py), so existing
clients (including the krea2-api skill's scripts) work unchanged against
either one. The image-input fields (`images`, `strength`) are extensions that
only this server implements; the Krea 2 server rejects them with a 400.

Unlike the Krea 2 image, which runs `sd-cli` once per request (reloading every
model file each time), this one starts stable-diffusion.cpp's `sd-server` once,
at container startup, and forwards each request to its A1111-compatible
`/sdapi/v1/txt2img` (or `/sdapi/v1/img2img`) endpoint on 127.0.0.1. The ~17 GB
of weights therefore stay loaded between requests. sd-server serializes
generations internally.

Qwen-Image-2.1 needs three files, plus a fourth for reference editing (loaded
via separate sd-server flags):
  --diffusion-model  the Qwen-Image-2.1 DiT: Comfy-Org's int8_convrot /
                     bf16 safetensors, or an unsloth GGUF
  --vae              qwen_image_2.1_vae_bf16.safetensors (NOT the Qwen-Image
                     1.x or Wan VAE: they aren't interchangeable)
  --llm              Qwen3-VL-8B text encoder: a GGUF (default: unsloth's
                     Qwen3-VL-8B-Instruct-Q4_K_M.gguf) or Comfy-Org's bf16
                     safetensors. NOT Comfy-Org's int8_convrot one, which
                     sd.cpp mis-loads (see the Dockerfile), and NOT the 4B
                     model Krea 2 uses (wrong feature width for this model).
  --llm_vision       (optional, VISION_PATH) Qwen3-VL-8B's vision weights, e.g.
                     unsloth's mmproj-BF16.gguf. A GGUF text encoder has no
                     vision weights of its own, and reference editing (see
                     below) needs them. Comfy-Org's bf16 text encoder already
                     contains them, so VISION_PATH can be empty with that one.
                     Without vision weights, text-to-image and img2img still
                     work; reference editing is reported unavailable.

POST /generate
  JSON body:
    prompt          (str, required)
    steps           (int, default 40)     -- DEFAULT_STEPS env var. The
                                            official pipeline uses 40-50 euler
                                            steps; ComfyUI's template uses 25.
    cfg             (float, default 1.0)  -- DEFAULT_CFG env var. 1.0 means
                                            "guidance off", which is the
                                            official way to sample this model
                                            (and half the work per step), for
                                            edits as well as text-to-image.
                                            Values > 1 enable real CFG against
                                            `negative_prompt`; sd.cpp's and
                                            unsloth's examples use 6.0.
                                            Any value <= 1 (including 0.0) is
                                            treated as 1.0: stable-diffusion.cpp
                                            would otherwise read exactly 0.0 as
                                            "unconditioned" and silently ignore
                                            the prompt.
    width           (int, optional)       -- explicit values are rounded DOWN
    height          (int, optional)          to a multiple of 32; the model is
                                            native up to 2048x2048. Omitted (or
                                            null) means automatic; see "Output
                                            size" below.
    seed            (int, optional)       -- omitted -> random seed each call
    negative_prompt (str, optional)       -- extension; only used when cfg > 1
    images          (list, optional)      -- extension; 1 or 2 input images
                                            (MAX_INPUT_IMAGES), each a base64
                                            string or a data URL
                                            ("data:image/png;base64,...").
                                            PNG, JPEG, WebP and anything else
                                            Pillow decodes; EXIF rotation is
                                            applied and alpha is kept.
    strength        (float, optional)     -- extension; 0 < strength <= 1.
                                            Selects img2img (see below) and
                                            requires exactly one image.

  What the request does depends on `images` and `strength`:
    no images             text-to-image (unchanged)
    images, no strength   reference editing, the model's native edit mode:
                          the images go to the model as references, labeled
                          <image1>, <image2> in the order given, and the
                          prompt can refer to them by those names. <image1>
                          is the image being edited; <image2> is extra
                          reference material, e.g. "put the shirt from
                          <image2> on the person in <image1>". Needs vision
                          weights (GET /health reports "edit").
    one image + strength  img2img: the image is noised by `strength` and
                          regenerated towards the prompt. Low values stay
                          close to the input; 1.0 keeps little but its size.
                          Doesn't need vision weights.

  Output size: explicit width/height are always honored. Without images, a
  missing width or height defaults to 1024. With images, the output follows
  the first image's aspect ratio: with neither given, it's scaled to an area
  of AUTO_SIZE_PIXELS (default 1024*1024, the model's standard budget); with
  just one given, the other is derived from the aspect ratio. Automatic
  dimensions are rounded to the nearest multiple of 32. An explicit size far
  from the first image's aspect ratio works, but the edit can shift.

  Returns: image/png bytes on success (an `X-Seed` header carries the seed
  that was used), JSON error on failure. The PNG is RGBA when the model
  produces transparency, which it does when the prompt asks for it, e.g.
  "This is an RGBA image with transparency. <subject>. The image has alpha
  channel and the background is transparent."

  Errors: 400 for invalid input, including images that don't decode, too
  many images, `strength` without exactly one image, and reference editing
  while the server has no vision weights; 413 for a body over
  MAX_REQUEST_MB; 503 while the backend isn't ready; 500/504 for generation
  failures and timeouts.

GET /health
  Returns: {"status": "ok", "img2img": true, "edit": <bool>,
  "max_images": <int>} once sd-server has loaded the model and is accepting
  requests. "edit" says whether reference editing is available (sd-server
  loaded vision weights); when it's false, "edit_error" says why and
  "max_images" is 1 (img2img only). 503 with {"status": "error", "missing":
  {...}} if a model path points nowhere, {"status": "loading"} while
  sd-server is still loading weights, or {"status": "error", "error": "...",
  "stderr": "..."} if sd-server exited (the tail of its log explains why).
"""

import base64
import binascii
import collections
import io
import json
import math
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
from PIL import Image, ImageOps, UnidentifiedImageError

app = Flask(__name__)

MODEL_PATH = os.environ["MODEL_PATH"]
VAE_PATH = os.environ["VAE_PATH"]
LLM_PATH = os.environ["LLM_PATH"]
VISION_PATH = os.environ.get("VISION_PATH", "")
SD_SERVER = os.environ.get("SD_SERVER", "/app/sd-server")
SD_SERVER_PORT = int(os.environ.get("SD_SERVER_PORT", "5010"))
SAMPLING_METHOD = os.environ.get("SAMPLING_METHOD", "euler")
OFFLOAD_TO_CPU = os.environ.get("OFFLOAD_TO_CPU", "1") == "1"
DIFFUSION_FA = os.environ.get("DIFFUSION_FA", "1") == "1"
SD_EXTRA_ARGS = os.environ.get("SD_EXTRA_ARGS", "")
DEFAULT_STEPS = int(os.environ.get("DEFAULT_STEPS", "40"))
DEFAULT_CFG = float(os.environ.get("DEFAULT_CFG", "1.0"))
GENERATION_TIMEOUT = float(os.environ.get("GENERATION_TIMEOUT", "600"))
# Input-image limits. MAX_INPUT_PIXELS is checked from the image header,
# before decoding, so a huge (or malicious) file is refused cheaply.
MAX_INPUT_IMAGES = int(os.environ.get("MAX_INPUT_IMAGES", "2"))
MAX_INPUT_PIXELS = int(os.environ.get("MAX_INPUT_PIXELS", str(50_000_000)))
MAX_REQUEST_MB = float(os.environ.get("MAX_REQUEST_MB", "64"))
# Output area when the size follows the first input image.
AUTO_SIZE_PIXELS = int(os.environ.get("AUTO_SIZE_PIXELS", str(1024 * 1024)))

BACKEND_URL = f"http://127.0.0.1:{SD_SERVER_PORT}"
# stable-diffusion.cpp requires dimensions divisible by 32 for this model.
DIM_MULTIPLE = 32
# Inputs larger than this (or than the output, if that's larger) are scaled
# down before they're handed to sd-server. sd-server scales references to the
# output's area and img2img inputs to the output size anyway, so this only
# saves encoding and transfer time. 2048x2048 is the model's native maximum.
TRANSFER_MAX_PIXELS = 2048 * 2048

app.config["MAX_CONTENT_LENGTH"] = int(MAX_REQUEST_MB * 1024 * 1024)
# We enforce MAX_INPUT_PIXELS ourselves, with a clear 400, instead of relying
# on Pillow's decompression-bomb warning/error.
Image.MAX_IMAGE_PIXELS = None

_proc = None
_proc_ready = False  # has the current sd-server process ever answered?
_proc_lock = threading.Lock()
_log = collections.deque(maxlen=200)
# sd-server logs "... vision disabled" while loading when it has no usable
# vision weights; the line is kept here (per process) to explain why
# reference editing is unavailable.
_vision_disabled_line = None


class BadInput(ValueError):
    """A client error, reported as a 400 with this message."""


def _missing_paths():
    paths = {"MODEL_PATH": MODEL_PATH, "VAE_PATH": VAE_PATH, "LLM_PATH": LLM_PATH}
    return {name: path for name, path in paths.items() if not os.path.isfile(path)}


def _vision_path_usable():
    return bool(VISION_PATH) and os.path.isfile(VISION_PATH)


def _log_tail(chars=4000):
    return "".join(_log)[-chars:]


def _pump_output(proc):
    # Keep a tail of sd-server's output for error responses, and mirror it to
    # our own stdout so it shows up in `docker logs`.
    global _vision_disabled_line
    for line in proc.stdout:
        _log.append(line)
        if "vision disabled" in line and proc is _proc:
            _vision_disabled_line = line.strip()
        sys.stdout.write(f"[sd-server] {line}")
        sys.stdout.flush()
    code = proc.wait()
    _log.append(f"sd-server exited with code {code}\n")
    sys.stdout.write(f"[sd-server] exited with code {code}\n")
    sys.stdout.flush()


def _start_backend():
    """Launch a new sd-server process. Caller must hold _proc_lock."""
    global _proc, _proc_ready, _vision_disabled_line
    cmd = [
        SD_SERVER,
        "--diffusion-model", MODEL_PATH,
        "--vae", VAE_PATH,
        "--llm", LLM_PATH,
        "--listen-ip", "127.0.0.1",
        "--listen-port", str(SD_SERVER_PORT),
        "--sampling-method", SAMPLING_METHOD,
    ]
    if _vision_path_usable():
        cmd += ["--llm_vision", VISION_PATH]
    elif VISION_PATH:
        app.logger.warning("VISION_PATH %s not found; reference editing will be unavailable", VISION_PATH)
    if DIFFUSION_FA:
        cmd.append("--diffusion-fa")
    if OFFLOAD_TO_CPU:
        cmd.append("--offload-to-cpu")
    cmd += shlex.split(SD_EXTRA_ARGS)

    _log.clear()
    _log.append("$ " + shlex.join(cmd) + "\n")
    _vision_disabled_line = None
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


def _edit_unavailable_reason():
    """Why reference editing can't run, or None if it can.

    Only meaningful once the backend is ready: sd-server reports missing or
    mismatched vision weights while it loads.
    """
    if VISION_PATH and not os.path.isfile(VISION_PATH):
        return f"vision weights not found at VISION_PATH={VISION_PATH}"
    if _vision_disabled_line is not None:
        if not VISION_PATH:
            return ("no vision weights: set VISION_PATH to Qwen3-VL-8B's mmproj GGUF "
                    f"(sd-server: {_vision_disabled_line})")
        return f"sd-server could not use the vision weights: {_vision_disabled_line}"
    return None


def _features():
    reason = _edit_unavailable_reason()
    features = dict(img2img=True, edit=reason is None,
                    max_images=MAX_INPUT_IMAGES if reason is None else 1)
    if reason is not None:
        features["edit_error"] = reason
    return features


@app.errorhandler(413)
def _too_large(_e):
    return jsonify(error=f"request body exceeds MAX_REQUEST_MB={MAX_REQUEST_MB:g}"), 413


@app.get("/health")
def health():
    not_ready = _check_backend()
    if not_ready is not None:
        return jsonify(not_ready[0]), not_ready[1]
    return jsonify(status="ok", **_features())


def _decode_image(value, index):
    """Decode one `images` entry into an RGB or RGBA PIL image (EXIF-rotated).

    Raises BadInput for anything that isn't a decodable image within limits.
    sd-server itself silently skips images it can't decode, which would turn
    an edit into a plain text-to-image without any error, so everything is
    checked (and later re-encoded as PNG) here.
    """
    where = f"images[{index}]"
    if not isinstance(value, str) or not value.strip():
        raise BadInput(f"{where} must be a non-empty base64 string or data URL")
    data = value.strip()
    if data.startswith("data:"):
        header, sep, data = data.partition(",")
        if not sep or not header.endswith(";base64"):
            raise BadInput(f"{where}: only base64 data URLs are supported")
    try:
        raw = base64.b64decode("".join(data.split()), validate=True)
    except (binascii.Error, ValueError):
        raise BadInput(f"{where} is not valid base64") from None
    if not raw:
        raise BadInput(f"{where} is empty")

    try:
        with Image.open(io.BytesIO(raw)) as im:
            w, h = im.size
            if w * h > MAX_INPUT_PIXELS:
                raise BadInput(f"{where} is {w}x{h}, over MAX_INPUT_PIXELS={MAX_INPUT_PIXELS}")
            im.load()
            img = ImageOps.exif_transpose(im)
    except BadInput:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, EOFError) as e:
        raise BadInput(f"{where} could not be decoded as an image: {e}") from None

    has_alpha = img.mode in ("RGBA", "LA", "PA", "RGBa", "La") or (
        img.mode == "P" and "transparency" in img.info)
    return img.convert("RGBA" if has_alpha else "RGB")


def _encode_for_backend(img, out_pixels):
    """PNG-encode an input image for sd-server, scaled down if needlessly large."""
    limit = max(TRANSFER_MAX_PIXELS, out_pixels)
    w, h = img.size
    if w * h > limit:
        scale = math.sqrt(limit / (w * h))
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=1)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _round_to_multiple(value):
    return max(DIM_MULTIPLE, int(round(value / DIM_MULTIPLE)) * DIM_MULTIPLE)


def _output_size(width, height, first_image):
    """Resolve the output size; width/height are explicit values or None."""
    if first_image is None:
        return (1024 if width is None else width), (1024 if height is None else height)
    if width is not None and height is not None:
        return width, height
    src_w, src_h = first_image.size
    aspect = src_w / src_h
    if width is None and height is None:
        w = math.sqrt(AUTO_SIZE_PIXELS * aspect)
        return _round_to_multiple(w), _round_to_multiple(w / aspect)
    if width is None:
        return _round_to_multiple(height * aspect), height
    return width, _round_to_multiple(width / aspect)


def _optional(body, key, convert):
    value = body.get(key)
    return None if value is None else convert(value)


@app.post("/generate")
def generate():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        body = {}

    prompt = body.get("prompt")
    if not prompt:
        return jsonify(error="'prompt' is required"), 400

    try:
        steps = int(body.get("steps", DEFAULT_STEPS))
        cfg = float(body.get("cfg", DEFAULT_CFG))
        width = _optional(body, "width", int)
        height = _optional(body, "height", int)
        seed = body.get("seed")
        seed = secrets.randbelow(2**31) if seed is None else int(seed)
        strength = _optional(body, "strength", float)
    except (TypeError, ValueError) as e:
        return jsonify(error=f"invalid parameter: {e}"), 400
    negative_prompt = body.get("negative_prompt") or ""

    images = body.get("images")
    if images is None:
        images = []
    if not isinstance(images, list):
        return jsonify(error="'images' must be a list of base64 strings or data URLs"), 400
    if len(images) > MAX_INPUT_IMAGES:
        return jsonify(error=f"at most {MAX_INPUT_IMAGES} images are accepted, got {len(images)}"), 400
    if strength is not None:
        if len(images) != 1:
            return jsonify(error="'strength' (img2img) requires exactly one image"), 400
        if not 0.0 < strength <= 1.0:
            return jsonify(error="'strength' must be > 0 and <= 1"), 400
    mode = "text-to-image" if not images else ("img2img" if strength is not None else "edit")

    if steps < 1:
        return jsonify(error="'steps' must be >= 1"), 400
    for name, value in (("width", width), ("height", height)):
        if value is not None and value - value % DIM_MULTIPLE < DIM_MULTIPLE:
            return jsonify(error=f"'{name}' must be >= {DIM_MULTIPLE}"), 400
    if width is not None:
        width -= width % DIM_MULTIPLE
    if height is not None:
        height -= height % DIM_MULTIPLE
    # See the module docstring: never let 0.0 reach sd.cpp (unconditioned
    # mode); everything <= 1 means "no guidance" here.
    cfg = max(cfg, 1.0)

    try:
        decoded = [_decode_image(value, i) for i, value in enumerate(images)]
    except BadInput as e:
        return jsonify(error=str(e)), 400
    width, height = _output_size(width, height, decoded[0] if decoded else None)

    not_ready = _check_backend()
    if not_ready is not None:
        return jsonify(not_ready[0]), not_ready[1]
    if mode == "edit":
        reason = _edit_unavailable_reason()
        if reason is not None:
            return jsonify(error=f"reference editing is unavailable: {reason}. "
                                 "img2img ('strength' with one image) still works."), 400

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
    encoded = [_encode_for_backend(img, width * height) for img in decoded]
    endpoint = "txt2img"
    if mode == "img2img":
        endpoint = "img2img"
        payload["init_images"] = encoded
        payload["denoising_strength"] = strength
    elif mode == "edit":
        payload["extra_images"] = encoded

    req = urllib.request.Request(
        f"{BACKEND_URL}/sdapi/v1/{endpoint}",
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
        return _generation_failed(detail, mode)
    except (TimeoutError, socket.timeout):
        # sd-server can't be interrupted mid-generation; it finishes this job
        # in the background and later requests queue behind it.
        return jsonify(error="generation timed out"), 504
    except (urllib.error.URLError, ConnectionError) as e:
        return jsonify(error=f"generation failed: {e}", stderr=_log_tail()), 500

    try:
        png = base64.b64decode(result["images"][0])
    except (KeyError, IndexError, TypeError, ValueError):
        return _generation_failed(json.dumps(result)[:2000], mode)
    if not png.startswith(b"\x89PNG"):
        return jsonify(error="generation failed: backend did not return a PNG", stderr=_log_tail()), 500

    app.logger.info("%s: %dx%d, %d image(s), %d steps, cfg %.2f in %.1fs", mode, width, height,
                    len(decoded), steps, cfg, time.monotonic() - started)
    return Response(png, mimetype="image/png", headers={"X-Seed": str(seed)})


def _generation_failed(detail, mode):
    global _vision_disabled_line
    # Backstop for the vision check above, in case sd-server didn't log its
    # "vision disabled" warning in the form we look for. Recording it makes
    # later edits fail fast and shows up in /health. sd-server's error line
    # can reach _pump_output a moment after its HTTP error, so give it a
    # little time.
    marker = "editing requires Qwen3-VL vision weights"
    tail = _log_tail()
    deadline = time.monotonic() + 1.0
    while mode == "edit" and marker not in tail and time.monotonic() < deadline:
        time.sleep(0.05)
        tail = _log_tail()
    if mode == "edit" and marker in tail:
        _vision_disabled_line = f"sd-server: Qwen Image 2.1 {marker}"
        return jsonify(error=f"reference editing is unavailable: {_edit_unavailable_reason()}",
                       stderr=tail[-2000:]), 400
    return jsonify(error="generation failed", stderr=(detail + "\n" + tail)[-4000:]), 500


if __name__ == "__main__":
    # Start loading the weights right away so the first /generate doesn't
    # have to wait for it; /health reports "loading" until sd-server is up.
    _check_backend()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5002")), threaded=True)
