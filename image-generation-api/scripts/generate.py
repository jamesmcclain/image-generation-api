#!/usr/bin/env python3
"""Generate one image via POST /generate, optionally from input images.

Uses `requests` so there's no shell-quoting of prompts and no need to
inspect the raw response with `file` afterward -- this script does that
check for you and prints one terse status line instead of a curl
transcript or a JSON dump.

Usage:
    generate.py "a red fox sitting in fresh snow, golden hour" --out output.png
    generate.py "..." --out output.png --width 1647 --height 926 --seed 42
    generate.py "..." --out output.png --host localhost --port 5003

Input images (Qwen-Image-2.1 server only; `health_check.py` shows whether
the server supports them):
    generate.py "make <image1> a watercolor painting" --image in.png --out out.png
    generate.py "put the shirt from <image2> on the person in <image1>" \
        --image person.png --image shirt.png --out out.png
    generate.py "a snowy version of this scene" --image in.jpg --strength 0.6 --out out.png

--image (repeatable) sends reference images for editing; the prompt refers
to them as <image1>, <image2> in order. Adding --strength (0-1) with a single
--image does img2img instead. With input images, omitting --width/--height
makes the output follow the first image's aspect ratio.

--host/--port pick the server (default 10.0.2.2:5002, the address this skill
has always used).

Guards against the cfg trap: refuses to send cfg=0.0 unless you pass
--force-cfg-zero, since that value silently ignores the prompt with no
error from the server.

Also rejects locally: --strength without exactly one --image, and --image
files that can't be read.

Also guards against the known VRAM ceiling (RTX 4090, empirically
determined): 1232x1232 square, 1647x926 widescreen, 926x1647 portrait.
Requesting above the relevant ceiling fails fast, locally, instead of
burning 10-20s on a server-side OOM with an unhelpful error.

Exit codes:
    0 -> wrote a PNG
    2 -> rejected locally (cfg trap or resolution ceiling)
    3 -> 4xx/5xx from the server, or response wasn't a PNG
    4 -> network/other error
"""
import argparse
import base64
import sys

import requests

DEFAULT_HOST = "10.0.2.2"
DEFAULT_PORT = 5002

# (max_width, max_height) for the three documented aspect ratios.
_CEILINGS = [
    (1232, 1232),
    (1647, 926),
    (926, 1647),
]


def base_url(host: str, port: int) -> str:
    # Bracket bare IPv6 addresses, e.g. ::1 -> http://[::1]:5002
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}"


def _over_ceiling(width: int, height: int) -> bool:
    # A request fits if there's some documented ceiling it's within on
    # both axes. If it exceeds every ceiling on at least one axis, reject.
    return not any(width <= mw and height <= mh for mw, mh in _CEILINGS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt")
    ap.add_argument("--out", required=True, help="Output .png path")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--cfg", type=float, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--image", action="append", default=[], metavar="PATH",
                    help="Input image (repeatable, in <image1>, <image2> order)")
    ap.add_argument("--strength", type=float, default=None,
                    help="img2img strength, 0-1; requires exactly one --image")
    ap.add_argument(
        "--force-cfg-zero",
        action="store_true",
        help="Allow cfg=0.0 (unconditioned generation, prompt ignored). "
        "You almost never want this.",
    )
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--host", default=DEFAULT_HOST,
                    help=f"API server host (default: {DEFAULT_HOST})")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help=f"API server port (default: {DEFAULT_PORT})")
    args = ap.parse_args()

    if args.cfg == 0.0 and not args.force_cfg_zero:
        print(
            "REJECTED_LOCAL cfg=0.0 means unconditioned generation, the "
            "prompt is silently ignored. Use cfg=1.0 for guidance-off "
            "Turbo generation, or pass --force-cfg-zero if you really want this."
        )
        return 2

    if args.strength is not None and len(args.image) != 1:
        print("REJECTED_LOCAL --strength (img2img) requires exactly one --image")
        return 2
    if args.strength is not None and not 0.0 < args.strength <= 1.0:
        print("REJECTED_LOCAL --strength must be > 0 and <= 1")
        return 2

    images = []
    for path in args.image:
        try:
            with open(path, "rb") as f:
                images.append(base64.b64encode(f.read()).decode("ascii"))
        except OSError as e:
            print(f"REJECTED_LOCAL cannot read --image {path}: {e}")
            return 2

    w = args.width or 1024
    h = args.height or 1024
    # With input images, a missing dimension is derived server-side from the
    # first image, so only a fully explicit size can be checked here.
    size_known = (args.width and args.height) if images else (args.width or args.height)
    if size_known and _over_ceiling(w, h):
        print(
            f"REJECTED_LOCAL {w}x{h} exceeds every known ceiling "
            f"(1232x1232 square, 1647x926 widescreen, 926x1647 portrait)"
        )
        return 2

    payload = {"prompt": args.prompt}
    if args.steps is not None:
        payload["steps"] = args.steps
    if args.cfg is not None:
        payload["cfg"] = args.cfg
    if args.width is not None:
        payload["width"] = args.width
    if args.height is not None:
        payload["height"] = args.height
    if args.seed is not None:
        payload["seed"] = args.seed
    if images:
        payload["images"] = images
    if args.strength is not None:
        payload["strength"] = args.strength

    try:
        r = requests.post(f"{base_url(args.host, args.port)}/generate", json=payload, timeout=args.timeout)
    except requests.RequestException as e:
        print(f"ERROR request failed: {e}")
        return 4

    if r.status_code != 200 or not r.content.startswith(b"\x89PNG"):
        try:
            detail = r.json()
        except ValueError:
            detail = r.text[:300]
        print(f"ERROR {r.status_code} not_a_png {detail}")
        return 3

    with open(args.out, "wb") as f:
        f.write(r.content)
    print(f"OK wrote {args.out} bytes={len(r.content)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
