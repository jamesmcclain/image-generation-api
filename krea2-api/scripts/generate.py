#!/usr/bin/env python3
"""Generate one image via POST /generate.

Uses `requests` so there's no shell-quoting of prompts and no need to
inspect the raw response with `file` afterward -- this script does that
check for you and prints one terse status line instead of a curl
transcript or a JSON dump.

Usage:
    generate.py "a red fox sitting in fresh snow, golden hour" --out output.png
    generate.py "..." --out output.png --width 1647 --height 926 --seed 42

Guards against the cfg trap: refuses to send cfg=0.0 unless you pass
--force-cfg-zero, since that value silently ignores the prompt with no
error from the server.

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
import sys

import requests

BASE_URL = "http://10.0.2.2:5002"

# (max_width, max_height) for the three documented aspect ratios.
_CEILINGS = [
    (1232, 1232),
    (1647, 926),
    (926, 1647),
]


def _over_ceiling(width: int, height: int) -> bool:
    # A request fits if there's some documented ceiling it's within on
    # both axes. If it exceeds every ceiling on at least one axis, reject.
    return not any(width <= mw and height <= mh for mw, mh in _CEILINGS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prompt")
    ap.add_argument("--out", required=True, help="Output .png path")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--cfg", type=float, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--force-cfg-zero",
        action="store_true",
        help="Allow cfg=0.0 (unconditioned generation, prompt ignored). "
        "You almost never want this.",
    )
    ap.add_argument("--timeout", type=float, default=600)
    args = ap.parse_args()

    if args.cfg == 0.0 and not args.force_cfg_zero:
        print(
            "REJECTED_LOCAL cfg=0.0 means unconditioned generation, the "
            "prompt is silently ignored. Use cfg=1.0 for guidance-off "
            "Turbo generation, or pass --force-cfg-zero if you really want this."
        )
        return 2

    w = args.width or 1024
    h = args.height or 1024
    if (args.width or args.height) and _over_ceiling(w, h):
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

    try:
        r = requests.post(f"{BASE_URL}/generate", json=payload, timeout=args.timeout)
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
