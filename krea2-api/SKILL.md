---
name: krea2-api
description: Use this skill whenever the user wants an image generated from a text prompt (e.g. "generate an image of...", "make a picture of...", "draw..."), via the local Krea 2 Turbo text-to-image REST API. Covers issuing /generate requests via a helper script, required parameters and safe defaults (especially cfg), reading back the resulting PNG, and diagnosing the most common failure modes (wrong cfg, generation timeouts, unhealthy service).
license: Proprietary. LICENSE.txt has complete terms
---

# Krea 2 API Guide

## Overview

Text-to-image generation is available via a REST API at
`http://10.0.2.2:5002` by default. Both scripts take `--host` and `--port`
to reach a server elsewhere (e.g. `--host localhost`, or `--port 5003` for a
second container mapped beside the first); use the same values for every
call in a session.
`/generate` synthesizes an image from a prompt; the response body is the
raw PNG bytes on success, or a JSON error object on failure.

There is no async mode and no queue: a `/generate` call blocks until the
image is fully generated and decoded (typically 10-20 seconds on a
consumer GPU), then returns the PNG directly. Sequencing is correct for
free — no polling, no job IDs, no state file.

**Use `scripts/*.py`, not raw `curl`.** They use the `requests` library, so
the prompt never needs shell-quoting, and they check the response is
actually a PNG (and, for `generate.py`, that the request isn't doomed
before it's even sent) instead of leaving that to a separate `file`
call. Each prints one terse status line rather than a raw response dump —
this keeps tool output small across a session with several generations.
Both scripts need `requests` installed (`pip install requests` if it's
missing).

## Scripts in 30 Seconds

**Generate:**
```bash
python3 scripts/generate.py "a red fox sitting in fresh snow, golden hour, photorealistic" --out output.png
```
Prints one of:
| Line | Meaning |
|------|---------|
| `OK wrote <path> bytes=<n>` | Success. `<path>` is a real PNG. |
| `REJECTED_LOCAL ...` | The script refused to send the request — see below. |
| `ERROR <code> ...` | The server rejected the request or didn't return a PNG. |
| `ERROR request failed: ...` | Network-level failure (couldn't reach the server). |

Exit code matches: `0` success, `2` rejected locally, `3` server error /
non-PNG response, `4` network error.

`generate.py` rejects two requests locally, before touching the network:
- `--cfg 0.0` (see "The cfg trap" below) — pass `--force-cfg-zero` to
  override, but you almost never want to.
- A `--width`/`--height` combination that exceeds every documented
  ceiling (see "Maximum Resolutions" below) — this would otherwise burn
  10-20s on a server-side OOM with an unhelpful error.

**Health check:**
```bash
python3 scripts/health_check.py
python3 scripts/health_check.py --host localhost --port 5003
# -> OK
# -> NOT_READY {...}
```
Exit code `0` means ready, `1` means not ready. Run this before a batch of
generations, or after the container restarts. `NOT_READY` means a setup
problem on the service side — surface it to the user rather than retrying
repeatedly.

**Options for `generate.py`:** positional `PROMPT`, plus `--out PATH`
(required), `--steps N`, `--cfg F`, `--width N`, `--height N`, `--seed N`,
`--timeout SECONDS` (default 600, matching the server's own internal
timeout), and `--host HOST` / `--port PORT` (default `10.0.2.2` / `5002`).
`health_check.py` takes only `--host` and `--port`. Don't pass anything
else — the server only accepts the documented fields.

## Request Options

The fields `generate.py` accepts map directly onto the API body:

```json
{
  "prompt": "required, the text description of the image",
  "steps": 8,        // optional, default 8 — Turbo is distilled for ~8 steps
  "cfg": 1.0,         // optional, default 1.0 — see "The cfg trap" below
  "width": 1024,      // optional, default 1024
  "height": 1024,     // optional, default 1024
  "seed": 42          // optional; omit for a random seed each call
}
```

### The `cfg` trap

**Never send `cfg: 0.0`.** It looks like the natural "guidance off" value
for a distilled/Turbo model, but this backend treats exactly `0.0` as
fully **unconditioned** generation — the prompt is silently ignored and
every request returns visually similar, generic output regardless of what
you asked for. There is no error or warning surfaced to the API caller;
it only appears in the container's own verbose logs. `generate.py` refuses
`--cfg 0.0` locally for this reason (see above).

Use `cfg: 1.0` for prompt-following generation with guidance effectively
disabled (the correct "off" value for Turbo). Only raise it above `1.0`
if the user explicitly wants stronger prompt adherence at the cost of
some visual diversity, and only after confirming the model in use
actually benefits from higher CFG (Turbo/distilled checkpoints generally
don't).

## Maximum Resolutions (empirically determined, RTX 4090 24 GB VRAM)

The hard limit is VRAM in the VAE decoder. Exceeding these dimensions by even
a few pixels causes an out-of-memory failure with no useful error beyond the
`stderr` field. `generate.py` checks your `--width`/`--height` against these
ceilings before sending the request.

| Aspect ratio | Maximum resolution |
|---|---|
| Square (1:1) | **1232 × 1232** |
| Widescreen (16:9) | **1647 × 926** |
| Portrait (9:16) | **926 × 1647** |

If you need a larger output, generate at the maximum native resolution and
upscale afterwards (see below).

## Upscaling with ImageMagick

If `convert` (ImageMagick) is available on the system, you can upscale a
generated PNG to any target resolution after the fact. Lanczos is a good
default filter for photorealistic content:

```bash
# Check availability
convert --version

# Upscale to 1440p (2× a 16:9 native image)
convert input.png -filter Lanczos -resize 2560x1440 output.png

# Upscale to 4K square
convert input.png -filter Lanczos -resize 2464x2464 output.png
```

Always `file output.png` afterwards to confirm the resize succeeded.

## Troubleshooting

- **Every image looks like a generic default, ignoring the prompt** — you
  almost certainly sent `cfg: 0.0` (with `--force-cfg-zero`, or via a raw
  request outside these scripts). Resend with `--cfg 1.0`.
- **`REJECTED_LOCAL` mentioning a resolution ceiling** — lower
  `--width`/`--height` to fit one of the three documented ceilings, or
  generate at the ceiling and upscale afterward.
- **`ERROR 503 ...`** — the service isn't ready yet. Run
  `health_check.py`; this needs a setup fix, not a retry.
- **`ERROR ... "generation timed out"`** — the request exceeded the
  server's internal timeout (600s). Very large `width`/`height` or an
  overloaded GPU are the likely causes; retry at a smaller resolution
  first to isolate which.
- **`ERROR ... "generation failed", "stderr": "..."`** — read the
  `stderr` field in the printed detail, it's the tail of the underlying
  generation process's own error output and usually names the exact
  problem.
- **Slow first request after a container restart** — expected; the
  service loads into VRAM on first use each time it starts. Subsequent
  requests in the same session are faster.
- **`ModuleNotFoundError: requests`** — run `pip install requests` in the
  environment these scripts execute in.

## Quick Reference

| Task | Command |
|------|---------|
| Generate an image | `python3 scripts/generate.py "..." --out output.png` |
| Generate with a fixed seed (reproducible) | add `--seed 42` |
| Generate at a specific size | add `--width 1647 --height 926` (stay within a ceiling above) |
| Health check | `python3 scripts/health_check.py` |
| Use a different server | add `--host HOST --port PORT` to either script |
| Upscale PNG 2× with Lanczos | `convert in.png -filter Lanczos -resize WxH out.png` |
