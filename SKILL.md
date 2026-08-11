---
name: krea2-api
description: Use this skill whenever the user wants an image generated from a text prompt (e.g. "generate an image of...", "make a picture of...", "draw..."), via the local Krea 2 Turbo text-to-image REST API. Covers issuing /generate requests via curl, required parameters and safe defaults (especially cfg), reading back the resulting PNG, and diagnosing the most common failure modes (wrong cfg, generation timeouts, unhealthy service).
license: Proprietary. LICENSE.txt has complete terms
---

# Krea 2 API Guide

## Overview

Text-to-image generation is available via a REST API at
`http://10.0.2.2:5002` (or wherever the container's port is mapped). POST
to `/generate` to synthesize an image from a prompt; the response body is
the raw PNG bytes on success, or a JSON error object on failure.

There is no async mode and no queue: a `/generate` call blocks until the
image is fully generated and decoded (typically 10-20 seconds on a
consumer GPU), then returns the PNG directly. Sequencing is correct for
free — no polling, no job IDs, no state file.

## Basic Usage

```bash
curl -s -X POST http://10.0.2.2:5002/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a red fox sitting in fresh snow, golden hour, photorealistic"}' \
  -o output.png
```

Always check the output is actually a PNG before treating the call as
successful — a failed request also writes a body to `-o`, but it's a JSON
error, not image bytes:

```bash
file output.png
# output.png: PNG image data, ...   -> success
# output.png: ASCII text, ...       -> read it, it's a JSON error
```

## Request Options

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
it only appears in the container's own verbose logs.

Use `cfg: 1.0` for prompt-following generation with guidance effectively
disabled (the correct "off" value for Turbo). Only raise it above `1.0`
if the user explicitly wants stronger prompt adherence at the cost of
some visual diversity, and only after confirming the model in use
actually benefits from higher CFG (Turbo/distilled checkpoints generally
don't).

## Health Check

Before a batch of generations, or after the container restarts, confirm
the service is ready:

```bash
curl -s http://10.0.2.2:5002/health
```

```json
{"status": "ok"}
```

or, if it's not ready yet:

```json
{"status": "error", ...}
```

An error here means a setup problem on the service side, not something
fixable via the API call itself — surface this to the user rather than
retrying repeatedly.

## Maximum Resolutions (empirically determined, RTX 4090 24 GB VRAM)

The hard limit is VRAM in the VAE decoder. Exceeding these dimensions by even
a few pixels causes an out-of-memory failure with no useful error beyond the
`stderr` field.

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
  almost certainly sent `cfg: 0.0`. Resend with `cfg: 1.0`.
- **`{"error": "..."}` with a 503 status** — the service isn't ready yet.
  Run the health check; this needs a setup fix, not a retry.
- **`{"error": "generation timed out"}`** — the request exceeded the
  server's internal timeout (600s). Very large `width`/`height` or an
  overloaded GPU are the likely causes; retry at a smaller resolution
  first to isolate which.
- **`{"error": "generation failed", "stderr": "..."}`** — read the
  `stderr` field, it's the tail of the underlying generation process's
  own error output and usually names the exact problem.
- **Slow first request after a container restart** — expected; the
  service loads into VRAM on first use each time it starts. Subsequent
  requests in the same session are faster.

## Quick Reference

| Task | Command |
|------|---------|
| Generate an image | `curl -s -X POST http://10.0.2.2:5002/generate -H "Content-Type: application/json" -d '{"prompt": "..."}' -o output.png` |
| Generate with a fixed seed (reproducible) | add `, "seed": 42` to the JSON body |
| Health check | `curl -s http://10.0.2.2:5002/health` |
| Verify the response is really an image | `file output.png` |
| Upscale PNG 2× with Lanczos | `convert in.png -filter Lanczos -resize WxH out.png` |
