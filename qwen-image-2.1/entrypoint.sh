#!/bin/sh
# entrypoint.sh — pick what the container runs, via its first argument:
#
#   api       (default) the REST API on port 5002 (POST /generate, GET
#             /health), backed by stable-diffusion.cpp's sd-server
#   comfyui   ComfyUI on port 8188. Any further arguments are passed to
#             ComfyUI's main.py, e.g. `comfyui --lowvram`.
#   anything else is executed as a command, e.g. `bash` or `/app/sd-cli --help`
#
# Both modes read the same model files from /models, laid out the way
# ComfyUI expects (diffusion_models/, text_encoders/, vae/, loras/).
set -e

mode="${1:-api}"
[ "$#" -gt 0 ] && shift

case "$mode" in
    api)
        exec python3 /app/server.py "$@"
        ;;
    comfyui)
        if [ ! -f /opt/ComfyUI/main.py ]; then
            echo "This image was built without ComfyUI (--build-arg WITH_COMFYUI=0)." >&2
            exit 1
        fi
        # Everything ComfyUI writes (outputs, inputs, user settings/workflows,
        # its database, custom nodes, temp files) goes under COMFYUI_DATA, so
        # the install in /opt/ComfyUI can stay read-only and owned by root.
        # custom_nodes must exist up front: ComfyUI's startup-script scan
        # crashes on a fresh (empty) base directory without it.
        mkdir -p "${COMFYUI_DATA}/custom_nodes"
        # shellcheck disable=SC2086  # COMFYUI_ARGS is intentionally word-split
        exec python3 /opt/ComfyUI/main.py \
            --listen "${COMFYUI_LISTEN:-0.0.0.0}" \
            --port "${COMFYUI_PORT:-8188}" \
            --base-directory "${COMFYUI_DATA}" \
            --extra-model-paths-config /app/extra_model_paths.yaml \
            --disable-auto-launch \
            ${COMFYUI_ARGS:-} "$@"
        ;;
    *)
        exec "$mode" "$@"
        ;;
esac
