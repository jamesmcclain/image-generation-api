#!/usr/bin/env python3
"""Check GET /health once. Run this before a batch of generations, or
after the container restarts.

Usage:
    health_check.py
    health_check.py --host localhost --port 5003

--host/--port pick the server (default 10.0.2.2:5002, the address this skill
has always used).

Prints one terse line:
    OK
    NOT_READY <detail>

Exit codes:
    0 -> ready
    1 -> not ready (setup problem on the service side -- surface to the
         user rather than retrying)
    4 -> network/other error
"""
import argparse
import sys

import requests

DEFAULT_HOST = "10.0.2.2"
DEFAULT_PORT = 5002


def base_url(host: str, port: int) -> str:
    # Bracket bare IPv6 addresses, e.g. ::1 -> http://[::1]:5002
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=DEFAULT_HOST,
                    help=f"API server host (default: {DEFAULT_HOST})")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help=f"API server port (default: {DEFAULT_PORT})")
    args = ap.parse_args()

    try:
        r = requests.get(f"{base_url(args.host, args.port)}/health", timeout=10)
    except requests.RequestException as e:
        print(f"ERROR request failed: {e}")
        return 4

    try:
        body = r.json()
    except ValueError:
        print(f"ERROR {r.status_code} non_json_body {r.text[:200]}")
        return 4

    if body.get("status") == "ok":
        print("OK")
        return 0

    print(f"NOT_READY {body}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
