#!/usr/bin/env python3
"""Check GET /health once. Run this before a batch of generations, or
after the container restarts.

Prints one terse line:
    OK
    NOT_READY <detail>

Exit codes:
    0 -> ready
    1 -> not ready (setup problem on the service side -- surface to the
         user rather than retrying)
    4 -> network/other error
"""
import sys

import requests

BASE_URL = "http://10.0.2.2:5002"


def main() -> int:
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=10)
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
