"""Create a Modal proxy token and store it in the ignored local .env file."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from dotenv import set_key


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("endpoint_url")
    args = parser.parse_args()

    completed = subprocess.run(
        [
            str(ROOT / ".venv" / "bin" / "modal"),
            "workspace",
            "proxy-tokens",
            "create",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    token = json.loads(completed.stdout)
    token_id = token["Modal-Key"]
    token_secret = token["Modal-Secret"]

    ENV_PATH.touch(mode=0o600, exist_ok=True)
    os.chmod(ENV_PATH, 0o600)
    set_key(ENV_PATH, "MODAL_ENDPOINT_URL", args.endpoint_url, quote_mode="never")
    set_key(ENV_PATH, "MODAL_PROXY_TOKEN_ID", token_id, quote_mode="never")
    set_key(ENV_PATH, "MODAL_PROXY_TOKEN_SECRET", token_secret, quote_mode="never")
    set_key(ENV_PATH, "RECONVIAGEN_REQUEST_TIMEOUT_SECONDS", "3600", quote_mode="never")

    print(f"Configured protected Modal endpoint with proxy token {token_id[:7]}…")


if __name__ == "__main__":
    main()
