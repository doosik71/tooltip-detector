#!/usr/bin/env python3
"""Download the YOLOv8s Cholec80 baseline weights from Hugging Face.

Fetches `yolov8s_cholec80.pt` from

    https://huggingface.co/cesaraha/yolov8s-surgical-instrument-detection-cholec80

into `baseline/yolov8s/data/`. The repository is public and not gated, so no
token is needed; the file is plain LFS content served over HTTPS and is
downloaded with the standard library only (no `huggingface_hub` dependency).

The expected size and SHA-256 come from the Hugging Face model API rather than
being hard-coded, so the check stays valid if the author re-uploads the
weights. The download goes to a `.part` file and is renamed only after the
checksum matches, so an interrupted run never leaves a half-written `.pt`
behind. Re-running is cheap: an already-downloaded file whose checksum matches
is left alone.

Usage:
    uv run python baseline/yolov8s/scripts/download-model.py
    uv run python baseline/yolov8s/scripts/download-model.py --force
    uv run python baseline/yolov8s/scripts/download-model.py --output /tmp/y.pt
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request

REPO_ID = "cesaraha/yolov8s-surgical-instrument-detection-cholec80"
FILENAME = "yolov8s_cholec80.pt"

API_URL = f"https://huggingface.co/api/models/{REPO_ID}?blobs=true"
DOWNLOAD_URL = f"https://huggingface.co/{REPO_ID}/resolve/main/{FILENAME}"

# baseline/yolov8s/scripts/download-model.py -> baseline/yolov8s/data/
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", FILENAME
)

TIMEOUT = 60
CHUNK_SIZE = 1 << 20  # 1 MiB


def fetch_file_info():
    """Return (size, sha256) of FILENAME as recorded in the model repository.

    sha256 is None for files that are not stored in LFS; the weights are, so in
    practice a None here means the repository layout changed.
    """
    try:
        with urllib.request.urlopen(API_URL, timeout=TIMEOUT) as response:
            info = json.load(response)
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        sys.exit(f"error: cannot read model metadata from {API_URL}: {exc}")

    for sibling in info.get("siblings", []):
        if sibling.get("rfilename") == FILENAME:
            lfs = sibling.get("lfs") or {}
            return sibling.get("size"), lfs.get("sha256")

    sys.exit(f"error: {FILENAME} is no longer part of {REPO_ID}")


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, path, expected_size):
    """Stream `url` into `path`, printing progress on a single line."""
    downloaded = 0
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
            total = expected_size or int(response.headers.get("Content-Length") or 0)
            with open(path, "wb") as handle:
                while chunk := response.read(CHUNK_SIZE):
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        print(
                            f"\r  {downloaded / 1e6:7.1f} / {total / 1e6:.1f} MB"
                            f"  ({downloaded * 100 / total:5.1f} %)",
                            end="",
                            flush=True,
                        )
                    else:
                        print(f"\r  {downloaded / 1e6:7.1f} MB", end="", flush=True)
    except urllib.error.URLError as exc:
        sys.exit(f"\nerror: download failed: {exc}")
    finally:
        print()

    return downloaded


def main():
    parser = argparse.ArgumentParser(
        description=f"Download {FILENAME} from {REPO_ID}."
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"destination file path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even if a valid file is already present",
    )
    args = parser.parse_args()

    output = os.path.abspath(args.output)
    size, sha256 = fetch_file_info()
    if sha256 is None:
        sys.exit(f"error: {FILENAME} carries no LFS checksum; cannot verify download")

    print(f"repo    : {REPO_ID}")
    print(f"file    : {FILENAME} ({size / 1e6:.1f} MB)")
    print(f"sha256  : {sha256}")
    print(f"output  : {output}")

    if os.path.exists(output) and not args.force:
        if sha256_of(output) == sha256:
            print("already downloaded and verified; nothing to do")
            return
        print("existing file does not match the expected checksum; re-downloading")

    os.makedirs(os.path.dirname(output), exist_ok=True)

    partial = output + ".part"
    print(f"downloading {DOWNLOAD_URL}")
    downloaded = download(DOWNLOAD_URL, partial, size)

    if size is not None and downloaded != size:
        os.remove(partial)
        sys.exit(f"error: expected {size} bytes but received {downloaded}")

    actual = sha256_of(partial)
    if actual != sha256:
        os.remove(partial)
        sys.exit(f"error: sha256 mismatch (expected {sha256}, got {actual})")

    os.replace(partial, output)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
