#!/usr/bin/env python3
"""Download BirdNET+ V3.0 ONNX model from Zenodo and patch for tract-onnx.

tract-onnx does not support the ``Resize`` operator with
``coordinate_transformation_mode = "pytorch_half_pixel"``.  This script:

 1. Downloads the original ONNX model from Zenodo record 18247420.
 2. Replaces ``pytorch_half_pixel`` with ``half_pixel`` in every
    ``Resize`` node that uses it.  The two modes are functionally
    equivalent for fractional scale factors (they only differ when
    the scale is exactly 1, which is not the case here).
 3. Saves the patched model to the output path.

Usage::

    python3 patch_birdnet3_onnx.py -o /convert/birdnet3.onnx

Dependencies: ``onnx`` (pip install onnx)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request

import onnx

ZENODO_RECORD = "18247420"
ONNX_FILENAME = "BirdNET%2B_V3.0-preview3_Global_11K_FP32.onnx"
ONNX_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{ONNX_FILENAME}/content"
EXPECTED_MD5 = "7b32222e13fd61114246b46a4c88d09d"

OLD_MODE = "pytorch_half_pixel"
NEW_MODE = "half_pixel"


def download(url: str, dest: str) -> None:
    """Download a file with progress reporting."""
    print(f"Downloading {url} ...")
    # Show progress every 50 MB
    chunk_size = 8192
    downloaded = 0
    last_report = 0
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            mb = downloaded / (1024 * 1024)
            if mb - last_report >= 50:
                print(f"  {mb:.0f} MB downloaded ...", flush=True)
                last_report = mb
    print(f"  Download complete: {downloaded / (1024 * 1024):.1f} MB")


def verify_md5(path: str, expected: str) -> bool:
    """Verify the MD5 checksum of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        print(f"MD5 mismatch: expected {expected}, got {actual}", file=sys.stderr)
        return False
    print(f"  MD5 verified: {actual}")
    return True


def patch_resize_nodes(model: onnx.ModelProto) -> int:
    """Patch Resize nodes: pytorch_half_pixel → half_pixel.

    Returns the number of nodes patched.
    """
    count = 0
    for node in model.graph.node:
        if node.op_type != "Resize":
            continue
        for attr in node.attribute:
            if attr.name == "coordinate_transformation_mode":
                val = attr.s.decode("utf-8") if isinstance(attr.s, bytes) else attr.s
                if val == OLD_MODE:
                    attr.s = NEW_MODE.encode("utf-8")
                    count += 1
                    print(f"  Patched node {node.name}: {OLD_MODE} → {NEW_MODE}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch BirdNET+ V3.0 ONNX for tract-onnx")
    parser.add_argument("-o", "--output", required=True, help="Output path for patched ONNX model")
    parser.add_argument("--keep-download", action="store_true", help="Keep the original downloaded file")
    args = parser.parse_args()

    work_dir = os.path.dirname(args.output) or "."
    os.makedirs(work_dir, exist_ok=True)
    raw_path = os.path.join(work_dir, "birdnet3_original.onnx")

    # Download
    if os.path.exists(raw_path):
        print(f"Using existing download: {raw_path}")
    else:
        download(ONNX_URL, raw_path)

    # Verify
    if not verify_md5(raw_path, EXPECTED_MD5):
        sys.exit(1)

    # Load and patch
    print("Loading ONNX model (this may take a moment for a 541 MB file) ...")
    model = onnx.load(raw_path)
    print(f"  Model IR version: {model.ir_version}, opset: {model.opset_import[0].version}")

    patched = patch_resize_nodes(model)
    if patched == 0:
        print("WARNING: No Resize nodes with pytorch_half_pixel found — saving unmodified model")
    else:
        print(f"Patched {patched} Resize node(s)")

    # Save
    print(f"Saving patched model to {args.output} ...")
    onnx.save(model, args.output)
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"  Saved: {args.output} ({size_mb:.1f} MB)")

    # Cleanup
    if not args.keep_download and os.path.exists(raw_path):
        os.remove(raw_path)
        print("  Removed original download")

    print("Done.")


if __name__ == "__main__":
    main()
