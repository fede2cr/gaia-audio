#!/usr/bin/env python3
"""Convert a TFLite model to ONNX format for use with tract-onnx.

tract-tflite does not support every TFLite operator (notably SPLIT_V, used
by BirdNET V2.4).  Converting the model to ONNX sidesteps the issue because
tract-onnx has broader operator coverage including Split/SplitToSequence.

Usage
-----
    # Install the converter (once)
    pip install tf2onnx

    # Convert a single model
    python scripts/convert_tflite_to_onnx.py models/birds/audio-model.tflite

    # Specify output path explicitly
    python scripts/convert_tflite_to_onnx.py models/birds/audio-model.tflite -o models/birds/audio-model.onnx

    # Convert all .tflite files in a directory
    python scripts/convert_tflite_to_onnx.py models/birds/

When run without ``-o``, the output is placed next to the input with an
``.onnx`` extension (e.g. ``audio-model.tflite`` → ``audio-model.onnx``).

After conversion, set ``onnx_file`` in the model's ``manifest.toml``::

    [model]
    onnx_file = "audio-model.onnx"

The processing server will then prefer the ONNX file over the TFLite one.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def convert_one(src: Path, dst: Path) -> bool:
    """Convert a single TFLite file to ONNX. Returns True on success."""
    if dst.exists():
        print(f"  skip (already exists): {dst}")
        return True

    print(f"  {src} → {dst}")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tf2onnx.convert",
            "--tflite",
            str(src),
            "--output",
            str(dst),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  ERROR converting {src}:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return False

    size_mb = dst.stat().st_size / (1024 * 1024)
    print(f"  ok ({size_mb:.1f} MB)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert TFLite models to ONNX for use with Gaia processing server."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a .tflite file or a directory containing .tflite files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .onnx path (only valid when input is a single file).",
    )
    args = parser.parse_args()

    # Check for tf2onnx
    try:
        import tf2onnx  # noqa: F401
    except ImportError:
        print("Error: tf2onnx is not installed. Run:", file=sys.stderr)
        print("  pip install tf2onnx", file=sys.stderr)
        sys.exit(1)

    src: Path = args.input

    if src.is_file():
        dst = args.output or src.with_suffix(".onnx")
        ok = convert_one(src, dst)
        sys.exit(0 if ok else 1)

    if src.is_dir():
        if args.output is not None:
            parser.error("-o/--output cannot be used when input is a directory")

        tflite_files = sorted(src.rglob("*.tflite"))
        if not tflite_files:
            print(f"No .tflite files found in {src}", file=sys.stderr)
            sys.exit(1)

        print(f"Converting {len(tflite_files)} file(s) in {src}:")
        failures = 0
        for tf_path in tflite_files:
            onnx_path = tf_path.with_suffix(".onnx")
            if not convert_one(tf_path, onnx_path):
                failures += 1

        if failures:
            print(f"\n{failures} conversion(s) failed.", file=sys.stderr)
            sys.exit(1)
        print("\nAll conversions succeeded.")
        sys.exit(0)

    print(f"Error: {src} is not a file or directory", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
