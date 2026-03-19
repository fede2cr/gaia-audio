#!/usr/bin/env python3
"""Validate ONNX models at container build time.

Loads each model with onnxruntime, verifies input/output shapes,
and runs a dummy inference to catch issues early — before they
surface at runtime in tract-onnx or ort.

Usage:

    # Classifier (mel-spectrogram already computed):
    python3 validate_onnx.py audio-model.onnx --shape 1,96,511,2

    # Metadata model (lat, lon, week):
    python3 validate_onnx.py meta-model.onnx --shape 1,3

    # End-to-end audio model (raw samples):
    python3 validate_onnx.py birdnet3.onnx --shape 1,96000

    # BatDetect2 (raw samples, 256 kHz × 1 s):
    python3 validate_onnx.py batdetect2.onnx --shape 1,256000

Exit code:
    0 — model loaded, shapes match, dummy inference succeeded
    1 — validation failed (error printed to stderr)
"""

import argparse
import sys
import time


def validate(model_path: str, expected_shape: list[int]) -> None:
    import numpy as np
    import onnxruntime as ort

    # ── 1. Load ──────────────────────────────────────────────────────
    t0 = time.monotonic()
    try:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # suppress ORT warnings
        sess = ort.InferenceSession(model_path, opts)
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model: {e}") from e
    dt = time.monotonic() - t0
    print(f"  Loaded in {dt:.2f}s")

    # ── 2. Check inputs ──────────────────────────────────────────────
    inputs = sess.get_inputs()
    if not inputs:
        raise RuntimeError("Model has no inputs")

    inp = inputs[0]
    print(f"  Input: name={inp.name!r}, shape={inp.shape}, "
          f"type={inp.type}")

    # Resolve symbolic dims — the expected_shape must match the concrete
    # (non-batch) dimensions.
    static_shape = []
    for i, dim in enumerate(inp.shape):
        if isinstance(dim, int):
            static_shape.append(dim)
        else:
            # Symbolic / dynamic dim — use the expected value so the
            # dummy tensor can be built.
            if i < len(expected_shape):
                static_shape.append(expected_shape[i])
            else:
                static_shape.append(1)

    for i, (got, want) in enumerate(zip(static_shape, expected_shape)):
        if got != want:
            raise RuntimeError(
                f"Input dim {i}: model has {got}, expected {want} "
                f"(full: model={static_shape}, expected={expected_shape})"
            )

    # ── 3. Check outputs ─────────────────────────────────────────────
    outputs = sess.get_outputs()
    if not outputs:
        raise RuntimeError("Model has no outputs")
    out = outputs[0]
    print(f"  Output: name={out.name!r}, shape={out.shape}, "
          f"type={out.type}")

    # ── 4. Dummy inference ───────────────────────────────────────────
    # Use low-amplitude random noise rather than silence — models with
    # internal STFT/normalisation layers produce NaN on all-zeros input
    # (0/0 in layer-norm or log-mel), which is expected behaviour.
    rng = np.random.RandomState(42)
    dummy = (rng.randn(*expected_shape) * 0.01).astype(np.float32)
    t0 = time.monotonic()
    try:
        results = sess.run(None, {inp.name: dummy})
    except Exception as e:
        raise RuntimeError(f"Dummy inference failed: {e}") from e
    dt = time.monotonic() - t0

    result = results[0]
    print(f"  Inference: {dt:.3f}s, output shape={result.shape}, "
          f"dtype={result.dtype}")

    # Sanity: output should be float32
    if result.dtype != np.float32:
        raise RuntimeError(
            f"Expected float32 output, got {result.dtype}")

    # Check no NaN/Inf in output (can indicate broken ops)
    if np.any(np.isnan(result)):
        raise RuntimeError("Output contains NaN values")
    if np.any(np.isinf(result)):
        raise RuntimeError("Output contains Inf values")

    num_classes = result.shape[-1]
    print(f"  Classes: {num_classes}")

    # ── 5. tract-onnx compatibility: Reshape shape inputs ────────────
    # tract-onnx rejects Reshape nodes whose shape input is computed
    # by a subgraph rather than being a constant initializer.  Check
    # this at build time so we catch it before runtime.
    try:
        import onnx
        model_onnx = onnx.load(model_path)
        graph = model_onnx.graph
        const_names = {i.name for i in graph.initializer}
        # Also count Constant op outputs as "constant".
        for node in graph.node:
            if node.op_type == "Constant":
                for out in node.output:
                    const_names.add(out)
        bad_reshapes = []
        for node in graph.node:
            if node.op_type == "Reshape" and len(node.input) >= 2:
                if node.input[1] not in const_names:
                    bad_reshapes.append(
                        f"'{node.name}' (shape input: '{node.input[1]}')")
        if bad_reshapes:
            raise RuntimeError(
                f"Reshape node(s) with non-constant shape inputs "
                f"(tract-onnx will reject these): "
                + "; ".join(bad_reshapes))
        print(f"  Reshape shapes: all constant ✓")
    except ImportError:
        print("  (onnx not installed, skipping Reshape check)")

    print(f"  PASS ✓")


def main():
    parser = argparse.ArgumentParser(
        description="Validate an ONNX model at build time")
    parser.add_argument("model", help="Path to .onnx file")
    parser.add_argument(
        "--shape", required=True,
        help="Expected input shape as comma-separated ints, "
             "e.g. 1,96,511,2")
    args = parser.parse_args()

    shape = [int(d) for d in args.shape.split(",")]
    print(f"Validating {args.model} (expected input: {shape})")

    try:
        validate(args.model, shape)
    except RuntimeError as e:
        print(f"  FAIL ✗  {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
