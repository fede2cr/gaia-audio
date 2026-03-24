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


def validate(
    model_path: str,
    expected_shape: list[int],
    enforce_tract_checks: bool = True,
) -> None:
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

    # ── 5. tract-onnx compatibility checks ─────────────────────────────
    # tract-onnx is stricter than onnxruntime.  Catch incompatibilities
    # at build time so they don't surface after publishing the container.
    if not enforce_tract_checks:
        print("  tract-onnx compatibility checks skipped (--skip-tract-checks)")
        print(f"  PASS ✓")
        return

    try:
        import onnx
        from onnx import numpy_helper
        model_onnx = onnx.load(model_path)
        graph = model_onnx.graph
        tract_errors: list[str] = []

        # ── 5a. Reshape shape inputs must be graph initializers ──────
        # tract-onnx only recognises graph initializers as "constant".
        # Constant-op outputs and compute-subgraph results are rejected
        # with: "shape input is variable".
        init_names = {i.name for i in graph.initializer}
        bad_reshapes = []
        for node in graph.node:
            if node.op_type == "Reshape" and len(node.input) >= 2:
                if node.input[1] not in init_names:
                    bad_reshapes.append(
                        f"'{node.name}' (shape input: '{node.input[1]}')")
        if bad_reshapes:
            tract_errors.append(
                f"Reshape node(s) with non-initializer shape inputs "
                f"(tract-onnx will reject these): "
                + "; ".join(bad_reshapes))
        else:
            print(f"  Reshape shapes: all initializers ✓")

        # ── 5b. Reshape shape values must be valid integers ──────────
        # Catch bogus float-valued shape tensors that would fail at
        # runtime (e.g. [1.001953125, 188.5, 128.0]).
        for node in graph.node:
            if node.op_type == "Reshape" and len(node.input) >= 2:
                shape_name = node.input[1]
                for init in graph.initializer:
                    if init.name == shape_name:
                        arr = numpy_helper.to_array(init).flatten()
                        for i, v in enumerate(arr):
                            fv = float(v)
                            if abs(fv - round(fv)) > 1e-6:
                                tract_errors.append(
                                    f"Reshape '{node.name}': shape "
                                    f"element [{i}] = {fv} is not an "
                                    f"integer (initializer '{shape_name}')")
                        break

        # ── 5c. Resize: no pytorch_half_pixel ────────────────────────
        # tract-onnx does not support coordinate_transformation_mode =
        # "pytorch_half_pixel".
        for node in graph.node:
            if node.op_type == "Resize":
                for attr in node.attribute:
                    if attr.name == "coordinate_transformation_mode":
                        val = (attr.s.decode("utf-8")
                               if isinstance(attr.s, bytes) else attr.s)
                        if val == "pytorch_half_pixel":
                            tract_errors.append(
                                f"Resize node '{node.name}' uses "
                                f"unsupported coordinate_transformation_mode "
                                f"'pytorch_half_pixel'")

        # ── 5d. DFT/STFT: no onesided=1 ─────────────────────────────
        # tract's DFT shape analysis enforces input[2] == output[2],
        # which fails for one-sided DFT (output = N/2+1).
        for node in graph.node:
            if node.op_type in ("DFT", "STFT"):
                for attr in node.attribute:
                    if attr.name == "onesided" and attr.i == 1:
                        tract_errors.append(
                            f"{node.op_type} node '{node.name}' has "
                            f"onesided=1 (tract-onnx rejects this)")

        # ── 5e. Residual symbolic dimensions ─────────────────────────
        # tract-onnx cannot parse expressions like ((samples//512))+1.
        # After patching, no dim_param should remain.
        symbolic_dims = []
        for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
            if not vi.type.HasField("tensor_type"):
                continue
            for d in vi.type.tensor_type.shape.dim:
                if d.dim_param and d.dim_param not in ("batch_size", "N"):
                    # Allow common benign symbolic names; reject
                    # expressions (contain operators or //).
                    if any(c in d.dim_param for c in ("//", "+", "*", "-")):
                        symbolic_dims.append(
                            f"'{vi.name}' dim='{d.dim_param}'")
        if symbolic_dims:
            # Only warn — some models legitimately use simple symbolic
            # dims (e.g. "batch") that tract handles fine.
            print(f"  WARNING: {len(symbolic_dims)} symbolic dimension(s) "
                  f"with expressions: {'; '.join(symbolic_dims[:5])}")

        if tract_errors:
            raise RuntimeError(
                f"{len(tract_errors)} tract-onnx compatibility issue(s):\n  "
                + "\n  ".join(tract_errors))
        print(f"  tract-onnx compatibility: all checks passed ✓")
    except ImportError:
        print("  (onnx not installed, skipping tract compatibility checks)")

    print(f"  PASS ✓")


def main():
    parser = argparse.ArgumentParser(
        description="Validate an ONNX model at build time")
    parser.add_argument("model", help="Path to .onnx file")
    parser.add_argument(
        "--shape", required=True,
        help="Expected input shape as comma-separated ints, "
             "e.g. 1,96,511,2")
    parser.add_argument(
        "--skip-tract-checks",
        action="store_true",
        help="Validate ONNX Runtime load/inference only; skip tract-onnx compatibility checks",
    )
    args = parser.parse_args()

    shape = [int(d) for d in args.shape.split(",")]
    print(f"Validating {args.model} (expected input: {shape})")

    try:
        validate(
            args.model,
            shape,
            enforce_tract_checks=not args.skip_tract_checks,
        )
    except RuntimeError as e:
        print(f"  FAIL ✗  {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
