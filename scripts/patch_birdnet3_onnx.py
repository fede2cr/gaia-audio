#!/usr/bin/env python3
"""Download BirdNET+ V3.0 ONNX model from Zenodo and patch for tract-onnx.

tract-onnx does not support:

 * The ``Resize`` operator with
   ``coordinate_transformation_mode = "pytorch_half_pixel"``.
 * Symbolic dimension expressions that use Python-style integer
   division (``//``), e.g. ``((samples//512)) + 1``.
 * One-sided DFT (``onesided=1``): tract's DFT shape analysis
   enforces ``inputs[0].shape[2] == outputs[0].shape[2]`` which
   fails when output is ``N/2+1`` (1025 vs 2048).

This script:

 1. Downloads the original ONNX model from Zenodo record 18247420.
 2. Replaces ``pytorch_half_pixel`` with ``half_pixel`` in every
    ``Resize`` node.
 3. Rewrites one-sided DFT nodes as full DFT + Slice so that
    tract's shape check passes.
 4. Freezes the model input to the concrete shape ``[1, 96000]``
    (batch=1, 3 s × 32 kHz) and runs ONNX shape inference so that
    **all** symbolic dimensions are replaced by concrete integers.
 5. Saves the patched model to the output path.

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
from onnx import TensorProto, helper, numpy_helper, shape_inference

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


def patch_onesided_dft_nodes(model: onnx.ModelProto) -> int:
    """Replace one-sided DFT nodes with full DFT + Slice.

    tract-onnx's DFT analysis enforces ``inputs[0].shape[2] ==
    outputs[0].shape[2]`` without accounting for the ``onesided``
    attribute.  When ``onesided=1``, the output length along the
    signal axis is ``floor(N/2)+1`` (e.g. 1025 for N=2048), which
    violates that rule.

    Fix: set ``onesided=0`` (full DFT, output length == input length)
    and insert a ``Slice`` node after to truncate the frequency axis
    to ``floor(N/2)+1`` bins, matching what downstream nodes expect.

    This function uses three strategies to determine the DFT length:

      1. Read ``dft_length`` from a graph initializer.
      2. Read ``dft_length`` from a Constant node output.
      3. When ``dft_length`` is not provided as an input, infer from
         the signal tensor's resolved shape at the DFT ``axis``
         dimension (works after ``freeze_input_shapes`` has been run).

    Call this function **both** before and after ``freeze_input_shapes``
    to maximise coverage: the first pass handles nodes whose length is
    statically known, and the second pass handles nodes whose signal
    shapes become concrete only after shape inference.

    Returns the number of DFT nodes patched.
    """
    import numpy as np

    graph = model.graph
    nodes_to_add: list[onnx.NodeProto] = []
    inits_to_add: list[onnx.TensorProto] = []
    fixed = 0

    # Build lookup: Constant-node output name → scalar int value.
    const_values: dict[str, int] = {}
    for cnode in graph.node:
        if cnode.op_type == "Constant" and len(cnode.output) == 1:
            for attr in cnode.attribute:
                if attr.name == "value" and attr.t is not None:
                    try:
                        arr = numpy_helper.to_array(attr.t)
                        const_values[cnode.output[0]] = int(arr.flat[0])
                    except Exception:
                        pass

    # Build shape lookup from value_info, graph inputs, and graph outputs.
    shape_map: dict[str, list[int | None]] = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        if not vi.type.HasField("tensor_type"):
            continue
        dims: list[int | None] = []
        for d in vi.type.tensor_type.shape.dim:
            dims.append(int(d.dim_value) if d.dim_value > 0 else None)
        shape_map[vi.name] = dims

    for node in list(graph.node):
        if node.op_type not in ("DFT", "STFT"):
            continue

        # Check if onesided=1
        onesided_attr = None
        for attr in node.attribute:
            if attr.name == "onesided":
                onesided_attr = attr
                break

        if onesided_attr is None or onesided_attr.i != 1:
            continue

        # ── Read the axis attribute (used for shape inference + Slice) ─
        # ONNX DFT default axis varies by opset; BirdNET V3 uses axis 2
        # (frequency axis in the STFT frame layout).
        axis = 2
        for attr in node.attribute:
            if attr.name == "axis":
                axis = int(attr.i)
                break

        # ── Determine the DFT length ──────────────────────────────────
        dft_length = None

        # Strategy 1: dft_length input from graph initializers.
        # For DFT: input[1] is dft_length (optional)
        # For STFT: input[2] is frame_length (the per-frame DFT size)
        dft_len_idx = 1 if node.op_type == "DFT" else 2
        if len(node.input) > dft_len_idx and node.input[dft_len_idx]:
            dft_len_name = node.input[dft_len_idx]
            # Check initializers
            for init in graph.initializer:
                if init.name == dft_len_name:
                    arr = numpy_helper.to_array(init)
                    dft_length = int(arr.flat[0])
                    break

        # Strategy 2: dft_length input from a Constant node output.
        if dft_length is None and len(node.input) > dft_len_idx and node.input[dft_len_idx]:
            dft_len_name = node.input[dft_len_idx]
            if dft_len_name in const_values:
                dft_length = const_values[dft_len_name]

        # Strategy 3: no explicit dft_length input — the DFT operates on
        # the full extent of the signal along `axis`.  Read the concrete
        # dimension from value_info (available after freeze_input_shapes).
        if dft_length is None:
            signal_name = node.input[0]
            if signal_name in shape_map:
                dims = shape_map[signal_name]
                ndim = len(dims)
                resolved_axis = axis if axis >= 0 else ndim + axis
                if 0 <= resolved_axis < ndim and dims[resolved_axis] is not None:
                    dft_length = dims[resolved_axis]

        if dft_length is None:
            print(
                f"  WARNING: cannot determine DFT length for {node.name} "
                f"— skipping onesided patch (will retry after shape freeze)"
            )
            continue

        onesided_bins = dft_length // 2 + 1
        print(
            f"  Patching {node.op_type} node '{node.name}': "
            f"onesided=1 → onesided=0 + Slice(axis={axis}, end={onesided_bins})"
        )

        # 1. Set onesided = 0
        onesided_attr.i = 0

        # 2. Rename the original output and wire a Slice after it.
        orig_output = node.output[0]
        full_output = f"{orig_output}_full_dft"
        node.output[0] = full_output

        # Slice parameters: start=0, end=onesided_bins on the DFT axis.
        # Using the ONNX Slice op (opset 10+): inputs = [data, starts, ends, axes]
        uid = f"_dft_slice_{node.name}"

        starts_name = f"{uid}_starts"
        ends_name = f"{uid}_ends"
        axes_name = f"{uid}_axes"

        inits_to_add.append(
            numpy_helper.from_array(
                np.array([0], dtype=np.int64), name=starts_name
            )
        )
        inits_to_add.append(
            numpy_helper.from_array(
                np.array([onesided_bins], dtype=np.int64), name=ends_name
            )
        )
        inits_to_add.append(
            numpy_helper.from_array(
                np.array([axis], dtype=np.int64), name=axes_name
            )
        )

        slice_node = helper.make_node(
            "Slice",
            inputs=[full_output, starts_name, ends_name, axes_name],
            outputs=[orig_output],
            name=f"slice_onesided{uid}",
        )
        nodes_to_add.append(slice_node)
        fixed += 1

    # Add new nodes and initializers to the graph
    for n in nodes_to_add:
        graph.node.append(n)
    for init in inits_to_add:
        graph.initializer.append(init)

    return fixed


# ── Input shape: 1 batch × (sample_rate × chunk_duration) samples ─────
BATCH_SIZE = 1
SAMPLE_RATE = 32_000
CHUNK_DURATION_S = 3.0
NUM_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_S)  # 96 000


def freeze_input_shapes(model: onnx.ModelProto) -> int:
    """Replace symbolic / dynamic input dimensions with concrete values.

    BirdNET+ V3.0 declares its input as ``[batch, samples]`` where both
    are symbolic.  Downstream nodes derive shapes such as
    ``((samples//512)) + 1`` which tract-onnx cannot parse because its
    TDim parser doesn't support Python-style integer division (``//``).

    Freezing the input to ``[1, 96000]`` and running ONNX shape inference
    propagates concrete integers through the entire graph, eliminating
    all symbolic expressions.

    Returns the number of input dimensions that were changed.
    """
    graph = model.graph
    changed = 0

    target_shapes = {
        # Map input name → desired concrete shape.
        # BirdNET+ V3.0 has a single input; handle any name.
    }
    for inp in graph.input:
        # Only patch float inputs (skip any non-tensor inputs).
        if inp.type.tensor_type.elem_type not in (
            TensorProto.FLOAT,
            TensorProto.FLOAT16,
            TensorProto.DOUBLE,
        ):
            continue
        target_shapes[inp.name] = [BATCH_SIZE, NUM_SAMPLES]

    # Collect symbolic variable names → concrete values so we can evaluate
    # leftover expressions like ``((samples//512)) + 1`` after shape inference.
    sym_variables: dict[str, int] = {}

    for inp in graph.input:
        if inp.name not in target_shapes:
            continue
        shape = target_shapes[inp.name]
        dims = inp.type.tensor_type.shape.dim
        old_dims = []
        for i, d in enumerate(dims):
            if d.dim_param:
                old_dims.append(d.dim_param)
                sym_variables[d.dim_param] = shape[i]
            else:
                old_dims.append(str(d.dim_value))

        # Clear existing dims and set concrete values.
        while len(dims) > 0:
            dims.pop()
        for val in shape:
            dim = dims.add()
            dim.dim_value = val

        print(f"  Froze input '{inp.name}': [{', '.join(old_dims)}] → {shape}")
        changed += len(shape)

    if sym_variables:
        print(f"  Symbolic variable map: {sym_variables}")

    if changed == 0:
        print("  WARNING: no input dimensions were frozen")
        return changed

    # Run shape inference to propagate concrete shapes through the graph.
    print("  Running ONNX shape inference to propagate concrete dimensions ...")
    try:
        model_inferred = shape_inference.infer_shapes(model, check_type=True)
        # Copy the inferred graph back into the model in-place.
        model.graph.CopyFrom(model_inferred.graph)
        print("  Shape inference complete")
    except Exception as e:
        print(f"  WARNING: shape inference failed ({e}), dimensions may still be symbolic")

    # Even after shape inference, some intermediate value_info entries may
    # retain symbolic dim_param values.  Clear them — either by evaluating
    # the expression against our variable map, or by stripping the symbolic
    # string entirely so tract-onnx doesn't choke on ``//``.
    cleared = _clear_residual_symbolic_dims(model, sym_variables)
    if cleared:
        print(f"  Cleared {cleared} residual symbolic dimension(s)")

    return changed


def _eval_symbolic_dim(expr: str, variables: dict[str, int]) -> int | None:
    """Try to evaluate a symbolic dimension expression to a concrete integer.

    Handles Python-style ``//`` (floor division) and standard arithmetic.
    Returns ``None`` if evaluation fails.

    >>> _eval_symbolic_dim("((samples//512)) + 1", {"samples": 96000})
    188
    """
    import re

    # Replace known variables with their concrete values.
    resolved = expr
    for var, val in variables.items():
        resolved = re.sub(rf'\b{re.escape(var)}\b', str(val), resolved)

    # Only allow digits, arithmetic operators, parentheses, and whitespace
    # to prevent arbitrary code execution.
    if re.search(r'[^0-9+\-*/() \t]', resolved):
        return None

    try:
        result = eval(resolved)  # noqa: S307 — safe: only digit/op chars
        if isinstance(result, (int, float)) and float(result).is_integer():
            return int(result)
    except Exception:
        pass
    return None


def _clear_residual_symbolic_dims(
    model: onnx.ModelProto,
    variables: dict[str, int] | None = None,
) -> int:
    """Remove leftover dim_param strings once dim_value is set.

    When *variables* is provided, also attempts to **evaluate** symbolic
    expressions (e.g. ``((samples//512)) + 1``) and replace them with
    their concrete value.  This handles the case where
    ``onnx.shape_inference`` could not propagate a concrete dim_value
    through certain operations.
    """
    if variables is None:
        variables = {}

    count = 0
    for vi in list(model.graph.value_info) + list(model.graph.output):
        if not vi.type.HasField("tensor_type"):
            continue
        for d in vi.type.tensor_type.shape.dim:
            if not d.dim_param:
                continue
            # Case 1: shape inference already set a concrete value.
            if d.dim_value > 0:
                d.ClearField("dim_param")
                count += 1
                continue
            # Case 2: try to evaluate the symbolic expression.
            concrete = _eval_symbolic_dim(d.dim_param, variables)
            if concrete is not None:
                print(f"    Evaluated '{d.dim_param}' → {concrete}")
                d.dim_value = concrete
                d.ClearField("dim_param")
                count += 1
            else:
                # Case 3: cannot evaluate — clear anyway so tract does
                # not choke.  The dimension becomes fully dynamic (0).
                print(f"    WARNING: clearing un-evaluable dim_param '{d.dim_param}'")
                d.ClearField("dim_param")
                count += 1
    return count


def propagate_constants(model: onnx.ModelProto) -> dict[str, "np.ndarray"]:
    """Mini constant-propagation pass over the frozen graph.

    After ``freeze_input_shapes`` + shape inference every tensor has a
    concrete shape.  Many ops that *look* dynamic (Shape, Gather, Cast,
    arithmetic) now have fully static inputs and can be evaluated at
    patch time.

    Returns a dict mapping tensor name → numpy array for every value
    that could be resolved statically.
    """
    import numpy as np

    graph = model.graph

    known: dict[str, np.ndarray] = {}

    # Seed with initializers.
    for init in graph.initializer:
        try:
            known[init.name] = numpy_helper.to_array(init)
        except Exception:
            pass

    # Seed with Constant node outputs.
    for node in graph.node:
        if node.op_type == "Constant" and len(node.output) == 1:
            for attr in node.attribute:
                if attr.name == "value" and attr.t is not None:
                    try:
                        known[node.output[0]] = numpy_helper.to_array(attr.t)
                    except Exception:
                        pass

    # Build shape map from value_info / inputs / outputs.
    shape_map: dict[str, list[int]] = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        if not vi.type.HasField("tensor_type"):
            continue
        shape = vi.type.tensor_type.shape
        dims = []
        all_concrete = True
        for d in shape.dim:
            if d.dim_value > 0:
                dims.append(int(d.dim_value))
            else:
                all_concrete = False
                break
        if all_concrete and dims:
            shape_map[vi.name] = dims

    # Multiple passes to resolve chains (Shape→Gather→Cast→Add→…).
    for _pass in range(5):
        progress = False
        for node in graph.node:
            # Skip if output already known.
            if len(node.output) == 0:
                continue
            out_name = node.output[0]
            if out_name in known:
                continue

            try:
                result = _eval_node(node, known, shape_map)
            except Exception:
                result = None

            if result is not None:
                known[out_name] = result
                progress = True

        if not progress:
            break

    return known


def _eval_node(
    node: "onnx.NodeProto",
    known: dict[str, "np.ndarray"],
    shape_map: dict[str, list[int]],
) -> "np.ndarray | None":
    """Try to statically evaluate a single node."""
    import numpy as np

    op = node.op_type

    def _inp(idx: int) -> "np.ndarray | None":
        if idx < len(node.input) and node.input[idx] in known:
            return known[node.input[idx]]
        return None

    if op == "Shape":
        name = node.input[0] if len(node.input) > 0 else ""
        if name in shape_map:
            return np.array(shape_map[name], dtype=np.int64)
        return None

    if op == "Gather":
        data = _inp(0)
        indices = _inp(1)
        if data is None or indices is None:
            return None
        axis = 0
        for attr in node.attribute:
            if attr.name == "axis":
                axis = int(attr.i)
        return np.take(data, indices, axis=axis)

    if op == "Cast":
        x = _inp(0)
        if x is None:
            return None
        to = TensorProto.FLOAT
        for attr in node.attribute:
            if attr.name == "to":
                to = int(attr.i)
        dtype_map = {
            TensorProto.FLOAT: np.float32,
            TensorProto.DOUBLE: np.float64,
            TensorProto.INT32: np.int32,
            TensorProto.INT64: np.int64,
            TensorProto.INT16: np.int16,
            TensorProto.BOOL: np.bool_,
        }
        return x.astype(dtype_map.get(to, np.float32))

    if op == "Unsqueeze":
        x = _inp(0)
        axes = _inp(1)
        if x is None:
            return None
        if axes is not None:
            for ax in sorted(axes.flatten()):
                x = np.expand_dims(x, axis=int(ax))
            return x
        # axes from attribute (opset < 13)
        for attr in node.attribute:
            if attr.name == "axes":
                for ax in sorted(attr.ints):
                    x = np.expand_dims(x, axis=int(ax))
                return x
        return None

    if op == "Squeeze":
        x = _inp(0)
        if x is None:
            return None
        axes = _inp(1)
        if axes is not None:
            for ax in sorted(axes.flatten(), reverse=True):
                x = np.squeeze(x, axis=int(ax))
            return x
        return np.squeeze(x)

    if op == "Reshape":
        x = _inp(0)
        shape = _inp(1)
        if x is None or shape is None:
            return None
        return x.reshape(shape.flatten().astype(int).tolist())

    if op in ("Add", "Sub", "Mul", "Div"):
        a, b = _inp(0), _inp(1)
        if a is None or b is None:
            return None
        ops = {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply, "Div": np.divide}
        return ops[op](a, b)

    if op == "Floor":
        x = _inp(0)
        return np.floor(x) if x is not None else None

    if op == "Ceil":
        x = _inp(0)
        return np.ceil(x) if x is not None else None

    if op == "Concat":
        axis = 0
        for attr in node.attribute:
            if attr.name == "axis":
                axis = int(attr.i)
        parts = [_inp(i) for i in range(len(node.input))]
        if any(p is None for p in parts):
            return None
        return np.concatenate(parts, axis=axis)

    if op == "Range":
        start, limit, delta = _inp(0), _inp(1), _inp(2)
        if start is None or limit is None or delta is None:
            return None
        return np.arange(
            start.flat[0], limit.flat[0], delta.flat[0],
            dtype=type(start.flat[0]) if isinstance(start.flat[0], (np.integer,)) else np.float32,
        )

    if op == "ConstantOfShape":
        shape = _inp(0)
        if shape is None:
            return None
        val = np.float32(0)
        for attr in node.attribute:
            if attr.name == "value" and attr.t is not None:
                val = numpy_helper.to_array(attr.t).flat[0]
        return np.full(shape.flatten().astype(int).tolist(), val)

    if op == "Slice":
        data = _inp(0)
        starts = _inp(1)
        ends = _inp(2)
        if data is None or starts is None or ends is None:
            return None
        axes = _inp(3)
        steps = _inp(4)
        # Simple single-axis case
        ndim = data.ndim
        slices = [slice(None)] * ndim
        ax_list = axes.flatten().tolist() if axes is not None else list(range(len(starts.flatten())))
        st_list = starts.flatten().tolist()
        en_list = ends.flatten().tolist()
        sp_list = steps.flatten().tolist() if steps is not None else [1] * len(ax_list)
        for a, s, e, p in zip(ax_list, st_list, en_list, sp_list):
            slices[int(a)] = slice(int(s), int(e), int(p))
        return data[tuple(slices)]

    return None


def fix_range_scalar_inputs(model: onnx.ModelProto) -> int:
    """Ensure all Range node inputs are true scalars (shape ``[]``).

    After shape inference freezes symbolic dimensions, some tensors
    that feed into ``Range`` nodes may have shape ``[1]`` instead of
    ``[]``.  The ONNX Range op spec requires 0-d scalar inputs and
    onnxruntime enforces this strictly.

    Strategy:
      * Initializers with shape ``[1]`` → squeezed in-place.
      * Constant-op outputs with shape ``[1]`` → squeezed in-place.
      * **Compute-node outputs** (Shape→Gather, etc.) → a ``Reshape``
        node is inserted to convert ``[1]`` → ``[]``.

    Returns the number of fixes applied.
    """
    import numpy as np

    graph = model.graph

    # Build lookups.
    init_by_name = {init.name: init for init in graph.initializer}
    const_by_output: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        if node.op_type == "Constant" and len(node.output) == 1:
            const_by_output[node.output[0]] = node

    fixed = 0
    reshape_nodes_to_add: list[onnx.NodeProto] = []

    # We may need a single "empty shape" initializer for Reshape → scalar.
    scalar_shape_init_name = "_scalar_shape_for_range"
    scalar_shape_added = False

    def _ensure_scalar_shape_init():
        nonlocal scalar_shape_added
        if not scalar_shape_added:
            graph.initializer.append(
                numpy_helper.from_array(
                    np.array([], dtype=np.int64), name=scalar_shape_init_name
                )
            )
            scalar_shape_added = True

    for node in list(graph.node):
        if node.op_type != "Range":
            continue

        for i, inp_name in enumerate(node.input):
            if not inp_name:
                continue

            squeezed = False

            # ── Try initializer ───────────────────────────────────
            if inp_name in init_by_name:
                init = init_by_name[inp_name]
                arr = numpy_helper.to_array(init)
                if arr.shape == (1,):
                    scalar = arr.flatten()[0]
                    new_init = numpy_helper.from_array(
                        np.array(scalar), name=inp_name
                    )
                    init.CopyFrom(new_init)
                    print(f"  Squeezed initializer '{inp_name}': [1] → scalar ({scalar})")
                    squeezed = True
                    fixed += 1
                # Shape () is already scalar – nothing to do.
                continue

            # ── Try Constant op ────────────────────────────────────
            if inp_name in const_by_output:
                cnode = const_by_output[inp_name]
                for attr in cnode.attribute:
                    if attr.name == "value" and attr.t is not None:
                        arr = numpy_helper.to_array(attr.t)
                        if arr.shape == (1,):
                            scalar = arr.flatten()[0]
                            new_tensor = numpy_helper.from_array(
                                np.array(scalar), name=attr.t.name or ""
                            )
                            attr.t.CopyFrom(new_tensor)
                            print(
                                f"  Squeezed Constant '{inp_name}': "
                                f"[1] → scalar ({scalar})"
                            )
                            squeezed = True
                            fixed += 1
                continue

            # ── Compute-node output: insert Reshape → scalar ──────
            # We don't know the shape statically for certain, but if
            # onnxruntime is complaining about it, it must be [1].
            # Insert Reshape(x, []) which converts [1] → scalar.
            _ensure_scalar_shape_init()
            new_name = f"{inp_name}_scalar_{node.name}_{i}"
            reshape_node = helper.make_node(
                "Reshape",
                inputs=[inp_name, scalar_shape_init_name],
                outputs=[new_name],
                name=f"reshape_to_scalar_{node.name}_inp{i}",
            )
            node.input[i] = new_name
            reshape_nodes_to_add.append(reshape_node)
            print(
                f"  Inserted Reshape for compute output '{inp_name}' "
                f"→ '{new_name}' (scalar) feeding {node.name} input {i}"
            )
            fixed += 1

    # Append the new Reshape nodes to the graph.
    for n in reshape_nodes_to_add:
        graph.node.append(n)

    # Fix value_info shapes for Range inputs: set them to 0-d.
    range_input_names: set[str] = set()
    for node in graph.node:
        if node.op_type == "Range":
            for inp_name in node.input:
                range_input_names.add(inp_name)
    for vi in graph.value_info:
        if vi.name in range_input_names:
            tt = vi.type.tensor_type
            if tt.HasField("shape") and len(tt.shape.dim) == 1:
                while len(tt.shape.dim) > 0:
                    tt.shape.dim.pop()

    return fixed


def fold_static_range_nodes(model: onnx.ModelProto, known: dict[str, "np.ndarray"] | None = None) -> int:
    """Constant-fold Range nodes whose inputs are all static.

    tract-onnx fails on ``Range`` nodes that produce I64 tensors:

        ``TDim == outputs[0].datum_type: Impossible to unify TDim with I64``

    After ``freeze_input_shapes`` + ``propagate_constants``, most Range
    inputs have been resolved to concrete values.  When all three inputs
    (start, limit, delta) are known, we precompute the output tensor and
    replace the Range node with a Constant.

    Parameters
    ----------
    model : onnx.ModelProto
        The model to patch in-place.
    known : dict, optional
        Pre-computed constant values from ``propagate_constants``.
        If ``None``, only initializers and Constant-node outputs are
        checked (less coverage).

    Returns the number of Range nodes folded.
    """
    import numpy as np

    graph = model.graph

    if known is None:
        known = {}

    # Also include initializers and Constant-node values for completeness.
    all_known: dict[str, np.ndarray] = dict(known)
    for init in graph.initializer:
        if init.name not in all_known:
            try:
                all_known[init.name] = numpy_helper.to_array(init)
            except Exception:
                pass
    for node in graph.node:
        if node.op_type == "Constant" and len(node.output) == 1:
            out = node.output[0]
            if out not in all_known:
                for attr in node.attribute:
                    if attr.name == "value" and attr.t is not None:
                        try:
                            all_known[out] = numpy_helper.to_array(attr.t)
                        except Exception:
                            pass

    def _get_scalar(name: str):
        """Return a scalar value for a tensor name, or None."""
        if name in all_known:
            arr = all_known[name]
            if arr.size == 1:
                return arr.flat[0]
        return None

    nodes_to_remove: list[onnx.NodeProto] = []
    nodes_to_add: list[onnx.NodeProto] = []
    folded = 0

    for node in list(graph.node):
        if node.op_type != "Range":
            continue
        if len(node.input) < 3:
            continue

        start = _get_scalar(node.input[0])
        limit = _get_scalar(node.input[1])
        delta = _get_scalar(node.input[2])

        if start is None or limit is None or delta is None:
            print(
                f"  WARNING: Range node '{node.name}' has non-static inputs "
                f"(start={start}, limit={limit}, delta={delta}) — cannot fold"
            )
            continue

        # Determine dtype from start (ONNX Range preserves input dtype).
        if isinstance(start, (np.integer, int)):
            dtype = np.int64
        else:
            dtype = np.float32

        result = np.arange(start, limit, delta, dtype=dtype)
        print(
            f"  Folding Range '{node.name}': "
            f"arange({start}, {limit}, {delta}) → Constant [{len(result)}] ({dtype.__name__})"
        )

        # Create a Constant node that produces the precomputed tensor.
        out_name = node.output[0]
        tensor = numpy_helper.from_array(result, name=out_name)
        const_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=[out_name],
            name=f"{node.name}_folded",
            value=tensor,
        )

        nodes_to_remove.append(node)
        nodes_to_add.append(const_node)
        folded += 1

    # Remove old Range nodes and add new Constant nodes.
    for n in nodes_to_remove:
        graph.node.remove(n)
    for n in nodes_to_add:
        graph.node.append(n)

    return folded


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
        print("WARNING: No Resize nodes with pytorch_half_pixel found — continuing anyway")
    else:
        print(f"Patched {patched} Resize node(s)")

    dft_fixed = patch_onesided_dft_nodes(model)
    if dft_fixed:
        print(f"Patched {dft_fixed} one-sided DFT node(s)")

    frozen = freeze_input_shapes(model)
    print(f"Froze {frozen} input dimension(s)")

    # Second DFT pass — after freeze, signal shapes are concrete so
    # nodes whose dft_length was not statically known can now be patched.
    dft_fixed_2 = patch_onesided_dft_nodes(model)
    if dft_fixed_2:
        print(f"Second-pass: patched {dft_fixed_2} additional one-sided DFT node(s)")
        # Re-run shape inference to propagate the corrected shapes.
        try:
            model_inferred = shape_inference.infer_shapes(model, check_type=True)
            model.graph.CopyFrom(model_inferred.graph)
            print("  Updated shape inference after second DFT pass")
        except Exception as e:
            print(f"  WARNING: post-DFT shape inference failed ({e})")

    range_fixed = fix_range_scalar_inputs(model)
    if range_fixed:
        print(f"Fixed {range_fixed} Range-node scalar input(s)")

    # Propagate constants through Shape/Gather/Cast/arithmetic chains
    # so that Range inputs (often derived from shape computations) become
    # known scalars.
    print("Running constant propagation ...")
    known = propagate_constants(model)
    print(f"  Resolved {len(known)} static tensors")

    range_folded = fold_static_range_nodes(model, known=known)
    if range_folded:
        print(f"Constant-folded {range_folded} Range node(s)")

    # Save
    print(f"Saving patched model to {args.output} ...")
    onnx.save(model, args.output)
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"  Saved: {args.output} ({size_mb:.1f} MB)")

    # Cleanup
    if not args.keep_download and os.path.exists(raw_path):
        os.remove(raw_path)
        print("  Removed original download")

    # ── Validate the patched model ────────────────────────────────────
    # Run a quick inference with dummy audio to confirm the model loads
    # and produces output with the expected shape.
    print("Validating patched model with onnxruntime ...")
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        inp_shape = sess.get_inputs()[0].shape
        print(f"  Input: name={inp_name!r}, shape={inp_shape}")

        # Feed 3 s of silence at 32 kHz.
        dummy = np.zeros((BATCH_SIZE, NUM_SAMPLES), dtype=np.float32)
        outputs = sess.run(None, {inp_name: dummy})
        out_shape = outputs[0].shape
        print(f"  Output shape: {out_shape}")
        if len(out_shape) < 2 or out_shape[-1] < 100:
            print(
                f"WARNING: output shape {out_shape} looks unexpected for a "
                f"species classifier — double-check the model",
                file=sys.stderr,
            )
        else:
            print(f"  Validation passed: {out_shape[-1]} class scores produced")
    except ImportError:
        print("  onnxruntime / numpy not installed — skipping runtime validation")
    except Exception as e:
        print(f"ERROR: model validation failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()
