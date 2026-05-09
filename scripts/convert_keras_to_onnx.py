#!/usr/bin/env python3
"""Convert a BirdNET-style Keras model to a classifier-only ONNX model.

BirdNET V2.4's Keras model contains MelSpecLayerSimple layers that use
tf.signal.stft (RFFT), which cannot be represented in standard ONNX.  This
script splits the model at the concatenate layer (after the two mel
spectrogram layers) and converts only the CNN classifier to ONNX.

The mel-spectrogram preprocessing is handled at runtime in Rust (mel.rs).

Usage:
    python3 convert_keras_to_onnx.py <model_dir>

where <model_dir> contains audio-model.h5 (and optionally the custom layer
code MelSpecLayerSimple.py).

The output ONNX file is written alongside the .h5 file.
"""

import argparse
import os
import sys

def find_concatenate_layer(model):
    """Find the concatenate layer that joins the two mel-spec channels."""
    for layer in model.layers:
        if 'concatenate' in layer.name.lower():
            return layer
    raise RuntimeError(
        "Cannot find a 'concatenate' layer in the model.  "
        "This script expects BirdNET-style models with two MelSpecLayerSimple "
        "layers feeding into a concatenate layer."
    )


def convert(model_dir: str, output_name: str = "audio-model.onnx") -> str:
    """Convert audio-model.h5 → classifier-only ONNX.

    Returns the path to the generated ONNX file.
    """
    h5_path = os.path.join(model_dir, "audio-model.h5")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Keras model not found: {h5_path}")

    # Import heavy deps only after checking the file exists.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # suppress TF noise
    try:
        import tf_keras as keras          # TF 2.16+
    except ImportError:
        from tensorflow import keras      # TF ≤ 2.15
    import numpy as np

    # Try loading the custom MelSpecLayerSimple if present.
    custom_layer_path = os.path.join(model_dir, "MelSpecLayerSimple.py")
    custom_objects = {}
    if os.path.exists(custom_layer_path):
        sys.path.insert(0, model_dir)
        from MelSpecLayerSimple import MelSpecLayerSimple
        custom_objects["MelSpecLayerSimple"] = MelSpecLayerSimple

    print(f"Loading Keras model from {h5_path} …")
    model = keras.models.load_model(h5_path, custom_objects=custom_objects, compile=False)
    print(f"  {model.name}: {len(model.layers)} layers, input={model.input_shape}")

    # Find the concatenate layer (boundary between mel-spec and classifier).
    concat_layer = find_concatenate_layer(model)
    concat_output = concat_layer.output
    print(f"  Split point: '{concat_layer.name}' → shape {concat_output.shape}")

    # ── Extract classifier sub-model using Keras graph tracing ────────
    # BirdNET's classifier has internal branching (inception-style blocks
    # with POOL_*_CONCAT merge layers), so a naive sequential replay of
    # layers doesn't work.  Instead we let Keras trace the computational
    # graph from the concatenate output tensor to the model output.
    #
    # Step 1: Build a "raw" sub-model.  Keras.Model() traces the graph
    #         backward from `outputs` to find every layer between the
    #         given `inputs` (an intermediate tensor) and `outputs`.
    # Step 2: Wrap with a proper keras.Input so tf2onnx gets a clean
    #         single-input model.
    print("  Extracting classifier sub-model via graph tracing …")
    classifier_raw = keras.Model(
        inputs=concat_output,
        outputs=model.output,
        name="classifier_raw",
    )
    print(f"  Raw sub-model: {len(classifier_raw.layers)} layers")

    inp = keras.Input(
        shape=tuple(concat_output.shape[1:]),
        name="mel_spectrogram",
    )
    out = classifier_raw(inp)
    classifier = keras.Model(inputs=inp, outputs=out, name="classifier")
    print(f"  Classifier: {len(classifier.layers)} layers, "
          f"input={classifier.input_shape} → output={classifier.output_shape}")

    # Quick sanity check — run inference through both paths.
    mel_extractor = keras.Model(inputs=model.input, outputs=concat_output)
    rng = np.random.RandomState(42)
    test_audio = (rng.randn(1, int(model.input_shape[1])) * 0.1).astype(np.float32)
    mel_from_keras = mel_extractor.predict(test_audio, verbose=0)
    full_pred = model.predict(test_audio, verbose=0)
    cls_pred = classifier.predict(mel_from_keras, verbose=0)
    max_diff = np.max(np.abs(full_pred - cls_pred))
    print(f"  Validation: full model vs classifier sub-model max diff = {max_diff:.2e}")
    if max_diff > 1e-4:
        print(f"  WARNING: large difference {max_diff}, conversion may be lossy")

    # Convert to ONNX via tf2onnx.
    import tf2onnx
    onnx_path = os.path.join(model_dir, output_name)
    print(f"Converting to ONNX: {onnx_path} …")
    model_proto, _ = tf2onnx.convert.from_keras(
        classifier, output_path=onnx_path
    )
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  Written {size_mb:.1f} MB ONNX model")

    # Validate with onnxruntime.
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        inp_name = sess.get_inputs()[0].name
        onnx_pred = sess.run(None, {inp_name: mel_from_keras})[0]
        ort_diff = np.max(np.abs(full_pred - onnx_pred))
        print(f"  ONNX validation: max diff vs Keras = {ort_diff:.2e}")
    except ImportError:
        print("  (onnxruntime not installed, skipping ONNX validation)")

    print("Done.")
    return onnx_path


def _promote_reshape_shapes_to_initializers(onnx_path: str, sample_input) -> None:
    """Ensure every Reshape node's shape input is a graph initializer.

    tract-onnx only recognises graph initializers as "constant" — Constant
    op outputs and compute-subgraph results are rejected, even when they
    are statically deterministic.

    This function handles three cases:

      1. Shape input is already an initializer → skip (OK).
      2. Shape input is produced by a ``Constant`` op → extract the tensor
         value and create an initializer.
      3. Shape input is produced by a compute subgraph → run ORT inference
         to capture the concrete value and create an initializer.

    After processing, all Reshape nodes point to initializers.
    """
    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper, helper
    import numpy as np

    model = onnx.load(onnx_path)
    graph = model.graph

    init_names = {i.name for i in graph.initializer}

    # Build Constant-op value map.
    const_values: dict[str, "np.ndarray"] = {}
    for node in graph.node:
        if node.op_type == "Constant" and len(node.output) == 1:
            for attr in node.attribute:
                if attr.name == "value" and attr.t is not None:
                    try:
                        const_values[node.output[0]] = numpy_helper.to_array(attr.t)
                    except Exception:
                        pass

    # Collect Reshape shape inputs that need promotion.
    # Map: original_shape_name → list of (node, input_index)
    to_promote: dict[str, list[tuple]] = {}
    for node in graph.node:
        if node.op_type == "Reshape" and len(node.input) >= 2:
            shape_inp = node.input[1]
            if shape_inp and shape_inp not in init_names:
                to_promote.setdefault(shape_inp, []).append((node, 1))

    if not to_promote:
        print("  All Reshape shape inputs are already initializers ✓")
        return

    print(f"  Promoting {len(to_promote)} Reshape shape input(s) to initializers …")
    modified = False
    counter = 0

    # ── Case 2: Constant-op outputs ──────────────────────────────────
    for shape_name in list(to_promote):
        if shape_name in const_values:
            val = const_values[shape_name].flatten().astype(np.int64)
            new_name = f"_reshape_shape_promoted_{counter}"
            counter += 1
            tensor = numpy_helper.from_array(val, name=new_name)
            graph.initializer.append(tensor)
            init_names.add(new_name)
            for node, idx in to_promote[shape_name]:
                node.input[idx] = new_name
            print(f"    Extracted Constant '{shape_name}' → "
                  f"initializer '{new_name}' = {val.tolist()}")
            del to_promote[shape_name]
            modified = True

    # ── Case 3: compute-subgraph outputs → capture via ORT ───────────
    if to_promote:
        # Add remaining shape inputs as temporary outputs.
        orig_output_count = len(graph.output)
        for shape_name in to_promote:
            graph.output.append(
                helper.make_tensor_value_info(
                    shape_name, onnx.TensorProto.INT64, None
                )
            )
        tmp_path = onnx_path + ".promote_tmp"
        onnx.save(model, tmp_path)

        try:
            sess = ort.InferenceSession(tmp_path)
            inp_name = sess.get_inputs()[0].name
            for shape_name in list(to_promote):
                val = sess.run([shape_name], {inp_name: sample_input})[0]
                val = val.flatten().astype(np.int64)
                new_name = f"_reshape_shape_promoted_{counter}"
                counter += 1
                tensor = numpy_helper.from_array(val, name=new_name)
                graph.initializer.append(tensor)
                init_names.add(new_name)
                for node, idx in to_promote[shape_name]:
                    node.input[idx] = new_name
                print(f"    Captured '{shape_name}' via inference → "
                      f"initializer '{new_name}' = {val.tolist()}")
                modified = True
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # Remove the temporary outputs we added.
        while len(graph.output) > orig_output_count:
            graph.output.pop()

    if modified:
        onnx.save(model, onnx_path)
        print(f"  Promoted all Reshape shape inputs to initializers ✓")


def _freeze_reshape_shape(onnx_path: str, reshape_node, shape_input_name: str,
                          sample_input) -> None:
    """Replace a Reshape node's dynamic shape input with a constant.

    Adds the shape tensor as a temporary model output, runs inference
    with onnxruntime to capture its concrete value, then replaces it
    with a constant initializer in the graph.
    """
    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper, helper
    import numpy as np

    model = onnx.load(onnx_path)
    graph = model.graph

    # Add the shape tensor as a temporary output so ORT will expose it.
    graph.output.append(
        helper.make_tensor_value_info(shape_input_name, onnx.TensorProto.INT64, None)
    )
    tmp_path = onnx_path + ".freeze_tmp"
    onnx.save(model, tmp_path)

    try:
        sess = ort.InferenceSession(tmp_path)
        inp_name = sess.get_inputs()[0].name
        shape_val = sess.run([shape_input_name], {inp_name: sample_input})[0]
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    shape_val = shape_val.flatten().astype(np.int64)
    print(f"    Captured shape value: {shape_val.tolist()}")

    # Reload original model (without the temporary output).
    model = onnx.load(onnx_path)
    graph = model.graph

    # Create a constant initializer with the captured shape.
    const_name = f"{reshape_node.name}_frozen_shape"
    tensor = numpy_helper.from_array(shape_val, name=const_name)
    graph.initializer.append(tensor)

    # Point every Reshape that references this shape input at the constant.
    for node in graph.node:
        if node.op_type == "Reshape":
            for i, inp in enumerate(node.input):
                if inp == shape_input_name:
                    node.input[i] = const_name

    onnx.save(model, onnx_path)
    print(f"    Froze as constant initializer '{const_name}' = {shape_val.tolist()}")


def convert_meta_model(model_dir: str, output_name: str = "meta-model.onnx") -> str:
    """Convert meta-model.h5 → ONNX.

    The metadata model uses a custom ``MDataLayer`` Keras layer (analogous
    to ``MelSpecLayerSimple`` in the audio model).  We load the custom
    layer definition from the model directory, then convert the full model
    to ONNX – no sub-model splitting is needed since ``MDataLayer`` uses
    only standard TensorFlow ops.

    Returns the path to the generated ONNX file.
    """
    h5_path = os.path.join(model_dir, "meta-model.h5")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Metadata Keras model not found: {h5_path}")

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tf_keras as keras
    except ImportError:
        from tensorflow import keras
    import numpy as np

    # Load custom layer definitions that may be referenced by the model.
    custom_objects = {}
    for layer_name in ["MDataLayer", "MelSpecLayerSimple"]:
        layer_path = os.path.join(model_dir, f"{layer_name}.py")
        if os.path.exists(layer_path):
            if model_dir not in sys.path:
                sys.path.insert(0, model_dir)
            mod = __import__(layer_name)
            custom_objects[layer_name] = getattr(mod, layer_name)
            print(f"  Loaded custom layer: {layer_name}")

    # BirdNET's MDataLayer is a per-feature embedding layer.
    # The .py file is NOT shipped in the Keras zip, so define a stub.
    # Config has {'embeddings': 48} — a weight matrix of shape
    # (input_dim, embeddings) is created.  For each input feature the
    # corresponding embedding row is scaled, producing output of shape
    # (batch, input_dim * embeddings).  E.g. (batch, 3) → (batch, 144).
    if "MDataLayer" not in custom_objects:
        class _MDataLayer(keras.layers.Layer):
            def __init__(self, embeddings=48, **kwargs):
                super().__init__(**kwargs)
                self.embeddings = embeddings
            def build(self, input_shape):
                self.kernel = self.add_weight(
                    name='kernel',
                    shape=(input_shape[-1], self.embeddings),
                )
                # Pre-build a Flatten layer to merge the last two dims.
                # Using Flatten instead of tf.reshape avoids the dynamic
                # Shape→Gather→Concat subgraph that tf2onnx emits for
                # tf.reshape(x, [-1, N]) — tract-onnx rejects Reshape
                # nodes whose shape input is not a constant initializer.
                self._flatten = keras.layers.Flatten()
                super().build(input_shape)
            def call(self, inputs):
                import tensorflow as tf
                # (batch, D, 1) * (D, E) → broadcast → (batch, D, E)
                expanded = tf.expand_dims(inputs, axis=-1) * self.kernel
                # Flatten (batch, D, E) → (batch, D*E) without explicit
                # shape constants — Flatten uses a Reshape whose shape
                # is derived from the static output spec, which tf2onnx
                # can emit as a constant initializer.
                return self._flatten(expanded)
            def compute_output_shape(self, input_shape):
                return (input_shape[0], input_shape[-1] * self.embeddings)
            def get_config(self):
                config = super().get_config()
                config['embeddings'] = self.embeddings
                return config
        custom_objects["MDataLayer"] = _MDataLayer
        print("  Using stub MDataLayer (per-feature embedding)")

    print(f"Loading metadata Keras model from {h5_path} …")
    try:
        model = keras.models.load_model(
            h5_path, custom_objects=custom_objects, compile=False
        )
    except ValueError as e:
        if "Layer count mismatch" not in str(e) and "Weight count mismatch" not in str(e):
            raise
        # Keras 3 struggles with Keras-2 h5 files that use custom layers:
        #  - "Layer count mismatch" — extra implicit InputLayer
        #  - "Weight count mismatch" — by_name loader can't match custom
        #    layer weights across the Keras 2→3 boundary.
        # Fall back to fully manual loading via h5py.
        print(f"  Working around Keras 3 compatibility issue …")
        import h5py, json

        with h5py.File(h5_path, "r") as f:
            cfg = f.attrs.get("model_config")
            if isinstance(cfg, bytes):
                cfg = cfg.decode("utf-8")
            model = keras.models.model_from_config(
                json.loads(cfg), custom_objects=custom_objects
            )
            # Force build so every layer allocates its weight tensors.
            dummy = np.zeros((1, 3), dtype=np.float32)
            model(dummy)

            # Walk the Keras-2 h5 weight groups and set weights by name.
            root = f["model_weights"] if "model_weights" in f else f
            layer_names = [
                n.decode("utf-8") if isinstance(n, bytes) else n
                for n in root.attrs["layer_names"]
            ]
            for lname in layer_names:
                g = root[lname]
                wnames = [
                    n.decode("utf-8") if isinstance(n, bytes) else n
                    for n in g.attrs.get("weight_names", [])
                ]
                if not wnames:
                    continue
                weight_values = [g[wn][()] for wn in wnames]
                for layer in model.layers:
                    if layer.name == lname:
                        layer.set_weights(weight_values)
                        print(f"    Loaded {len(weight_values)} weight(s) "
                              f"for layer '{lname}'")
                        break
    print(f"  {model.name}: {len(model.layers)} layers, "
          f"input={model.input_shape} → output={model.output_shape}")

    # Quick sanity check.
    rng = np.random.RandomState(42)
    test_input = rng.randn(1, 3).astype(np.float32)  # [lat, lon, week]
    keras_pred = model.predict(test_input, verbose=0)
    print(f"  Keras output shape: {keras_pred.shape}, "
          f"range [{keras_pred.min():.4f}, {keras_pred.max():.4f}]")

    # Convert to ONNX.
    import tf2onnx
    onnx_path = os.path.join(model_dir, output_name)
    print(f"Converting metadata model to ONNX: {onnx_path} …")
    model_proto, _ = tf2onnx.convert.from_keras(model, output_path=onnx_path)

    # ── Post-process: constant-fold via onnxruntime ──────────────────
    # tf2onnx emits the Reshape shape as a computed subgraph
    # (Shape → Gather → Unsqueeze → Concat) even when we use [-1, N]
    # in Python.  tract-onnx rejects this ("shape input is variable").
    #
    # ORT's basic graph optimizer includes constant folding, which
    # collapses these subgraphs into constant initializers.
    import onnxruntime as ort
    print("  Running onnxruntime constant-folding …")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    optimized_path = onnx_path + ".optimized"
    opts.optimized_model_filepath = optimized_path
    ort.InferenceSession(onnx_path, opts)  # triggers optimize + save
    os.replace(optimized_path, onnx_path)
    print("  Constant-folded model saved")

    # ── Verify: all Reshape shape inputs must be constant ────────────
    # tract-onnx only accepts Reshape shapes that are graph initializers.
    # ORT constant folding may produce Constant-op outputs instead, which
    # pass onnxruntime validation but fail at runtime in tract.  Use a
    # comprehensive promotion step that handles both cases.
    import onnx
    _promote_reshape_shapes_to_initializers(onnx_path, test_input)

    # Final check — must use initializer-only test (matching tract).
    model_onnx = onnx.load(onnx_path)
    init_names = {i.name for i in model_onnx.graph.initializer}
    for node in model_onnx.graph.node:
        if node.op_type == "Reshape" and len(node.input) >= 2:
            if node.input[1] not in init_names:
                raise RuntimeError(
                    f"FATAL: Reshape node '{node.name}' still has "
                    f"non-constant shape input '{node.input[1]}' — "
                    f"tract-onnx will reject this model")

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  Written {size_mb:.1f} MB ONNX metadata model")
    print(f"  All Reshape shape inputs verified as constant ✓")

    # Validate with onnxruntime.
    sess = ort.InferenceSession(onnx_path)
    inp_name = sess.get_inputs()[0].name
    onnx_pred = sess.run(None, {inp_name: test_input})[0]
    max_diff = np.max(np.abs(keras_pred - onnx_pred))
    print(f"  ONNX validation: max diff vs Keras = {max_diff:.2e}")

    print("Metadata model conversion done.")
    return onnx_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="Directory containing audio-model.h5")
    parser.add_argument(
        "-o", "--output", default="audio-model.onnx",
        help="Output ONNX filename (default: audio-model.onnx)"
    )
    parser.add_argument(
        "--meta", action="store_true",
        help="Also convert meta-model.h5 to ONNX"
    )
    parser.add_argument(
        "--meta-only", action="store_true",
        help="Convert ONLY meta-model.h5 to ONNX (skip audio model)"
    )
    parser.add_argument(
        "--meta-output", default="meta-model.onnx",
        help="Output filename for metadata ONNX model (default: meta-model.onnx)"
    )
    args = parser.parse_args()
    if not args.meta_only:
        convert(args.model_dir, args.output)
    if args.meta or args.meta_only:
        meta_h5 = os.path.join(args.model_dir, "meta-model.h5")
        if os.path.exists(meta_h5):
            convert_meta_model(args.model_dir, args.meta_output)
        else:
            print(f"Skipping metadata model: {meta_h5} not found")


if __name__ == "__main__":
    main()
