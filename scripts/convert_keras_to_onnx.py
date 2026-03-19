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
                super().build(input_shape)
            def call(self, inputs):
                import tensorflow as tf
                # (batch, D, 1) * (D, E) → broadcast → (batch, D, E)
                expanded = tf.expand_dims(inputs, axis=-1) * self.kernel
                # Use -1 for the batch dim and the *static* product for the
                # feature dim.  This lets Keras infer (None, D*E) and makes
                # tf2onnx emit a constant shape [-1, D*E] in the ONNX
                # Reshape node — required by tract which rejects variable
                # shape inputs.
                d = inputs.shape[-1]
                return tf.reshape(expanded, [-1, d * self.embeddings])
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

    # ── Post-process: freeze Reshape shape inputs ────────────────────
    # tf2onnx emits the Reshape shape as a computed subgraph
    # (Shape → Gather → Unsqueeze → Concat) even when we use [-1, N]
    # in Python.  tract-onnx rejects this ("shape input is variable").
    # Fix by replacing each Reshape's shape input with a constant
    # initializer inferred from onnxruntime's shape inference.
    import onnx
    from onnx import numpy_helper, TensorProto

    model_onnx = onnx.load(onnx_path)
    graph = model_onnx.graph

    # Build a lookup of initialiser names for quick membership test.
    init_names = {init.name for init in graph.initializer}

    patched = 0
    for node in graph.node:
        if node.op_type != "Reshape" or len(node.input) < 2:
            continue
        shape_input = node.input[1]
        if shape_input in init_names:
            continue  # already a constant — nothing to do

        # Use onnx shape inference to determine the Reshape output shape.
        from onnx import shape_inference
        inferred = shape_inference.infer_shapes(model_onnx)
        # Find the shape of this Reshape node's output.
        output_name = node.output[0]
        target_shape = None
        for vi in inferred.graph.value_info:
            if vi.name == output_name:
                dims = vi.type.tensor_type.shape.dim
                target_shape = []
                for d in dims:
                    if d.dim_value > 0:
                        target_shape.append(d.dim_value)
                    else:
                        target_shape.append(-1)  # dynamic batch → -1
                break
        # Also check graph outputs in case it's a final node.
        if target_shape is None:
            for vi in inferred.graph.output:
                if vi.name == output_name:
                    dims = vi.type.tensor_type.shape.dim
                    target_shape = []
                    for d in dims:
                        if d.dim_value > 0:
                            target_shape.append(d.dim_value)
                        else:
                            target_shape.append(-1)
                    break

        if target_shape is None:
            print(f"  WARNING: could not infer shape for Reshape "
                  f"node '{node.name}', skipping")
            continue

        # Create a constant initializer with the target shape.
        const_name = f"{node.name}_shape_const"
        shape_array = np.array(target_shape, dtype=np.int64)
        tensor = numpy_helper.from_array(shape_array, name=const_name)
        graph.initializer.append(tensor)
        init_names.add(const_name)

        # Point the Reshape node at the constant.
        node.input[1] = const_name
        patched += 1
        print(f"  Froze Reshape '{node.name}' shape to {target_shape}")

    if patched:
        onnx.save(model_onnx, onnx_path)
        print(f"  Patched {patched} Reshape node(s) with constant shapes")

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  Written {size_mb:.1f} MB ONNX metadata model")

    # Validate with onnxruntime.
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        inp_name = sess.get_inputs()[0].name
        onnx_pred = sess.run(None, {inp_name: test_input})[0]
        max_diff = np.max(np.abs(keras_pred - onnx_pred))
        print(f"  ONNX validation: max diff vs Keras = {max_diff:.2e}")
    except ImportError:
        print("  (onnxruntime not installed, skipping ONNX validation)")

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
