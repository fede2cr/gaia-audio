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
    import tf_keras as keras
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

    # Build classifier sub-model: concat output → final output.
    classifier_input = keras.layers.Input(
        shape=concat_output.shape[1:], name="mel_spectrogram"
    )

    # Replay the graph from concat → end.
    x = classifier_input
    found_concat = False
    for layer in model.layers:
        if layer == concat_layer:
            found_concat = True
            continue
        if not found_concat:
            continue
        x = layer(x)

    classifier = keras.Model(inputs=classifier_input, outputs=x, name="classifier")
    print(f"  Classifier sub-model: {len(classifier.layers)} layers, "
          f"input={classifier.input_shape} → output={classifier.output_shape}")

    # Quick sanity check: run inference with zeros through both models.
    test_input = np.zeros((1,) + tuple(concat_output.shape[1:]), dtype=np.float32)
    ref_out = model.layers[-1].output  # we need an extractor for the concat→end path
    mel_extractor = keras.Model(inputs=model.input, outputs=concat_output)
    # Use random audio to check end-to-end equivalence.
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
    spec = (tf2onnx.tf_loader.keras2onnx_spec(classifier),)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="Directory containing audio-model.h5")
    parser.add_argument(
        "-o", "--output", default="audio-model.onnx",
        help="Output ONNX filename (default: audio-model.onnx)"
    )
    args = parser.parse_args()
    convert(args.model_dir, args.output)


if __name__ == "__main__":
    main()
