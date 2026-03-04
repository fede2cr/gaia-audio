#!/usr/bin/env python3
"""Export BatDetect2 PyTorch model to ONNX and extract class labels.

Downloads the pre-trained PyTorch checkpoint from the batdetect2 package
(or a local path), exports it to ONNX with a fixed spectrogram input
shape, and writes the class labels to a CSV file.

Usage:
    python convert_batdetect2_to_onnx.py [--output-dir /path/to/output]

Outputs:
    <output-dir>/batdetect2.onnx   — ONNX model (~8 MB)
    <output-dir>/labels.csv        — One class name per line

The ONNX model expects input shape [1, 1, 128, T] where T is variable
(spectrogram width = time frames).  For tract compatibility we export
with a fixed T=256 (representing ~1 second at 256 kHz with default
FFT parameters).  Longer inputs will need to be tiled.
"""

import argparse
import os
import sys

import torch
import torch.onnx


def main():
    parser = argparse.ArgumentParser(description="Export BatDetect2 to ONNX")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to .pth.tar checkpoint (downloads default if omitted)",
    )
    parser.add_argument(
        "--output-dir",
        default="/convert",
        help="Directory to write batdetect2.onnx and labels.csv",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load the checkpoint ──────────────────────────────────────────────
    if args.model_path:
        model_path = args.model_path
    else:
        # Use the model bundled with the batdetect2 package
        from batdetect2.detector.parameters import DEFAULT_MODEL_PATH

        model_path = DEFAULT_MODEL_PATH

    print(f"Loading checkpoint from {model_path}")
    if not os.path.isfile(model_path):
        print(f"ERROR: Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cpu")
    net_params = torch.load(model_path, map_location=device, weights_only=False)
    params = net_params["params"]

    class_names = params["class_names"]
    print(f"Model: {params['model_name']}, classes: {len(class_names)}")
    print(f"  num_filters={params['num_filters']}, ip_height={params['ip_height']}")
    print(f"  resize_factor={params['resize_factor']}, emb_dim={params['emb_dim']}")

    # ── Instantiate model and load weights ───────────────────────────────
    from batdetect2.detector import models

    ModelClass = getattr(models, params["model_name"])
    model = ModelClass(
        params["num_filters"],
        num_classes=len(class_names),
        emb_dim=params["emb_dim"],
        ip_height=params["ip_height"],
        resize_factor=params["resize_factor"],
    )
    model.load_state_dict(net_params["state_dict"])
    model.eval()

    # ── Export to ONNX ───────────────────────────────────────────────────
    # Input: [batch, channels=1, freq_bins=128, time_frames]
    # The ip_height after resize_factor is used as freq dimension.
    ip_height = params["ip_height"]
    # Time dimension: 256 frames ≈ 1 second at default FFT params.
    # We use dynamic axes so tract can handle variable lengths.
    time_frames = 256

    dummy_input = torch.randn(1, 1, ip_height, time_frames)
    onnx_path = os.path.join(args.output_dir, "batdetect2.onnx")

    print(f"Exporting to ONNX: {onnx_path}")
    print(f"  Input shape: [1, 1, {ip_height}, {time_frames}] (dynamic time axis)")

    # BatDetect2 forward() returns a ModelOutput namedtuple with multiple
    # fields.  We need to handle the multi-output export.
    #
    # Wrap the model to return a flat tuple of tensors.
    class OnnxWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            out = self.inner(x)
            # out is ModelOutput(pred_det, pred_size, pred_class, pred_class_un_norm, features)
            return out.pred_det, out.pred_size, out.pred_class

    wrapper = OnnxWrapper(model)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        dummy_input,
        onnx_path,
        input_names=["spectrogram"],
        output_names=["detection_prob", "detection_size", "class_prob"],
        dynamic_axes={
            "spectrogram": {0: "batch", 3: "time"},
            "detection_prob": {0: "batch", 3: "time"},
            "detection_size": {0: "batch", 3: "time"},
            "class_prob": {0: "batch", 3: "time"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"  Written: {onnx_path} ({os.path.getsize(onnx_path) / 1024 / 1024:.1f} MB)")

    # ── Verify with onnxruntime ──────────────────────────────────────────
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(onnx_path)
        test_input = np.random.randn(1, 1, ip_height, time_frames).astype(np.float32)
        outputs = sess.run(None, {"spectrogram": test_input})
        print(f"  Verification OK: {len(outputs)} outputs")
        for i, o in enumerate(outputs):
            print(f"    output[{i}]: shape={o.shape}, dtype={o.dtype}")
    except ImportError:
        print("  (onnxruntime not installed — skipping verification)")

    # ── Write labels ─────────────────────────────────────────────────────
    labels_path = os.path.join(args.output_dir, "labels.csv")
    with open(labels_path, "w") as f:
        for name in class_names:
            f.write(name + "\n")
    print(f"  Written: {labels_path} ({len(class_names)} classes)")

    print("\nDone.")


if __name__ == "__main__":
    main()
