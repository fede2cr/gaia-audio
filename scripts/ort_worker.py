#!/usr/bin/env python3
"""Persistent ONNX Runtime inference worker.

Spawned by the Rust processing server for models with `prefer_ort=true`.
The Rust `ort` crate hangs during session creation for models with
DFT/STFT ops; Python's `onnxruntime` pip package handles them fine.

Communicates over stdin/stdout using a JSON-lines protocol:

  Load a model
  → {"cmd":"load","id":"birdnet3","path":"/models/birdnet3/birdnet3.onnx"}
  ← {"ok":true,"inputs":1,"outputs":2}

  Run inference
  → {"cmd":"predict","id":"birdnet3","shape":[1,96000],"output_index":1,"n_floats":96000}
  (followed by 96000 * 4 = 384000 raw bytes of f32 little-endian)
  ← {"ok":true,"n_floats":6522}
  (followed by 6522 * 4 = 26088 raw bytes of f32 little-endian)

  Shutdown
  → {"cmd":"quit"}

Raw binary framing for tensor data avoids the overhead of JSON-encoding
hundreds of thousands of floats (~800 KB as JSON vs ~384 KB raw).
"""

from __future__ import annotations

import json
import struct
import sys

import numpy as np
import onnxruntime as ort


sessions: dict[str, ort.InferenceSession] = {}


def handle_load(msg: dict) -> dict:
    model_id = msg["id"]
    path = msg["path"]
    opts = ort.SessionOptions()
    opts.log_severity_level = 3       # suppress ORT verbose logging
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 4
    try:
        sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    except Exception as e:
        return {"ok": False, "error": str(e)}
    sessions[model_id] = sess
    return {
        "ok": True,
        "inputs": len(sess.get_inputs()),
        "outputs": len(sess.get_outputs()),
    }


def handle_predict(msg: dict, stdin) -> dict | None:
    """Read raw f32 bytes from stdin, run inference, write raw f32 bytes to stdout."""
    model_id = msg["id"]
    shape = msg["shape"]
    output_index = msg["output_index"]
    n_floats = msg["n_floats"]

    # Read raw input tensor bytes from the binary stdin handle
    n_bytes = n_floats * 4
    raw = stdin.read(n_bytes)
    if len(raw) != n_bytes:
        return {"ok": False, "error": f"Expected {n_bytes} bytes, got {len(raw)}"}

    data = np.frombuffer(raw, dtype=np.float32).reshape(shape)

    sess = sessions.get(model_id)
    if sess is None:
        return {"ok": False, "error": f"Model '{model_id}' not loaded"}

    inp_name = sess.get_inputs()[0].name
    try:
        outputs = sess.run(None, {inp_name: data})
    except Exception as e:
        return {"ok": False, "error": str(e)}

    result = outputs[output_index].flatten().astype(np.float32)
    n_out = len(result)
    out_bytes = result.tobytes()

    # Write JSON header + raw output bytes
    header = json.dumps({"ok": True, "n_floats": n_out})
    sys.stdout.write(header + "\n")
    sys.stdout.flush()
    sys.stdout.buffer.write(out_bytes)
    sys.stdout.buffer.flush()
    return None  # already sent


def main() -> None:
    # IMPORTANT: read from sys.stdin.buffer (binary mode) exclusively.
    # `for line in sys.stdin:` uses text-mode read-ahead buffering that
    # consumes the raw binary tensor bytes that follow JSON lines,
    # causing predict to hang waiting for data that's already been eaten.
    stdin = sys.stdin.buffer

    while True:
        raw_line = stdin.readline()
        if not raw_line:
            break  # EOF
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            resp = {"ok": False, "error": f"Invalid JSON: {e}"}
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
            continue

        cmd = msg.get("cmd")

        if cmd == "quit":
            break
        elif cmd == "load":
            resp = handle_load(msg)
        elif cmd == "predict":
            resp = handle_predict(msg, stdin)
            if resp is None:
                continue  # already sent in handle_predict
        else:
            resp = {"ok": False, "error": f"Unknown command: {cmd}"}

        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
