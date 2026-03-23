#!/usr/bin/env python3
"""Persistent ONNX Runtime inference worker.

Spawned by the Rust processing server for models with `prefer_ort=true`.
The Rust `ort` crate hangs during session creation for models with
DFT/STFT ops; Python's `onnxruntime` pip package handles them fine.

Communicates over stdin/stdout using a binary protocol on raw
(unbuffered) file descriptors:

  Load a model
  → {"cmd":"load","id":"birdnet3","path":"/models/birdnet3/birdnet3.onnx"}\n
  ← {"ok":true,"inputs":1,"outputs":2}\n

  Run inference
  → {"cmd":"predict","id":"birdnet3","shape":[1,96000],"output_index":1,"n_floats":96000}\n
  (followed by 96000 * 4 = 384000 raw bytes of f32 little-endian)
  ← {"ok":true,"n_floats":6522}\n
  (followed by 6522 * 4 = 26088 raw bytes of f32 little-endian)

  Shutdown
  → {"cmd":"quit"}\n

All I/O uses raw OS file descriptors (unbuffered) to avoid Python's
BufferedReader/TextIOWrapper read-ahead consuming binary tensor data
meant for explicit read() calls.
"""

from __future__ import annotations

import json
import os
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


# ── Raw I/O helpers ───────────────────────────────────────────────────

def read_exact(fd: int, n: int) -> bytes:
    """Read exactly n bytes from a raw file descriptor."""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = os.read(fd, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_line(fd: int) -> bytes:
    """Read a single \\n-terminated line from a raw file descriptor."""
    buf = bytearray()
    while True:
        b = os.read(fd, 1)
        if not b or b == b"\n":
            return bytes(buf)
        buf.extend(b)


def write_all(fd: int, data: bytes) -> None:
    """Write all bytes to a raw file descriptor."""
    mv = memoryview(data)
    while mv:
        n = os.write(fd, mv)
        mv = mv[n:]


# ── Predict ───────────────────────────────────────────────────────────

def handle_predict(msg: dict, in_fd: int, out_fd: int) -> bool:
    """Read raw f32 bytes, run inference, write raw f32 bytes.  Returns True on success."""
    model_id = msg["id"]
    shape = msg["shape"]
    output_index = msg["output_index"]
    n_floats = msg["n_floats"]

    # Read raw input tensor bytes
    n_bytes = n_floats * 4
    raw = read_exact(in_fd, n_bytes)
    if len(raw) != n_bytes:
        resp = {"ok": False, "error": f"Expected {n_bytes} bytes, got {len(raw)}"}
        write_all(out_fd, (json.dumps(resp) + "\n").encode())
        return False

    data = np.frombuffer(raw, dtype=np.float32).copy().reshape(shape)

    sess = sessions.get(model_id)
    if sess is None:
        resp = {"ok": False, "error": f"Model '{model_id}' not loaded"}
        write_all(out_fd, (json.dumps(resp) + "\n").encode())
        return False

    inp_name = sess.get_inputs()[0].name
    try:
        outputs = sess.run(None, {inp_name: data})
    except Exception as e:
        resp = {"ok": False, "error": str(e)}
        write_all(out_fd, (json.dumps(resp) + "\n").encode())
        return False

    result = outputs[output_index].flatten().astype(np.float32)
    n_out = len(result)
    out_bytes = result.tobytes()

    # Write JSON header + raw output bytes atomically
    header = (json.dumps({"ok": True, "n_floats": n_out}) + "\n").encode()
    write_all(out_fd, header + out_bytes)
    return True


# ── Main loop ─────────────────────────────────────────────────────────

def main() -> None:
    # Use raw OS file descriptors — no Python buffering layers that
    # could read-ahead and consume binary data meant for explicit reads.
    in_fd = sys.stdin.fileno()
    out_fd = sys.stdout.fileno()

    while True:
        raw_line = read_line(in_fd)
        if not raw_line:
            break  # EOF
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            resp = {"ok": False, "error": f"Invalid JSON: {e}"}
            write_all(out_fd, (json.dumps(resp) + "\n").encode())
            continue

        cmd = msg.get("cmd")

        if cmd == "quit":
            break
        elif cmd == "load":
            resp = handle_load(msg)
        elif cmd == "predict":
            handle_predict(msg, in_fd, out_fd)
            continue  # response already sent
        else:
            resp = {"ok": False, "error": f"Unknown command: {cmd}"}

        write_all(out_fd, (json.dumps(resp) + "\n").encode())


if __name__ == "__main__":
    main()
