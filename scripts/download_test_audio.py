#!/usr/bin/env python3
"""Download a known bird recording for build-stage smoke tests.

Downloads a clear, well-labelled recording of Common Blackbird (Turdus merula),
converts to 48 kHz mono WAV (the highest sample rate used by any model —
BirdNET 2.4 runs at 48 kHz, others at 32 kHz).

The recording is chosen because:
  - Common Blackbird is in all three bird model label sets
    (BirdNET 2.4, BirdNET+ V3.0, and Google Perch 2.0).
  - Multiple CC-licensed recordings are available on stable hosting.

Sources tried in order:
  1. Wikimedia Commons (extremely stable, no API changes)
  2. Xeno-Canto direct download (bypasses API)

Usage:
    python3 download_test_audio.py -o /path/to/test_audio.wav

The output file is a 48 kHz mono 16-bit PCM WAV, trimmed to 15 seconds
(enough for all chunk sizes: 3 s for BirdNET, 5 s for Perch).
"""

import argparse
import os
import struct
import subprocess
import sys
import tempfile
import urllib.request

# ── Configuration ─────────────────────────────────────────────────────

TARGET_SR = 48000      # BirdNET 2.4 sample rate (highest of all models)
TARGET_DURATION = 15   # seconds — covers 3 × 5 s Perch chunks or 5 × 3 s BirdNET chunks
SPECIES = "Turdus merula"

# Sources to try, in order.  Each is (description, URL).
# All are CC-licensed Common Blackbird (Turdus merula) recordings.
DOWNLOAD_CANDIDATES = [
    # Wikimedia Commons — stable hosting, CC BY-SA 3.0
    # ~25s clear blackbird song, OGG format
    (
        "Wikimedia Commons: Blackbird song (Turdus merula)",
        "https://upload.wikimedia.org/wikipedia/commons/a/a7/Blackbird_song.ogg",
    ),
    # Wikimedia Commons — another blackbird recording
    (
        "Wikimedia Commons: Common Blackbird song",
        "https://upload.wikimedia.org/wikipedia/commons/6/6e/Turdus_merula_2.ogg",
    ),
    # Xeno-Canto direct download (bypasses the JSON API which may change)
    (
        "Xeno-Canto: XC800880 (Turdus merula)",
        "https://xeno-canto.org/800880/download",
    ),
    (
        "Xeno-Canto: XC763498 (Turdus merula)",
        "https://xeno-canto.org/763498/download",
    ),
]


def download_recording(desc: str, url: str, dest: str) -> bool:
    """Download a recording from a URL.  Returns True on success."""
    print(f"  Source: {desc}")
    print(f"  URL: {url}")
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "GaiaAudio-SmokeTest/1.0 (build-stage CI)"
        })
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
            if len(data) < 1000:
                print(f"  Response too small ({len(data)} bytes), skipping")
                return False
            with open(dest, "wb") as f:
                f.write(data)
        size_kb = os.path.getsize(dest) / 1024
        print(f"  Downloaded: {size_kb:.0f} KB")
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def convert_to_wav(src: str, dest: str, sr: int, duration: int):
    """Convert any audio file to mono WAV at the target sample rate using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-ac", "1",                  # mono
        "-ar", str(sr),              # target sample rate
        "-t", str(duration),         # limit duration
        "-sample_fmt", "s16",        # 16-bit PCM
        dest,
    ]
    print(f"  Converting to {sr} Hz mono WAV ({duration}s)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f"  Output: {dest} ({size_mb:.1f} MB)")


def verify_wav(path: str, expected_sr: int, min_duration: float):
    """Quick sanity check on the output WAV."""
    with open(path, "rb") as f:
        riff = f.read(4)
        if riff != b"RIFF":
            raise ValueError(f"Not a valid WAV file (header: {riff!r})")
        f.read(4)  # file size
        wave = f.read(4)
        if wave != b"WAVE":
            raise ValueError(f"Not a WAVE file (got {wave!r})")
        # Find fmt chunk
        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                raise ValueError("Could not find fmt chunk")
            chunk_size = struct.unpack("<I", f.read(4))[0]
            if chunk_id == b"fmt ":
                fmt_data = f.read(chunk_size)
                audio_fmt = struct.unpack("<H", fmt_data[0:2])[0]
                channels = struct.unpack("<H", fmt_data[2:4])[0]
                sample_rate = struct.unpack("<I", fmt_data[4:8])[0]
                break
            else:
                f.seek(chunk_size, 1)

    if audio_fmt != 1:
        raise ValueError(f"Expected PCM (1), got format {audio_fmt}")
    if channels != 1:
        raise ValueError(f"Expected mono (1 channel), got {channels}")
    if sample_rate != expected_sr:
        raise ValueError(f"Expected {expected_sr} Hz, got {sample_rate} Hz")

    file_size = os.path.getsize(path)
    # Approximate duration: (file_size - header) / (sr * 2 bytes)
    duration = (file_size - 44) / (sample_rate * 2)
    if duration < min_duration:
        raise ValueError(
            f"Recording too short: {duration:.1f}s < {min_duration}s minimum"
        )
    print(f"  Verified: {sample_rate} Hz, {channels}ch, ~{duration:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Download a known bird recording for smoke tests"
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output WAV file path"
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = os.path.join(tmpdir, "raw_audio")

        downloaded = False
        for i, (desc, url) in enumerate(DOWNLOAD_CANDIDATES, 1):
            print(f"\nCandidate {i}/{len(DOWNLOAD_CANDIDATES)}:")
            if download_recording(desc, url, raw_path):
                downloaded = True
                break

        if not downloaded:
            print(
                "ERROR: All download candidates failed.",
                file=sys.stderr,
            )
            sys.exit(1)

        convert_to_wav(raw_path, args.output, TARGET_SR, TARGET_DURATION)
        verify_wav(args.output, TARGET_SR, min_duration=10.0)

    print(f"\nTest audio ready: {args.output}")
    print(f"Species: {SPECIES}")
    print(f"Expected detection by: BirdNET V2.4, BirdNET+ V3.0, Google Perch 2.0")


if __name__ == "__main__":
    main()
