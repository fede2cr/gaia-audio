#!/usr/bin/env python3
"""Download the BirdNET V2.4 TFLite zip and extract only meta-model.tflite."""

import os
import shutil
import urllib.request
import zipfile

URL = "https://zenodo.org/api/records/15050749/files/BirdNET_v2.4_tflite.zip/content"
OUTPUT = "/convert/meta-model.tflite"

print("Downloading BirdNET V2.4 TFLite (fp32) for meta-model...")
urllib.request.urlretrieve(URL, "tflite.zip")

z = zipfile.ZipFile("tflite.zip")
z.extractall("/convert/tflite_raw")

meta = next(
    (
        os.path.join(r, f)
        for r, _, fs in os.walk("/convert/tflite_raw")
        for f in fs
        if "meta" in f.lower() and f.endswith(".tflite")
    ),
    None,
)
assert meta, "meta-model.tflite not found in TFLite zip"
shutil.copy2(meta, OUTPUT)

# Also extract en_us.txt (V2.4 species labels) for reuse by birdnet3.
labels = next(
    (
        os.path.join(r, f)
        for r, _, fs in os.walk("/convert/tflite_raw")
        for f in fs
        if f == "en_us.txt"
    ),
    None,
)
if labels:
    shutil.copy2(labels, "/convert/en_us.txt")
    print(f"Extracted labels: {os.path.getsize('/convert/en_us.txt')} bytes → /convert/en_us.txt")
else:
    print("WARNING: en_us.txt not found in TFLite zip")

os.remove("tflite.zip")
shutil.rmtree("/convert/tflite_raw")
print(f"Extracted: {os.path.getsize(OUTPUT)} bytes → {OUTPUT}")
