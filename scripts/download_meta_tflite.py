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

os.remove("tflite.zip")
shutil.rmtree("/convert/tflite_raw")
print(f"Extracted: {os.path.getsize(OUTPUT)} bytes → {OUTPUT}")
