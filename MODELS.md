# Bioacoustics Models for Gaia Audio

An overview of publicly available bioacoustics models that can be integrated
into gaia-audio's manifest-based model system.  Models are grouped by
readiness: **ready to download and use** vs. **requires training**.

---

## Ready to Download and Use

These models have pre-trained weights in TFLite or ONNX format and can be
plugged into gaia-audio by writing a `manifest.toml` and placing the files
in a model subdirectory.

### BirdNET V2.4

| Field | Value |
|-------|-------|
| **Species** | ~6,500 bird species worldwide |
| **Format** | TFLite (fp32, fp16, int8) + ONNX via Keras conversion |
| **Sample rate** | 48 kHz |
| **Chunk duration** | 3.0 s |
| **Source** | [Zenodo 15050749](https://zenodo.org/records/15050749) |
| **Status** | Fully integrated — auto-download, metadata model, language packs |

Already shipped as `examples/birds_manifest.toml`.  The Containerfile
bundles the manifest and pre-converts the Keras model to ONNX at build
time — no manual download or conversion needed.

**Deployment:** Automatic.  The processing container image includes the
pre-converted ONNX classifier (`audio-model.onnx`, ~49 MB) and metadata
model (`meta-model.onnx`, ~28 MB) at `/usr/local/share/gaia/models/`.
The entrypoint seeds the manifest into the `/models` volume; on first
start the server copies the baked-in ONNX files into place.

---

### Google Perch 2.0

| Field | Value |
|-------|-------|
| **Species** | ~15,000 (birds, frogs, crickets, grasshoppers, mammals) |
| **Format** | ONNX (HuggingFace) |
| **Sample rate** | 32 kHz |
| **Chunk duration** | 5.0 s |
| **Source (ONNX)** | [HuggingFace — justinchuby/Perch-onnx](https://huggingface.co/justinchuby/Perch-onnx) (public, Apache-2.0) |
| **Source (labels)** | [HuggingFace — cgeorgiaw/Perch](https://huggingface.co/cgeorgiaw/Perch) (public) |
| **Slug** | `perch` |
| **Status** | Fully integrated — auto-download from HuggingFace |

**Best immediate option for multi-taxa detection.**  Perch covers birds plus
a broad set of non-bird species (frogs, crickets, grasshoppers, mammals) in
a single model.  Unlike BirdNET, it accepts raw audio input — no mel
spectrogram preprocessing is needed (`onnx_is_classifier = false`).

> **Note:** The original Google repo
> (`google/bird-vocalization-classifier`) is **gated** and requires
> HuggingFace authentication.  The manifest uses community mirrors that
> are publicly accessible without login.

Already shipped as `examples/perch_manifest.toml`.

**Deployment:** Automatic.  The entrypoint seeds the manifest; the
processing server downloads the ONNX model (~150 MB) and labels on
first start via the `[download.direct_files]` section.

---

### BatDetect2

| Field | Value |
|-------|-------|
| **Species** | UK bat species |
| **Format** | PyTorch (`.pth.tar`); needs ONNX export |
| **Sample rate** | 256 kHz |
| **Chunk duration** | 1.0 s |
| **Source** | [GitHub — macaodha/batdetect2](https://github.com/macaodha/batdetect2) |
| **Weights (public)** | [Net2DFast_UK_same.pth.tar](https://raw.githubusercontent.com/macaodha/batdetect2/main/batdetect2/models/Net2DFast_UK_same.pth.tar) (~7.6 MB) |
| **Slug** | `batdetect2` |
| **Status** | Experimental — manifest + infrastructure in place; ONNX conversion required |

Already shipped as `examples/bats_manifest.toml`.  Requires a microphone
capable of capturing ultrasonic frequencies (256 kHz sample rate), such
as an AudioMoth or Dodotronic Ultramic.

Like Perch, it accepts raw audio input (`onnx_is_classifier = false`)
and outputs raw logits (`apply_softmax = true`).

**Deployment status:**  The container infrastructure (manifest bundling,
entrypoint seeding, gaia-core container management via
`gaia-audio-processing-batdetect2`) is fully in place.  However:

1. **No ONNX file available** — BatDetect2 ships as a PyTorch
   checkpoint (`.pth.tar`).  The HuggingFace repo (`macaodha/batdetect2`)
   is **gated** (returns 401).  The PyTorch weights are publicly
   available on GitHub but need to be exported to ONNX.
2. **Detection architecture** — BatDetect2 is an object detector (not
   a simple classifier).  It outputs bounding boxes + class predictions
   on spectrograms.  Adapting the gaia-audio inference pipeline to
   support detection models is future work.

To prepare manually:
```bash
pip install batdetect2
# Export model to ONNX (conversion script TBD)
# Place batdetect2.onnx + labels.csv in /models/batdetect2/
```

---

### BirdNET+ V3.0 (Developer Preview)

| Field | Value |
|-------|-------|
| **Species** | ~11,000 (birds + expanded non-bird taxa) |
| **Format** | TFLite |
| **Sample rate** | 32 kHz |
| **Chunk duration** | Variable length input |
| **Source** | [Zenodo 18247420](https://zenodo.org/records/18247420) |
| **Status** | Preview — needs testing with `tract`; variable-length input may require padding |

Developer preview from the BirdNET team.  Extends the species list to include
non-bird species (amphibians, insects, mammals).  Uses 32 kHz (vs. 48 kHz
for V2.4) and supports variable-length input, which may need adapter code in
the processing server to pad or tile chunks.

---

### SurfPerch (Coral Reef Sounds)

| Field | Value |
|-------|-------|
| **Species/Classes** | Coral reef bioacoustic events |
| **Format** | TFLite |
| **Sample rate** | 32 kHz |
| **Chunk duration** | 5.0 s |
| **Source** | [Kaggle — google/surfperch](https://www.kaggle.com/models/google/surfperch) |
| **Status** | Compatible — same architecture as Perch, needs `manifest.toml` |

A Perch variant specialised for underwater reef soundscapes.  Niche use case
but demonstrates the system's ability to load arbitrary classifiers.

---

---

## Requires Training

These approaches don't have a single downloadable model file you can drop in.
They represent state-of-the-art architectures and training recipes from
recent BirdCLEF competitions that would need to be trained on target data and
exported to ONNX.

### BirdCLEF+ 2025 — Dedicated Insecta/Amphibia Model (1st Place)

| Field | Value |
|-------|-------|
| **Architecture** | EfficientNet-B0/B3/B4 + SED head (timm) |
| **Training data** | Xeno-Canto: ~16,218 insecta + ~979 amphibia samples (~700 species) |
| **Insect families** | Cicadidae (cicadas), Gryllidae (crickets), Tettigoniidae (katydids) |
| **Mel params** | 224 bins, fmin=0, fmax=16000, n_fft=4096, hop=1252, 32 kHz |
| **Input duration** | 20 s chunks (longer helps with repetitive insect calls) |
| **Inference** | OpenVINO; overlapping predictions, smoothing kernel `[0.1, 0.2, 0.4, 0.2, 0.1]` |
| **Source** | Solution writeup only (no public code or weights) |
| **Reference** | [Kaggle writeup](https://www.kaggle.com/competitions/birdclef-2025/discussion) |

The winning BirdCLEF+ 2025 team trained a **separate model specifically for
amphibians and insects** using extended Xeno-Canto data.  This is the most
relevant recipe for building a frog/cricket/cicada classifier.  The approach
is: download Xeno-Canto recordings for target families → train a single
EfficientNet SED model → export to ONNX.

Key insight: longer chunks (20 s vs. BirdNET's 3 s) improved detection of
repetitive insect calls.

---

### BirdCLEF+ 2025 — Self-Distillation Pipeline (5th Place)

| Field | Value |
|-------|-------|
| **Architecture** | EfficientNetV2-S/B3 + SED head |
| **Mel params** | 192 bins, fmin=20, fmax=15000, 32 kHz |
| **Input duration** | 10 s random segments |
| **Inference** | OpenVINO, 2.5 s overlap, smoothing `[0.1, 0.8, 0.1]` |
| **Source (code)** | [GitHub — myso1987/BirdCLEF-2025-5th-place-solution](https://github.com/myso1987/BirdCLEF-2025-5th-place-solution) |
| **Source (weights)** | Available on Kaggle (PyTorch + OpenVINO, competition species only) |

Open-source training pipeline with 3-stage iterative self-distillation.
Pre-trained weights are available but classify **only the ~460 competition
species from Colombia** — not a general-purpose model.  The code is useful as
a training template if you want to retrain on different species.

The weights could be converted: PyTorch → ONNX → `tract-onnx`, but the
species list is too narrow for general use without retraining.

---

### BirdCLEF+ 2025 — Pseudo-Label Pipeline (2nd Place)

| Field | Value |
|-------|-------|
| **Architecture** | EfficientNetV2-S, eca_nfnet_l0 + SED/MLP heads |
| **Training recipe** | XC pretrain (~7,500 species) → fine-tune → iterative pseudo-labeling |
| **Mel params** | Standard timm-compatible spectrograms, 32 kHz |
| **Source (code)** | [GitHub — VSydorskyy/BirdCLEF_2025_2nd_place](https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place) |
| **Source (weights)** | Competition-specific only |

Provides a complete, open-source training + inference pipeline.  The key
innovation is multi-iteration pseudo-labeling on unlabeled soundscapes,
boosting AUC from 0.83 → 0.91.  The code would be a solid starting point
for training a custom classifier on new species.

---

### BirdCLEF+ 2025 — ONNX Ensemble (3rd Place)

| Field | Value |
|-------|-------|
| **Architecture** | EfficientNet-B0, EfficientNetV2-B3/S, MNASNet, SPNASNet |
| **Mel params** | n_mels=128 or 96, fmin=0, fmax=16000, 32 kHz |
| **Key technique** | All 20 models exported to ONNX (2-3× faster than PyTorch) |
| **Source** | Writeup only (no public code) |

Most relevant for validating that **ONNX inference** at competition quality is
practical and fast.  Uses the broadest range of backbone architectures.

---

### BirdCLEF 2024 — Raw Signal + Mel Ensemble (4th Place)

| Field | Value |
|-------|-------|
| **Architecture** | seresnext26ts, rexnet_150, inception-next-nano, EfficientNet-B0 |
| **Key technique** | Raw waveform CNN alongside mel models; OpenVINO INT8 quantization (30-40% speedup) |
| **Source (code)** | [GitHub — yoku001/BirdCLEF2024-4th-place-solution-melspec](https://github.com/yoku001/BirdCLEF2024-4th-place-solution-melspec) |
| **Source (raw signal)** | [GitHub — tamotamo17/BirdCLEF2024-4th-place-solution-raw-signal](https://github.com/tamotamo17/BirdCLEF2024-4th-place-solution-raw-signal) |

Interesting for the **raw-signal approach**: downsample audio to 16 kHz,
reshape 80,000 samples to 625×128, feed directly to EfficientNet.  This
bypasses mel spectrograms entirely.  Also shows INT8 post-training
quantization for edge inference.

---

## Datasets Only (No Pre-trained Model)

These are labeled audio datasets that could be used to **train** a custom
classifier but do not include a downloadable model.

| Dataset | Taxa | Source |
|---------|------|--------|
| **AnuraSet** | 42 Neotropical frog species | [Zenodo 11368834](https://zenodo.org/records/11368834) |
| **FrogID** | Australian frog species | Dataset via Australian Museum |
| **Xeno-Canto (insects)** | Cicadas, crickets, katydids | [xeno-canto.org](https://xeno-canto.org) (filter by Insecta) |

---

## Common Architecture Across BirdCLEF Winners

All BirdCLEF 2024-2025 top solutions share a common pattern that informs what
a custom gaia-audio model should look like:

```
Audio (32 kHz) → Mel Spectrogram → 2D CNN backbone (EfficientNet) → SED Head → Species probabilities
```

| Parameter | Typical Range |
|-----------|---------------|
| Sample rate | 32 kHz |
| Mel bins | 96-224 |
| fmin / fmax | 0-20 Hz / 15000-16000 Hz |
| n_fft | 2048-4096 |
| Chunk duration | 5-20 s (longer for insects) |
| Backbone | EfficientNet-B0 to B4, EfficientNetV2-S |
| Head | SED (Sound Event Detection) with attention pooling |
| Loss | Focal BCE |
| Export format | ONNX or OpenVINO |

---

## Integration Checklist

For any new model, you need:

1. **Model file** — `.onnx` (preferred) or `.tflite`
2. **Labels file** — one label per line, matching model output order
3. **`manifest.toml`** — placed in `$MODEL_DIR/<slug>/manifest.toml`
4. **`slug` field** — must be set explicitly in the manifest `[model]`
   section; the slug is used as the container name suffix and the
   `MODEL_SLUGS` filter value
5. Verify `sample_rate` and `chunk_duration` match the model's expectations
6. Set `apply_softmax = true` if the model outputs raw logits (Perch, BatDetect2, competition models)
7. Set `onnx_is_classifier = false` for models that accept raw audio input
   (only BirdNET's extracted classifier sub-model needs `true`)
8. Mel spectrogram parameters in `mel.rs` only apply when `onnx_is_classifier = true`

### Container deployment

Manifests for built-in models (BirdNET, Perch, BatDetect2) are bundled
in the processing container image at
`/usr/local/share/gaia/manifests/<slug>/manifest.toml`.
The `entrypoint.sh` script seeds them into the `/models` volume on
container start (no-clobber — existing files are not overwritten).

The processing server auto-discovers model directories via `discover_manifests()`
and filters by the `MODEL_SLUGS` environment variable.  Each model runs
in its own container instance: `gaia-audio-processing-<slug>`.

After starting a processing container, `gaia-core` spawns a background
validation task (`validate_audio_processing()`) that monitors container
logs for up to 30 seconds, checking for `"Model ready:"`,
`"No models loaded"`, or `"Cannot load model"` messages.  The result is
shown in the dashboard container status.

### Adding a new model

1. Write a `manifest.toml` — see the existing examples in `examples/`
2. Add a `COPY` line to `processing/Containerfile` to bundle it:
   ```dockerfile
   COPY examples/mymodel_manifest.toml /usr/local/share/gaia/manifests/myslug/manifest.toml
   ```
3. Add the model to `default_audio_models()` in `gaia-core/src/config.rs`
4. Rebuild the processing container image
5. The next `gaia-core` start will create a
   `gaia-audio-processing-myslug` container automatically
