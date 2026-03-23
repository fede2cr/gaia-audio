#!/bin/sh
# Gaia Processing Server – entrypoint
#
# Seeds bundled model assets (manifests, metadata models, label files)
# into the /models volume before starting the processing server binary.
# Bundled files always overwrite existing copies so that fixes from
# newer container builds take effect without manual intervention.

set -e

BUNDLED_DIR="/usr/local/share/gaia/manifests"
MODELS_DIR="/models"

# ── Purge stale meta-model.onnx that tract-onnx cannot load ──────
# Early container builds shipped a meta-model.onnx with a dynamic
# Reshape shape input (node #13 "BirdNET/MNET_CONVERT/Reshape").
# tract-onnx rejects this at runtime ("shape input is variable").
# The fixed version is baked into newer images and will be seeded
# below, but only if the stale copy is removed first.
STALE_META="/models/birdnet3/meta-model.onnx"
if [ -f "$STALE_META" ] && [ -f "$BUNDLED_DIR/birdnet3/meta-model.onnx" ]; then
    if ! cmp -s "$STALE_META" "$BUNDLED_DIR/birdnet3/meta-model.onnx"; then
        echo "[entrypoint] Removing stale $STALE_META (will be replaced by bundled copy)"
        rm -f "$STALE_META"
    fi
fi

# ── Purge stale Perch no-DFT variant (switched to standard model) ─
# Previous builds downloaded perch_v2_no_dft.onnx (MatMul-replaced
# DFT) as model.onnx.  That variant hangs ORT during graph
# optimisation.  The manifest now downloads the standard
# perch_v2.onnx (native DFT) which ORT handles correctly.
# Remove the stale no-DFT copy so the download logic fetches the
# correct standard model.
STALE_PERCH="/models/perch/perch_v2.onnx"
STALE_PERCH_NODFT="/models/perch/perch_v2_no_dft.onnx"
for f in "$STALE_PERCH" "$STALE_PERCH_NODFT"; do
    if [ -f "$f" ]; then
        echo "[entrypoint] Removing stale $f (using standard model.onnx now)"
        rm -f "$f"
    fi
done
# Force re-download of model.onnx if it's the old no-DFT variant.
# We detect it by size: the no-DFT variant is ~300 MB, the standard
# model is ~68 MB.
PERCH_MODEL="/models/perch/model.onnx"
if [ -f "$PERCH_MODEL" ]; then
    SIZE=$(stat -c%s "$PERCH_MODEL" 2>/dev/null || echo 0)
    # If > 200 MB it's almost certainly the no-DFT variant
    if [ "$SIZE" -gt 200000000 ]; then
        echo "[entrypoint] Removing stale Perch no-DFT model.onnx (${SIZE} bytes, > 200 MB)"
        rm -f "$PERCH_MODEL"
    fi
fi

if [ -d "$BUNDLED_DIR" ]; then
    for model_dir in "$BUNDLED_DIR"/*/; do
        slug=$(basename "$model_dir")
        target="$MODELS_DIR/$slug"
        mkdir -p "$target"

        # Always overwrite every bundled file.  These are small assets
        # baked into the container image (manifests, metadata models,
        # label files).  Large downloaded artefacts (e.g. birdnet3.onnx)
        # are NOT in this directory — they are fetched at runtime and
        # stored directly in the volume, so they are unaffected.
        for src_file in "$model_dir"*; do
            [ -f "$src_file" ] || continue
            fname=$(basename "$src_file")
            dest="$target/$fname"
            cp "$src_file" "$dest"
            echo "[entrypoint] Updated $dest"
        done
    done
fi

exec gaia-processing "$@"
