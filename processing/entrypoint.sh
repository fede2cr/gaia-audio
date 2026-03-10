#!/bin/sh
# Gaia Processing Server – entrypoint
#
# Seeds bundled model manifests into the /models volume (no-clobber)
# before starting the processing server binary.  This ensures every
# processing container has the manifest files it needs to discover and
# download its models automatically.
#
# Files already present in the volume are NOT overwritten, so
# user-customised manifests are preserved.

set -e

BUNDLED_DIR="/usr/local/share/gaia/manifests"
MODELS_DIR="/models"

if [ -d "$BUNDLED_DIR" ]; then
    for model_dir in "$BUNDLED_DIR"/*/; do
        slug=$(basename "$model_dir")
        target="$MODELS_DIR/$slug"
        mkdir -p "$target"

        # Always overwrite manifest.toml so updated manifests from
        # newer container builds replace stale ones on the volume.
        # Other files (model data, labels) use no-clobber to avoid
        # re-downloading large artefacts that haven't changed.
        for src_file in "$model_dir"*; do
            [ -f "$src_file" ] || continue
            fname=$(basename "$src_file")
            dest="$target/$fname"
            if [ "$fname" = "manifest.toml" ]; then
                cp "$src_file" "$dest"
                echo "[entrypoint] Updated $dest"
            elif [ ! -f "$dest" ]; then
                cp "$src_file" "$dest"
                echo "[entrypoint] Seeded $dest"
            fi
        done
    done
fi

exec gaia-processing "$@"
