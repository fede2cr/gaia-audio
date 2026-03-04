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

        # Copy each file only if the target does not already exist.
        for src_file in "$model_dir"*; do
            [ -f "$src_file" ] || continue
            dest="$target/$(basename "$src_file")"
            if [ ! -f "$dest" ]; then
                cp "$src_file" "$dest"
                echo "[entrypoint] Seeded $dest"
            fi
        done
    done
fi

exec gaia-processing "$@"
