#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Gaia Audio Server – quick installer
#
# Creates the directory layout, compose.yaml, gaia.conf, and birds model
# manifest with automatic Zenodo download enabled.
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/.../install.sh | bash
#   # or
#   bash install.sh [INSTALL_DIR]
#
# Default install directory: ~/gaia
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configurable defaults ────────────────────────────────────────────────────
INSTALL_DIR="${1:-${GAIA_DIR:-$HOME/gaia}}"
REGISTRY="${GAIA_REGISTRY:-docker.io/fede2}"

# Colours (disabled if not a terminal)
if [[ -t 1 ]]; then
  GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; NC='\033[0m'
else
  GREEN=''; CYAN=''; YELLOW=''; NC=''
fi

info()  { echo -e "${GREEN}[+]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
step()  { echo -e "${CYAN}───${NC} $*"; }

# ── Pre-flight checks ───────────────────────────────────────────────────────
check_cmd() {
  if ! command -v "$1" &>/dev/null; then
    warn "$1 not found. Please install it first."
    return 1
  fi
}

COMPOSE_CMD=""
if command -v podman &>/dev/null && podman compose version &>/dev/null 2>&1; then
  COMPOSE_CMD="podman compose"
elif command -v docker &>/dev/null && docker compose version &>/dev/null 2>&1; then
  COMPOSE_CMD="docker compose"
elif command -v docker-compose &>/dev/null; then
  COMPOSE_CMD="docker-compose"
else
  warn "Neither 'podman compose' nor 'docker compose' found."
  warn "Install one of them before running the stack."
  COMPOSE_CMD="podman compose"  # default for instructions
fi
info "Using: $COMPOSE_CMD"

# ── Create directory structure ───────────────────────────────────────────────
info "Installing to ${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}"/{models/birds,data/extracted,backups}

step "Created directories:"
echo "  ${INSTALL_DIR}/"
echo "  ├── models/birds/"
echo "  ├── data/extracted/"
echo "  └── backups/"

# ── gaia.conf ────────────────────────────────────────────────────────────────
CONF="${INSTALL_DIR}/gaia.conf"
if [[ -f "$CONF" ]]; then
  warn "gaia.conf already exists – skipping (won't overwrite)"
else
  cat > "$CONF" << 'EOF'
# ─────────────────────────────────────────────────────────────────────────────
# Gaia Audio Server configuration
# ─────────────────────────────────────────────────────────────────────────────
# Location (used for species filtering & metadata model)
LATITUDE=-1
LONGITUDE=-1

# Detection thresholds
CONFIDENCE=0.7
SENSITIVITY=1.25
OVERLAP=0.0

# Audio capture
RECORDING_LENGTH=15
CHANNELS=1
# REC_CARD=default
# For USB mics use the ALSA card identifier, e.g.:
#   REC_CARD=plughw:2,0
#   REC_CARD=plughw:CARD=Usb_Microphone,DEV=0
# Run "arecord -l" on the host to list available capture devices.

# Directories (container paths – match compose.yaml volumes)
RECS_DIR=/data
EXTRACTED=/data/extracted
MODEL_DIR=/models
DB_PATH=/data/birds.db

# Language for common names (en, de, fr, es, …)
DATABASE_LANG=en

# Model variant: fp32, fp16 (default), or int8
# MODEL_VARIANT=fp16

# RTSP streams (comma-separated, leave empty for local mic)
# RTSP_STREAMS=rtsp://cam1:554/stream

# Network
CAPTURE_LISTEN_ADDR=0.0.0.0:8089
CAPTURE_SERVER_URL=http://capture:8089
POLL_INTERVAL_SECS=5

# Optional integrations
# BIRDWEATHER_ID=
# HEARTBEAT_URL=
EOF
  info "Created gaia.conf with defaults"
fi

# ── birds manifest.toml (with Zenodo auto-download) ─────────────────────────
MANIFEST="${INSTALL_DIR}/models/birds/manifest.toml"
if [[ -f "$MANIFEST" ]]; then
  warn "models/birds/manifest.toml already exists – skipping"
else
  cat > "$MANIFEST" << 'EOF'
# BirdNET V2.4 – model manifest
# Model files are downloaded automatically from Zenodo on first start.

[model]
name = "BirdNET V2.4"
domain = "birds"
sample_rate = 48000
chunk_duration = 3.0
tflite_file = "BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite"
labels_file = "BirdNET_GLOBAL_6K_V2.4_Model_FP16_Labels.txt"
v1_metadata = false
apply_softmax = false

[metadata_model]
enabled = true
tflite_file = "BirdNET_GLOBAL_6K_V2.4_MData_Model_V2_FP16.tflite"

[language]
dir = "l18n"

# ── Automatic download from Zenodo ────────────────────────────────────
# Set MODEL_VARIANT in gaia.conf to choose: fp32, fp16 (default), int8
[download]
zenodo_record_id = "15050749"
default_variant = "fp16"

[download.variants.fp32]
zenodo_file = "BirdNET_v2.4_tflite.zip"
md5 = "c13f7fd28a5f7a3b092cd993087f93f7"

[download.variants.fp16]
zenodo_file = "BirdNET_v2.4_tflite_fp16.zip"
md5 = "4cd35da63e442d974faf2121700192b5"

[download.variants.int8]
zenodo_file = "BirdNET_v2.4_tflite_int8.zip"
md5 = "69becc3e8eb1c72d1d9dae7f21062c74"
EOF
  info "Created models/birds/manifest.toml (Zenodo auto-download enabled)"
fi

# ── compose.yaml ─────────────────────────────────────────────────────────────
COMPOSE="${INSTALL_DIR}/compose.yaml"
if [[ -f "$COMPOSE" ]]; then
  warn "compose.yaml already exists – skipping"
else
  cat > "$COMPOSE" << EOF
# Gaia Audio Server – generated by install.sh
# Docs: https://github.com/mcguirepr89/BirdNET-Pi/tree/main/gaia-audio-server

services:
  # ── Audio capture ───────────────────────────────────────────────────
  capture:
    image: ${REGISTRY}/gaia-capture
    pull_policy: always
    restart: unless-stopped
    devices:
      - /dev/snd:/dev/snd
    group_add:
      - audio
    # privileged: true   # uncomment if group_add alone is not enough
    volumes:
      - ./gaia.conf:/etc/gaia/gaia.conf:ro
    ports:
      - "8089:8089"

  # ── Model inference & analysis ──────────────────────────────────────
  processing:
    image: ${REGISTRY}/gaia-processing
    pull_policy: always
    restart: unless-stopped
    depends_on:
      - capture
    volumes:
      - ./gaia.conf:/etc/gaia/gaia.conf:ro
      - ./models:/models
      - ./data:/data
    environment:
      - CAPTURE_SERVER_URL=http://capture:8089

  # ── Web dashboard ──────────────────────────────────────────────────
  web:
    image: ${REGISTRY}/gaia-web
    pull_policy: always
    restart: unless-stopped
    depends_on:
      - processing
    volumes:
      - ./data:/data
      - ./backups:/backups
    ports:
      - "3000:3000"
    environment:
      - GAIA_DB_PATH=/data/birds.db
      - GAIA_EXTRACTED_DIR=/data/extracted
EOF
  info "Created compose.yaml"
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
info "Installation complete!"
echo ""
echo "  Directory:  ${INSTALL_DIR}"
echo ""
step "Next steps:"
echo ""
echo "  1. Edit gaia.conf with your location:"
echo "     \$ nano ${INSTALL_DIR}/gaia.conf"
echo "       LATITUDE=9.9346"
echo "       LONGITUDE=-84.0706"
echo ""
echo "  2. Start the stack:"
echo "     \$ cd ${INSTALL_DIR}"
echo "     \$ ${COMPOSE_CMD} up -d"
echo ""
echo "  3. Open the dashboard:"
echo "     http://localhost:3000"
echo ""
echo "  The BirdNET V2.4 model will be downloaded automatically"
echo "  from Zenodo on first start (~53 MB for fp16 variant)."
echo ""
echo "  To import a BirdNET-Pi backup:"
echo "     \$ cp ~/backup.tar ${INSTALL_DIR}/backups/"
echo "     Then visit http://localhost:3000/import"
echo ""
