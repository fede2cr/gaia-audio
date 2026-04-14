#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/test_processing_container_e2e.sh [options]

Builds or reuses the Gaia capture/processing images, seeds a real example
recording into the capture node, verifies mDNS discovery, and requires the
processing container to process the file and exit cleanly.

Options:
  --engine <docker|podman>     Container engine to use (auto-detected)
  --capture-image <tag>        Capture image tag (default: gaia-audio-capture:e2e)
  --processing-image <tag>     Processing image tag (default: gaia-audio-processing:e2e)
  --timeout <seconds>          Time allowed after startup/worker-ready (default: 600)
  --startup-timeout <seconds>  Time allowed for first-run downloads/model loading (default: 900)
  --strace                     Run processing under strace and preserve artifacts on failure
  --no-build                   Reuse existing images instead of rebuilding them
  -h, --help                   Show this help
EOF
}

ENGINE=""
CAPTURE_IMAGE="gaia-audio-capture:e2e"
PROCESSING_IMAGE="gaia-audio-processing:e2e"
REDIS_IMAGE="${REDIS_IMAGE:-docker.io/valkey/valkey:8-alpine}"
TIMEOUT_SECS=600
STARTUP_TIMEOUT_SECS="${GAIA_E2E_STARTUP_TIMEOUT_SECS:-900}"
BUILD_IMAGES=1
ENABLE_STRACE=0
KEEP_TMP_DIR=0
LOGS_PID=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine)
      ENGINE="$2"
      shift 2
      ;;
    --capture-image)
      CAPTURE_IMAGE="$2"
      shift 2
      ;;
    --processing-image)
      PROCESSING_IMAGE="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT_SECS="$2"
      shift 2
      ;;
    --startup-timeout)
      STARTUP_TIMEOUT_SECS="$2"
      shift 2
      ;;
    --strace)
      ENABLE_STRACE=1
      shift
      ;;
    --no-build)
      BUILD_IMAGES=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$ENGINE" ]]; then
  if command -v podman >/dev/null 2>&1; then
    ENGINE="podman"
  elif command -v docker >/dev/null 2>&1; then
    ENGINE="docker"
  else
    echo "Neither podman nor docker is available." >&2
    exit 1
  fi
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required for the e2e test harness." >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required for the e2e test harness." >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIXTURE="${ROOT_DIR}/processing/test-fixtures/smoke-test-bird.opus"
if [[ ! -f "$FIXTURE" ]]; then
  echo "Fixture audio not found: $FIXTURE" >&2
  exit 1
fi

if [[ "$BUILD_IMAGES" -eq 1 ]]; then
  echo "==> Building capture image: $CAPTURE_IMAGE"
  "$ENGINE" build -f "$ROOT_DIR/capture/Containerfile" -t "$CAPTURE_IMAGE" "$ROOT_DIR"

  echo "==> Building processing image: $PROCESSING_IMAGE"
  "$ENGINE" build -f "$ROOT_DIR/processing/Containerfile" --target runtime -t "$PROCESSING_IMAGE" "$ROOT_DIR"
fi

TMP_DIR="$(mktemp -d)"
CAPTURE_DATA="$TMP_DIR/capture-data"
PROCESSING_DATA="$TMP_DIR/processing-data"
MODELS_DATA="${GAIA_E2E_MODELS_DIR:-$ROOT_DIR/.e2e-model-cache}"
mkdir -p "$CAPTURE_DATA/StreamData" "$PROCESSING_DATA" "$MODELS_DATA" "$MODELS_DATA/_taxonomy"
if [[ ! -f "$MODELS_DATA/_taxonomy/taxonomy_equivalences.toml" ]]; then
  cp "$ROOT_DIR/examples/taxonomy_equivalences.toml" "$MODELS_DATA/_taxonomy/taxonomy_equivalences.toml"
fi
echo "==> Using model cache: $MODELS_DATA"

SEEDED_FILE="$CAPTURE_DATA/StreamData/2024-06-15-birdnet-10:00:00.wav"
ffmpeg -hide_banner -loglevel error -y \
  -t 6 \
  -i "$FIXTURE" \
  -ac 1 -ar 48000 -c:a pcm_s16le "$SEEDED_FILE"
touch -d '2024-06-15 10:00:00 UTC' "$SEEDED_FILE" 2>/dev/null || true

cat > "$TMP_DIR/capture.conf" <<'EOF'
RECORDING_LENGTH=15
CHANNELS=1
RECS_DIR=/data
CAPTURE_LISTEN_ADDR=0.0.0.0:18089
EOF

cat > "$TMP_DIR/processing.conf" <<'EOF'
LATITUDE=18.50
LONGITUDE=-88.30
CONFIDENCE=0.01
SENSITIVITY=1.25
OVERLAP=0.0
MODEL_DIR=/models
DATABASE_LANG=en
DB_PATH=/data/detections.db
CAPTURE_SERVER_URL=http://127.0.0.1:18089
POLL_INTERVAL_SECS=2
PROCESSING_THREADS=1
RECS_DIR=/data
EXTRACTED=/data/extracted
EOF

REDIS_PORT=6389
CAPTURE_PORT=18089
REDIS_NAME="gaia-e2e-valkey-$$"
CAPTURE_NAME="gaia-e2e-capture-$$"
PROCESSING_NAME="gaia-e2e-processing-$$"
PROCESSING_LOG="$TMP_DIR/processing.log"
CAPTURE_LOG="$TMP_DIR/capture.log"
TAXONOMY_LOG="$TMP_DIR/taxonomy-review.log"

stop_container() {
  local name="$1"
  "$ENGINE" stop -t 5 "$name" >/dev/null 2>&1 || true
  "$ENGINE" kill "$name" >/dev/null 2>&1 || true
  "$ENGINE" rm -f "$name" >/dev/null 2>&1 || true
}

cleanup() {
  set +e
  if [[ -n "${LOGS_PID:-}" ]]; then
    kill "$LOGS_PID" >/dev/null 2>&1 || true
    wait "$LOGS_PID" >/dev/null 2>&1 || true
  fi
  "$ENGINE" logs "$PROCESSING_NAME" > "$PROCESSING_LOG" 2>&1 || true
  "$ENGINE" logs "$CAPTURE_NAME" > "$CAPTURE_LOG" 2>&1 || true
  stop_container "$PROCESSING_NAME"
  stop_container "$CAPTURE_NAME"
  stop_container "$REDIS_NAME"
  if [[ "${KEEP_TMP_DIR:-0}" -eq 0 ]]; then
    rm -rf "$TMP_DIR"
  else
    echo "Artifacts preserved in: $TMP_DIR" >&2
  fi
}

handle_interrupt() {
  KEEP_TMP_DIR=1
  echo >&2
  echo "Interrupted — force stopping e2e containers..." >&2
  stop_container "$PROCESSING_NAME"
  stop_container "$CAPTURE_NAME"
  stop_container "$REDIS_NAME"
  exit 130
}
trap handle_interrupt INT TERM
trap cleanup EXIT

wait_for_processing_ready() {
  local deadline=$((SECONDS + STARTUP_TIMEOUT_SECS))
  while (( SECONDS < deadline )); do
    if grep -Eq 'Worker [0-9]+ ready|Polling [0-9]+ capture server\(s\)' "$PROCESSING_LOG" 2>/dev/null; then
      return 0
    fi

    local status
    status=$("$ENGINE" inspect -f '{{.State.Status}}' "$PROCESSING_NAME" 2>/dev/null || echo "missing")
    case "$status" in
      exited|stopped|dead|missing)
        return 1
        ;;
    esac

    sleep 1
  done

  return 124
}

verify_container_taxonomy_review() {
  local review_name="gaia-e2e-taxonomy-$$"
  echo "==> Verifying taxonomy init inside runtime container"

  if ! timeout --foreground 90s \
    "$ENGINE" run --rm \
      --entrypoint gaia-processing \
      --name "$review_name" \
      --network host \
      -e GAIA_PROCESS_TIMEOUT_SECS=60 \
      -v "$PROCESSING_DATA:/data" \
      -v "$MODELS_DATA:/models" \
      "$PROCESSING_IMAGE" \
      review-taxonomy /models/_taxonomy/taxonomy_equivalences.toml /data/_taxonomy/taxonomy_merged.toml \
      > "$TAXONOMY_LOG" 2>&1
  then
    echo "Container taxonomy review regression failed." >&2
    cat "$TAXONOMY_LOG" >&2 || true
    echo "Hint: if you changed Rust code recently, rebuild the runtime image or omit --no-build." >&2
    stop_container "$review_name"
    exit 1
  fi

  if ! grep -q "taxonomy review OK:" "$TAXONOMY_LOG"; then
    echo "Container taxonomy review did not report success." >&2
    cat "$TAXONOMY_LOG" >&2 || true
    exit 1
  fi
}

verify_container_taxonomy_review

echo "==> Starting Valkey on localhost:${REDIS_PORT}"
"$ENGINE" run -d \
  --name "$REDIS_NAME" \
  --network host \
  "$REDIS_IMAGE" \
  valkey-server --save '' --appendonly no --port "$REDIS_PORT" >/dev/null

echo "==> Starting capture container with seeded example audio"
"$ENGINE" run -d \
  --name "$CAPTURE_NAME" \
  --network host \
  -e GAIA_SKIP_CAPTURE=1 \
  -v "$CAPTURE_DATA:/data" \
  -v "$TMP_DIR/capture.conf:/etc/gaia/gaia.conf:ro" \
  "$CAPTURE_IMAGE" \
  /etc/gaia/gaia.conf >/dev/null

for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${CAPTURE_PORT}/api/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -fsS "http://127.0.0.1:${CAPTURE_PORT}/api/health" >/dev/null 2>&1; then
  echo "Capture health endpoint never became ready." >&2
  "$ENGINE" logs "$CAPTURE_NAME" || true
  exit 1
fi

echo "==> Starting processing container and waiting for clean exit"
TOTAL_TIMEOUT_SECS=$((STARTUP_TIMEOUT_SECS + TIMEOUT_SECS + 30))
"$ENGINE" run -d \
  --name "$PROCESSING_NAME" \
  --network host \
  -e REDIS_URL="redis://127.0.0.1:${REDIS_PORT}" \
  -e GAIA_EXIT_AFTER_ONE_BATCH=1 \
  -e GAIA_SHUTDOWN_GRACE_SECS=15 \
  -e GAIA_PROCESS_TIMEOUT_SECS="$TOTAL_TIMEOUT_SECS" \
  -e GAIA_FORCE_CPU_ORT=1 \
  -e GAIA_DEBUG_ANALYSIS=1 \
  -e GAIA_STRACE="${ENABLE_STRACE:-0}" \
  -v "$PROCESSING_DATA:/data" \
  -v "$MODELS_DATA:/models" \
  -v "$TMP_DIR/processing.conf:/etc/gaia/gaia.conf:ro" \
  "$PROCESSING_IMAGE" \
  /etc/gaia/gaia.conf >/dev/null

"$ENGINE" logs -f "$PROCESSING_NAME" 2>&1 | tee "$PROCESSING_LOG" &
LOGS_PID=$!

processing_exit_code=""
echo "==> Waiting for processing startup/model warm-up (up to ${STARTUP_TIMEOUT_SECS}s)"
ready_rc=0
wait_for_processing_ready || ready_rc=$?
if [[ "$ready_rc" != "0" ]]; then
  KEEP_TMP_DIR=1
  if [[ "$ready_rc" == "124" ]]; then
    echo "Processing container did not finish startup within ${STARTUP_TIMEOUT_SECS}s." >&2
  else
    echo "Processing container exited before reaching worker-ready state." >&2
  fi
  stop_container "$PROCESSING_NAME"
  processing_exit_code=124
else
  echo "==> Processing worker ready; waiting up to ${TIMEOUT_SECS}s for the seeded batch to finish"
  start_ts=$(date +%s)
  while :; do
    status=$("$ENGINE" inspect -f '{{.State.Status}}' "$PROCESSING_NAME" 2>/dev/null || echo "missing")
    case "$status" in
      running|created)
        now=$(date +%s)
        if (( now - start_ts >= TIMEOUT_SECS )); then
          echo "Processing container did not exit successfully within ${TIMEOUT_SECS}s after startup." >&2
          KEEP_TMP_DIR=1
          stop_container "$PROCESSING_NAME"
          processing_exit_code=124
          break
        fi
        sleep 1
        ;;
      exited|stopped)
        processing_exit_code=$("$ENGINE" inspect -f '{{.State.ExitCode}}' "$PROCESSING_NAME" 2>/dev/null || echo 1)
        break
        ;;
      *)
        processing_exit_code=1
        break
        ;;
    esac
  done
fi

if [[ -n "${LOGS_PID:-}" ]]; then
  wait "$LOGS_PID" 2>/dev/null || true
fi

if [[ "$processing_exit_code" != "0" ]]; then
  echo
  echo "--- capture logs ---" >&2
  "$ENGINE" logs "$CAPTURE_NAME" || true
  if [[ "${ENABLE_STRACE:-0}" -eq 1 ]]; then
    echo
    echo "--- strace tail ---" >&2
    find "$PROCESSING_DATA" -type f -name 'gaia-processing.strace*' -print -exec tail -n 80 {} \; 2>/dev/null || true
    KEEP_TMP_DIR=1
  fi
  exit 1
fi

if ! grep -q "mDNS discovered" "$PROCESSING_LOG"; then
  echo "mDNS discovery was not observed in processing logs." >&2
  cat "$PROCESSING_LOG" >&2
  exit 1
fi

if ! grep -q "taxonomy init done" "$PROCESSING_LOG"; then
  echo "Processing did not complete taxonomy initialization during the e2e run." >&2
  cat "$PROCESSING_LOG" >&2
  exit 1
fi

if ! grep -q "shared analysis context ready" "$PROCESSING_LOG"; then
  echo "Processing did not finish building the shared analysis context." >&2
  cat "$PROCESSING_LOG" >&2
  exit 1
fi

if ! grep -q "Analysis complete:" "$PROCESSING_LOG"; then
  echo "Processing did not complete analysis of the seeded recording." >&2
  cat "$PROCESSING_LOG" >&2
  exit 1
fi

for model_name in "Google Perch 2.0" "BirdNET+ V3.0" "BirdNET V2.4" "BatDetect2"; do
  if ! grep -q "Running analysis with model: ${model_name}" "$PROCESSING_LOG"; then
    echo "Expected model did not participate in the e2e analysis: ${model_name}" >&2
    cat "$PROCESSING_LOG" >&2
    exit 1
  fi
done

remaining_files=$(find "$CAPTURE_DATA/StreamData" -maxdepth 1 -type f \( -name '*.wav' -o -name '*.opus' \) | wc -l | tr -d ' ')
if [[ "$remaining_files" != "0" ]]; then
  echo "Seeded recording was not deleted from the capture queue." >&2
  find "$CAPTURE_DATA/StreamData" -maxdepth 1 -type f >&2 || true
  exit 1
fi

echo "✅ Container e2e passed: capture served seeded audio, processing discovered it via mDNS, processed it, and exited cleanly."
