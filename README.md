# Gaia Audio Server

A distributed audio capture, species-detection, and monitoring system split
into three networked containers.  Generalised for multiple model domains –
birds, bats, insects, primates, or any TFLite classifier that operates on
audio chunks.

Evolved from the [BirdNET-Pi](https://github.com/mcguirepr89/BirdNET-Pi) Rust
backend (`birdnet-server/`).

## Architecture

```
┌──────────────────────┐         HTTP/REST          ┌──────────────────────────┐
│   CAPTURE SERVER     │ ◄────────────────────────► │   PROCESSING SERVER      │
│                      │                            │                          │
│  arecord / ffmpeg    │  GET  /api/recordings      │  HTTP client (polling)   │
│  → segmented WAV     │  GET  /api/recordings/:f   │  ↓                       │
│  → axum HTTP server  │  DEL  /api/recordings/:f   │  TFLite inference        │
│                      │  GET  /api/events (SSE)    │  species filtering       │
│  Container: alsa +   │                            │  SQLite DB writes ──────────┐
│  ffmpeg + tiny Rust  │                            │  spectrogram generation  │  │
│  binary              │                            │  BirdWeather / heartbeat │  │
└──────────────────────┘                            └──────────────────────────┘  │
                                                                                  │
                            ┌──────────────────────────┐    reads SQLite (WAL)    │
                            │   WEB DASHBOARD          │ ◄────────────────────────┘
                            │                          │
                            │  Leptos SSR + WASM       │
                            │  Real-time detection feed│
                            │  Calendar / species views│
                            │  iNaturalist images      │
                            │  Dark-themed responsive  │
                            └──────────────────────────┘
```

### Crates

| Crate | Purpose |
|-------|---------|
| **common** (`gaia-common`) | Shared config parsing, detection types, audio I/O, HTTP protocol types |
| **capture** (`gaia-capture`) | Audio capture + HTTP server that serves raw recordings |
| **processing** (`gaia-processing`) | Model inference, analysis pipeline, DB, spectrograms, reporting |
| **web** (`gaia-web`) | Leptos web dashboard – live feed, calendar, species pages, iNaturalist images |

## Multi-Model Support

Each model lives in its own directory with a `manifest.toml`:

```
/models/
├── birds/
│   ├── manifest.toml
│   ├── BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite
│   ├── BirdNET_GLOBAL_6K_V2.4_Model_FP16_Labels.txt
│   └── l18n/
├── bats/
│   ├── manifest.toml
│   ├── BatDetect2.tflite
│   └── labels.txt
└── insects/
    ├── manifest.toml
    └── ...
```

### Model Manifest (`manifest.toml`)

```toml
[model]
name = "BirdNET V2.4"
domain = "birds"
sample_rate = 48000
chunk_duration = 3.0
tflite_file = "BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite"
labels_file = "BirdNET_GLOBAL_6K_V2.4_Model_FP16_Labels.txt"
v1_metadata = false

[metadata_model]
enabled = true
tflite_file = "BirdNET_GLOBAL_6K_V2.4_MData_Model_V2_FP16.tflite"
```

### Automatic Model Download from Zenodo

Manifests can include a `[download]` section that tells the processing server
how to fetch model files from [Zenodo](https://zenodo.org) on first start.
If the expected `tflite_file` is already on disk, no download occurs.

```toml
[download]
zenodo_record_id = "15050749"   # BirdNET V2.4 on Zenodo
default_variant = "fp16"         # used when MODEL_VARIANT is not set

[download.variants.fp32]          # Full precision (~77 MB)
zenodo_file = "BirdNET_v2.4_tflite.zip"
md5 = "c13f7fd28a5f7a3b092cd993087f93f7"

[download.variants.fp16]          # Half precision (~53 MB, default)
zenodo_file = "BirdNET_v2.4_tflite_fp16.zip"
md5 = "4cd35da63e442d974faf2121700192b5"

[download.variants.int8]          # Quantised (~46 MB, ideal for RPi)
zenodo_file = "BirdNET_v2.4_tflite_int8.zip"
md5 = "69becc3e8eb1c72d1d9dae7f21062c74"
```

| Variant | Size | Best for |
|---------|------|----------|
| `fp32` | 77 MB | Powerful servers, highest accuracy |
| `fp16` | 53 MB | Default — good accuracy/size balance |
| `int8` | 46 MB | Raspberry Pi, low-memory devices |

Variant entries can optionally override `tflite_file`, `labels_file`, and
`metadata_tflite_file` when different variants ship different filenames.

See `examples/birds_manifest.toml` for a complete example.

## Configuration

Both servers read the same `birdnet.conf`-style `KEY=VALUE` file
(default: `/etc/gaia/gaia.conf`).

| Key | Default | Used By | Description |
|-----|---------|---------|-------------|
| `LATITUDE` | `-1` | processing | Location latitude |
| `LONGITUDE` | `-1` | processing | Location longitude |
| `CONFIDENCE` | `0.7` | processing | Minimum detection confidence |
| `SENSITIVITY` | `1.25` | processing | Sigmoid sensitivity |
| `OVERLAP` | `0.0` | processing | Chunk overlap (seconds) |
| `RECORDING_LENGTH` | `15` | capture | Segment length (seconds) |
| `CHANNELS` | `1` | capture | Mic channels |
| `REC_CARD` | | capture | ALSA card name |
| `RECS_DIR` | `/data` | both | Base recording directory |
| `EXTRACTED` | `/data/Extracted` | processing | Extracted clip directory |
| `MODEL_DIR` | `/models` | processing | Root model directory (auto-discovers subdirs) |
| `MODEL_VARIANT` | | processing | Model variant: `fp32`, `fp16`, or `int8` (default from manifest) |
| `DATABASE_LANG` | `en` | processing | Language for common names |
| `RTSP_STREAMS` | | capture | Comma-separated RTSP URLs |
| `CAPTURE_LISTEN_ADDR` | `0.0.0.0:8089` | capture | Capture HTTP bind address |
| `CAPTURE_SERVER_URL` | `http://capture:8089` | processing | URL to reach capture server |
| `POLL_INTERVAL_SECS` | `5` | processing | How often to poll for new recordings |
| `BIRDWEATHER_ID` | | processing | BirdWeather station token |
| `HEARTBEAT_URL` | | processing | Uptime heartbeat URL |
| `DB_PATH` | `/data/birds.db` | processing | SQLite database path |

## Building

```bash
# Full workspace
cargo build --release

# Individual crates
cargo build --release -p gaia-capture
cargo build --release -p gaia-processing

# Web dashboard (requires cargo-leptos + wasm32 target)
cargo install cargo-leptos
rustup target add wasm32-unknown-unknown
cd web && cargo leptos build --release
```

## Container Images

```bash
# Build capture image
podman build -f capture/Containerfile -t fede2/gaia-capture .

# Build processing image
podman build -f processing/Containerfile -t fede2/gaia-processing .

# Build web dashboard image
podman build -f web/Containerfile -t fede2/gaia-web .
```

## Running with Podman/Docker Compose

### Quick install

The included `install.sh` creates everything you need:

```bash
# Install to ~/gaia (default)
bash install.sh

# Or specify a custom directory
bash install.sh /opt/gaia

# Or set the registry for custom image names
GAIA_REGISTRY=ghcr.io/myorg bash install.sh
```

The script is **idempotent** – running it again won't overwrite existing
config files.  It creates:

- `compose.yaml` with all three services
- `gaia.conf` with commented defaults
- `models/birds/manifest.toml` with Zenodo auto-download for BirdNET V2.4
- `data/`, `data/extracted/`, and `backups/` directories

### Directory layout

Create a working directory with the following structure before starting:

```
gaia/
├── compose.yaml          # see below
├── gaia.conf             # KEY=VALUE config (see Configuration table above)
├── models/               # model directories with manifest.toml each
│   └── birds/
│       └── manifest.toml # auto-downloads model on first start if [download] is set
├── data/                 # shared volume – DB, recordings, extracted clips
│   ├── birds.db          # SQLite database (created automatically)
│   └── extracted/        # audio clips + spectrograms (created automatically)
└── backups/              # (optional) place BirdNET-Pi .tar backups here for import
```

### `compose.yaml`

```yaml
services:
  # ── Audio capture ───────────────────────────────────────────────────
  capture:
    image: fede2/gaia-capture
    restart: unless-stopped
    devices:
      - /dev/snd:/dev/snd          # ALSA sound devices
    group_add:
      - audio                       # required to access /dev/snd
    # privileged: true              # uncomment if group_add alone is not enough
    volumes:
      - ./gaia.conf:/etc/gaia/gaia.conf:ro
    ports:
      - "8089:8089"

  # ── Model inference & analysis ──────────────────────────────────────
  processing:
    image: fede2/gaia-processing
    restart: unless-stopped
    depends_on:
      - capture
    volumes:
      - ./gaia.conf:/etc/gaia/gaia.conf:ro
      - ./models:/models:ro        # model dirs with manifest.toml
      - ./data:/data               # DB + extracted clips (read-write)
    environment:
      - CAPTURE_SERVER_URL=http://capture:8089

  # ── Web dashboard ──────────────────────────────────────────────────
  web:
    image: fede2/gaia-web
    restart: unless-stopped
    depends_on:
      - processing
    volumes:
      - ./data:/data:ro            # SQLite DB (read-only)
      - ./data/extracted:/data/extracted:ro   # audio clips for playback
      - ./backups:/backups         # BirdNET-Pi backup .tar files for import
    ports:
      - "3000:3000"
    environment:
      - GAIA_DB_PATH=/data/birds.db
      - GAIA_EXTRACTED_DIR=/data/extracted
```

### Starting the stack

```bash
# Start all three services in the background
podman compose up -d

# Follow logs from all services
podman compose logs -f

# Follow logs from a single service
podman compose logs -f processing

# Restart a single service after config changes
podman compose restart processing

# Stop everything
podman compose down
```

### Importing a BirdNET-Pi backup

If you have a BirdNET-Pi backup `.tar` file, place it in the `backups/`
directory and use the web dashboard to import it:

1. Copy the backup into the bind-mounted directory:
   ```bash
   cp ~/backup-20260221T152422.tar ./backups/
   ```
2. Open the dashboard at `http://localhost:3000/import`
3. Enter the **container path**: `/backups/backup-20260221T152422.tar`
4. Click **Analyse Backup** to review the contents (detections, species, files)
5. Click **Import All Data** to import detections into the Gaia DB and extract
   audio clips and spectrograms into `data/extracted/`

### RTSP cameras (no local mic)

If you are using network cameras instead of a local microphone, you can skip
the ALSA device binding and configure RTSP streams in `gaia.conf`:

```
RTSP_STREAMS=rtsp://cam1:554/stream,rtsp://cam2:554/stream
```

Then remove the `devices:` section from the capture service.

### Checking service health

```bash
# Capture server – should return JSON list of recordings
curl http://localhost:8089/api/recordings

# Web dashboard – should return HTML
curl -s http://localhost:3000/ | head -5
```

### Upgrading

```bash
# Pull latest images
podman compose pull

# Recreate containers with new images (data volume is preserved)
podman compose up -d
```

## Tests

```bash
cargo test --workspace
```
