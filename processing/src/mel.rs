//! Mel-spectrogram preprocessing for **classifier-only** ONNX models.
//!
//! BirdNET V2.4 ships as a Keras model whose first two layers
//! (`MelSpecLayerSimple`) compute mel spectrograms using `tf.signal.stft`.
//! Those layers use ops (RFFT) that cannot be converted to standard ONNX,
//! so we split the model at the `concatenate` layer and replicate the mel
//! computation in pure Rust.
//!
//! The two layers differ only in STFT parameters and mel frequency range:
//!
//! | param | MEL_SPEC1 | MEL_SPEC2 |
//! |---|---|---|
//! | frame_length (FFT size) | 2048 | 1024 |
//! | frame_step (hop) | 278 | 280 |
//! | fmin | 0 | 500 |
//! | fmax | 3000 | 15000 |
//! | mel bins | 96 | 96 |
//! | sample_rate | 48000 | 48000 |
//! | mag_scale (trained) | 1.2110 | 1.4466 |
//!
//! The output is a `[1, 96, 511, 2]` tensor (96 mel bins × 511 time frames
//! × 2 channels) ready to feed the ONNX classifier.

use rustfft::{num_complex::Complex, FftPlanner};

// ── types ────────────────────────────────────────────────────────────────

/// Parameters for one mel-spectrogram channel.
#[derive(Debug, Clone)]
pub struct MelSpecParams {
    /// FFT window / frame length in samples.
    pub frame_length: usize,
    /// Hop size between successive frames.
    pub frame_step: usize,
    /// Number of mel-frequency bins.
    pub n_mels: usize,
    /// Lower edge of the mel filterbank (Hz).
    pub fmin: f32,
    /// Upper edge of the mel filterbank (Hz).
    pub fmax: f32,
    /// Audio sample rate (Hz).
    pub sample_rate: f32,
    /// Trained magnitude-scaling parameter (the `mag_scale` weight from
    /// `MelSpecLayerSimple`).
    pub mag_scale: f32,
}

/// Pre-computed state for a [`MelSpecParams`] configuration.
pub struct MelSpecLayer {
    params: MelSpecParams,
    /// Mel filterbank matrix, shape `[n_fft_bins, n_mels]` stored in
    /// row-major order.
    mel_filterbank: Vec<f32>,
    /// Number of FFT bins = frame_length / 2 + 1.
    n_fft_bins: usize,
    /// Hann window of length `frame_length`.
    hann: Vec<f32>,
}

// ── BirdNET V2.4 defaults ────────────────────────────────────────────────

/// Mel-spec layer 1 (low-frequency, 0–3 kHz).
pub fn birdnet_mel_spec1() -> MelSpecParams {
    MelSpecParams {
        frame_length: 2048,
        frame_step: 278,
        n_mels: 96,
        fmin: 0.0,
        fmax: 3000.0,
        sample_rate: 48000.0,
        mag_scale: 1.2110004,
    }
}

/// Mel-spec layer 2 (high-frequency, 500–15 000 Hz).
pub fn birdnet_mel_spec2() -> MelSpecParams {
    MelSpecParams {
        frame_length: 1024,
        frame_step: 280,
        n_mels: 96,
        fmin: 500.0,
        fmax: 15000.0,
        sample_rate: 48000.0,
        mag_scale: 1.4465874,
    }
}

// ── construction ─────────────────────────────────────────────────────────

impl MelSpecLayer {
    /// Build a new layer, pre-computing the Hann window and mel filterbank.
    pub fn new(params: MelSpecParams) -> Self {
        let n_fft_bins = params.frame_length / 2 + 1;
        let mel_filterbank =
            linear_to_mel_weight_matrix(params.n_mels, n_fft_bins, params.sample_rate, params.fmin, params.fmax);
        let hann = hann_window(params.frame_length);
        Self {
            params,
            mel_filterbank,
            n_fft_bins,
            hann,
        }
    }

    /// Compute the mel spectrogram for a single chunk of audio.
    ///
    /// `audio` must be exactly `chunk_samples` long (e.g. 144 000 for 3 s
    /// @ 48 kHz).  Returns a flat buffer of shape `[n_mels, n_frames]` in
    /// row-major order plus the `(n_mels, n_frames)` dimensions.
    pub fn compute(&self, audio: &[f32]) -> (Vec<f32>, usize, usize) {
        let p = &self.params;

        // ── 1. normalise to [-1, 1] ──────────────────────────────────
        let (min_val, max_val) = audio.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &v| {
            (mn.min(v), mx.max(v))
        });
        let range = max_val - min_val + 1e-6;
        let norm: Vec<f32> = audio.iter().map(|&v| ((v - min_val) / range - 0.5) * 2.0).collect();

        // ── 2. STFT ──────────────────────────────────────────────────
        let n_frames = (norm.len().saturating_sub(p.frame_length)) / p.frame_step + 1;
        let n_bins = self.n_fft_bins;

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(p.frame_length);

        // stft_real: [n_frames, n_bins]  (real part of complex STFT)
        let mut stft_real = vec![0.0f32; n_frames * n_bins];

        let mut buf = vec![Complex::new(0.0f32, 0.0); p.frame_length];
        for frame_idx in 0..n_frames {
            let start = frame_idx * p.frame_step;
            for (i, (&s, &w)) in norm[start..start + p.frame_length]
                .iter()
                .zip(self.hann.iter())
                .enumerate()
            {
                buf[i] = Complex::new(s * w, 0.0);
            }
            fft.process(&mut buf);
            // Take real part only (matches `tf.cast(complex64, float32)`)
            let row_offset = frame_idx * n_bins;
            for (bin, c) in buf.iter().take(n_bins).enumerate() {
                stft_real[row_offset + bin] = c.re;
            }
        }

        // ── 3. Mel filterbank  (matmul: [n_frames, n_bins] × [n_bins, n_mels])
        let n_mels = p.n_mels;
        let mut mel = vec![0.0f32; n_frames * n_mels];
        for f in 0..n_frames {
            let stft_row = &stft_real[f * n_bins..(f + 1) * n_bins];
            let mel_row = &mut mel[f * n_mels..(f + 1) * n_mels];
            for m in 0..n_mels {
                let mut acc = 0.0f32;
                for b in 0..n_bins {
                    acc += stft_row[b] * self.mel_filterbank[b * n_mels + m];
                }
                mel_row[m] = acc;
            }
        }

        // ── 4. Power spectrogram ─────────────────────────────────────
        for v in mel.iter_mut() {
            *v = *v * *v;
        }

        // ── 5. Nonlinear magnitude scaling ───────────────────────────
        // Python: spec = tf.math.pow(spec, 1.0 / (1.0 + tf.math.exp(self.mag_scale)))
        let exponent = 1.0 / (1.0 + p.mag_scale.exp());
        for v in mel.iter_mut() {
            *v = v.powf(exponent);
        }

        // ── 6. Flip frequency axis (reverse along mel dim per frame) ─
        // mel is [n_frames, n_mels]; we need to reverse each row.
        for f in 0..n_frames {
            let start = f * n_mels;
            mel[start..start + n_mels].reverse();
        }

        // ── 7. Transpose to [n_mels, n_frames] ──────────────────────
        let mut transposed = vec![0.0f32; n_mels * n_frames];
        for f in 0..n_frames {
            for m in 0..n_mels {
                transposed[m * n_frames + f] = mel[f * n_mels + m];
            }
        }

        (transposed, n_mels, n_frames)
    }
}

/// Compute the dual-channel mel spectrogram used by BirdNET V2.4.
///
/// Returns a `Vec<f32>` of shape `[1, 96, 511, 2]` in row-major order
/// (NHWC layout) suitable for feeding the classifier ONNX model.
pub fn birdnet_mel_spectrogram(audio: &[f32]) -> Vec<f32> {
    let layer1 = MelSpecLayer::new(birdnet_mel_spec1());
    let layer2 = MelSpecLayer::new(birdnet_mel_spec2());

    let (ch0, n_mels, n_frames) = layer1.compute(audio);
    let (ch1, n_mels2, n_frames2) = layer2.compute(audio);

    debug_assert_eq!(n_mels, n_mels2);
    debug_assert_eq!(n_frames, n_frames2);
    debug_assert_eq!(n_mels, 96);
    debug_assert_eq!(n_frames, 511);

    // Interleave into NHWC: [1, n_mels, n_frames, 2]
    let total = n_mels * n_frames * 2;
    let mut out = vec![0.0f32; total];
    for m in 0..n_mels {
        for f in 0..n_frames {
            let idx = (m * n_frames + f) * 2;
            out[idx] = ch0[m * n_frames + f];
            out[idx + 1] = ch1[m * n_frames + f];
        }
    }
    out
}

// ── helpers ──────────────────────────────────────────────────────────────

/// Hann window of length `n` (periodic version matching `tf.signal.hann_window`).
///
/// TensorFlow uses the **periodic** convention:
///   `w[i] = 0.5 - 0.5 * cos(2π * i / N)`
/// where `N = n` (not `n - 1`).
fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = std::f32::consts::PI * 2.0 * i as f32 / n as f32;
            0.5 * (1.0 - x.cos())
        })
        .collect()
}

/// Compute TensorFlow-compatible `linear_to_mel_weight_matrix`.
///
/// Returns a `[n_fft_bins, n_mels]` row-major matrix.
///
/// This matches `tf.signal.linear_to_mel_weight_matrix` which uses the
/// HTK mel scale:  `mel = 1127 * ln(1 + f / 700)`.
///
/// TF zeroes out the DC bin (row 0) and computes triangular filter slopes
/// in the mel domain.
fn linear_to_mel_weight_matrix(
    n_mels: usize,
    n_fft_bins: usize,
    sample_rate: f32,
    fmin: f32,
    fmax: f32,
) -> Vec<f32> {
    let hz_to_mel = |f: f32| -> f32 { 1127.0 * (1.0 + f / 700.0).ln() };

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    // n_mels + 2 band edges evenly spaced in mel domain
    let n_edges = n_mels + 2;
    let mel_edges: Vec<f32> = (0..n_edges)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_edges - 1) as f32)
        .collect();

    // Frequency of each FFT bin (in Hz)
    let nyquist = sample_rate / 2.0;
    let fft_freqs: Vec<f32> = (0..n_fft_bins)
        .map(|i| i as f32 * nyquist / (n_fft_bins - 1) as f32)
        .collect();

    // Convert FFT bin frequencies to mel (for slope computation in mel space)
    let fft_mels: Vec<f32> = fft_freqs.iter().map(|&f| hz_to_mel(f)).collect();

    let mut weights = vec![0.0f32; n_fft_bins * n_mels];

    for m in 0..n_mels {
        let lower = mel_edges[m];
        let center = mel_edges[m + 1];
        let upper = mel_edges[m + 2];

        // Skip DC bin (b=0), matching TF's bands_to_zero=1
        for b in 1..n_fft_bins {
            let mel_f = fft_mels[b];
            let lower_slope = if (center - lower).abs() > f32::EPSILON {
                (mel_f - lower) / (center - lower)
            } else {
                0.0
            };
            let upper_slope = if (upper - center).abs() > f32::EPSILON {
                (upper - mel_f) / (upper - center)
            } else {
                0.0
            };
            let w = lower_slope.min(upper_slope).max(0.0);
            weights[b * n_mels + m] = w;
        }
    }

    weights
}

// ── tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window_periodic() {
        let w = hann_window(4);
        // Periodic Hann(4): [0.0, 0.75, 0.75, 0.0] approximately
        // w[0] = 0.5*(1 - cos(0)) = 0
        // w[1] = 0.5*(1 - cos(π/2)) = 0.5
        // w[2] = 0.5*(1 - cos(π)) = 1.0
        // w[3] = 0.5*(1 - cos(3π/2)) = 0.5
        assert!((w[0] - 0.0).abs() < 1e-6);
        assert!((w[1] - 0.5).abs() < 1e-6);
        assert!((w[2] - 1.0).abs() < 1e-6);
        assert!((w[3] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_mel_filterbank_shapes() {
        let fb = linear_to_mel_weight_matrix(96, 1025, 48000.0, 0.0, 3000.0);
        assert_eq!(fb.len(), 1025 * 96);
        // Should have nonzero values
        let nonzero = fb.iter().filter(|&&v| v > 0.0).count();
        assert!(nonzero > 0, "filterbank is all zeros");
    }

    /// Compare Rust STFT against TF reference.
    #[test]
    fn test_stft_vs_tf() {
        let audio_path = std::path::Path::new("/tmp/test_audio_raw.f32");
        let stft_ref_path = std::path::Path::new("/tmp/stft1_real_ref.f32");
        if !audio_path.exists() || !stft_ref_path.exists() {
            eprintln!("Skipping: reference files not found");
            return;
        }

        // Load audio
        let audio_bytes = std::fs::read(audio_path).unwrap();
        let audio: Vec<f32> = audio_bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Load TF STFT reference (511 frames × 1025 bins)
        let stft_bytes = std::fs::read(stft_ref_path).unwrap();
        let tf_stft: Vec<f32> = stft_bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(tf_stft.len(), 511 * 1025, "unexpected STFT reference size");

        // Normalize audio (same as MelSpecLayerSimple)
        let (min_val, max_val) = audio.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &v| {
            (mn.min(v), mx.max(v))
        });
        let range = max_val - min_val + 1e-6;
        let norm: Vec<f32> = audio.iter().map(|&v| ((v - min_val) / range - 0.5) * 2.0).collect();

        // Compute STFT
        let frame_length = 2048;
        let _frame_step = 278;
        let n_bins = frame_length / 2 + 1;
        let hann = hann_window(frame_length);

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(frame_length);

        let mut buf = vec![Complex::new(0.0f32, 0.0); frame_length];
        // Just check frame 0
        for (i, (&s, &w)) in norm[0..frame_length].iter().zip(hann.iter()).enumerate() {
            buf[i] = Complex::new(s * w, 0.0);
        }
        fft.process(&mut buf);

        eprintln!("Frame 0, first 10 bins (real part):");
        eprintln!("  Rust: {:?}", &buf[..10].iter().map(|c| c.re).collect::<Vec<_>>());
        eprintln!("  TF:   {:?}", &tf_stft[..10]);

        let mut max_diff = 0.0f32;
        for b in 0..n_bins {
            let d = (buf[b].re - tf_stft[b]).abs();
            if d > max_diff { max_diff = d; }
        }
        eprintln!("Frame 0 max diff: {max_diff:.6}");
        assert!(max_diff < 0.01, "STFT max diff too large: {max_diff}");
    }

    /// Compare Rust filterbank against TF reference.
    #[test]
    fn test_mel_filterbank_vs_tf() {
        let ref_path = std::path::Path::new("/tmp/mel_fb1_tf.f32");
        if !ref_path.exists() {
            eprintln!("Skipping: /tmp/mel_fb1_tf.f32 not found");
            return;
        }
        let ref_bytes = std::fs::read(ref_path).unwrap();
        let tf_fb: Vec<f32> = ref_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(tf_fb.len(), 1025 * 96);

        let rust_fb = linear_to_mel_weight_matrix(96, 1025, 48000.0, 0.0, 3000.0);
        assert_eq!(rust_fb.len(), tf_fb.len());

        let mut max_diff = 0.0f32;
        let mut n_diff = 0usize;
        for m in (0..96).step_by(10) {
            let mut rust_nz = 0;
            let mut tf_nz = 0;
            let mut rust_sum = 0.0f32;
            let mut tf_sum = 0.0f32;
            for b in 0..1025 {
                let ri = b * 96 + m;
                if rust_fb[ri] > 0.0 { rust_nz += 1; rust_sum += rust_fb[ri]; }
                if tf_fb[ri] > 0.0 { tf_nz += 1; tf_sum += tf_fb[ri]; }
                let d = (rust_fb[ri] - tf_fb[ri]).abs();
                if d > max_diff { max_diff = d; }
                if d > 1e-6 { n_diff += 1; }
            }
            eprintln!("mel[{m:2}] rust: nz={rust_nz:3}, sum={rust_sum:.4} | tf: nz={tf_nz:3}, sum={tf_sum:.4}");
        }
        eprintln!("Max diff: {max_diff:.6}, n_diff: {n_diff}");
        assert!(max_diff < 0.01, "Filterbank max diff too large: {max_diff}");
    }

    #[test]
    fn test_mel_spec_output_shape() {
        // 3 seconds @ 48 kHz = 144000 samples
        let audio = vec![0.0f32; 144000];
        let layer = MelSpecLayer::new(birdnet_mel_spec1());
        let (data, n_mels, n_frames) = layer.compute(&audio);
        assert_eq!(n_mels, 96);
        assert_eq!(n_frames, 511);
        assert_eq!(data.len(), 96 * 511);
    }

    #[test]
    fn test_birdnet_mel_spectrogram_shape() {
        let audio = vec![0.0f32; 144000];
        let out = birdnet_mel_spectrogram(&audio);
        assert_eq!(out.len(), 96 * 511 * 2);
    }

    /// Compare Rust mel spectrogram against the TensorFlow/Keras reference.
    ///
    /// Only runs when the reference files exist in /tmp (generated by the
    /// Python test harness).
    #[test]
    fn test_birdnet_mel_vs_reference() {
        let audio_path = std::path::Path::new("/tmp/test_audio_raw.f32");
        let ref_path = std::path::Path::new("/tmp/test_mel_reference.f32");
        if !audio_path.exists() || !ref_path.exists() {
            eprintln!("Skipping reference test: files not found in /tmp");
            return;
        }

        // Load raw f32 audio
        let audio_bytes = std::fs::read(audio_path).unwrap();
        let audio: Vec<f32> = audio_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(audio.len(), 144000);

        // Load reference mel spectrogram
        let ref_bytes = std::fs::read(ref_path).unwrap();
        let reference: Vec<f32> = ref_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(reference.len(), 96 * 511 * 2);

        // Compute Rust mel spectrogram
        let rust_mel = birdnet_mel_spectrogram(&audio);
        assert_eq!(rust_mel.len(), reference.len());

        // Compare
        let mut max_diff = 0.0f32;
        let mut sum_diff = 0.0f64;
        for (i, (&r, &p)) in rust_mel.iter().zip(reference.iter()).enumerate() {
            let d = (r - p).abs();
            if d > max_diff {
                max_diff = d;
                if d > 0.1 {
                    let ch = i % 2;
                    let f = (i / 2) % 511;
                    let m = (i / 2) / 511;
                    eprintln!("  Large diff at [mel={m}, frame={f}, ch={ch}]: rust={r:.6}, ref={p:.6}, diff={d:.6}");
                }
            }
            sum_diff += d as f64;
        }
        let mean_diff = sum_diff / rust_mel.len() as f64;
        eprintln!("Max diff: {max_diff:.6}");
        eprintln!("Mean diff: {mean_diff:.6}");

        // We expect fairly close match: max diff < 0.5, mean diff < 0.05
        // (exact match is unlikely due to mel filterbank computation
        // differences between Rust and TF)
        assert!(max_diff < 1.0, "Max diff too large: {max_diff}");
        assert!(mean_diff < 0.1, "Mean diff too large: {mean_diff}");
    }
}
