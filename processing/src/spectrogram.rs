//! Spectrogram generation using FFT.
//!
//! Reused from `birdnet-server/src/spectrogram.rs`.

use std::path::Path;

use anyhow::{Context, Result};
use image::{ImageBuffer, Rgb};
use rustfft::{num_complex::Complex, FftPlanner};
use tracing::debug;

/// Parameters for spectrogram rendering.
pub struct SpectrogramParams {
    pub fft_size: usize,
    pub hop_size: usize,
    /// Maximum frequency to display (Hz). Set to 0 for full range.
    pub max_freq: f64,
    pub width: u32,
    pub height: u32,
}

impl Default for SpectrogramParams {
    fn default() -> Self {
        Self {
            fft_size: 1024,
            hop_size: 512,
            max_freq: 12000.0,
            width: 800,
            height: 256,
        }
    }
}

/// Generate a spectrogram PNG from a mono f32 audio buffer.
pub fn generate(
    samples: &[f32],
    sample_rate: u32,
    out_path: &Path,
    params: &SpectrogramParams,
) -> Result<()> {
    debug!(
        "Generating spectrogram: {} samples, sr={}, → {}",
        samples.len(),
        sample_rate,
        out_path.display()
    );

    let fft_size = params.fft_size;
    let hop = params.hop_size;

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_size);

    let n_frames = if samples.len() > fft_size {
        (samples.len() - fft_size) / hop + 1
    } else {
        1
    };

    let n_bins = fft_size / 2 + 1;

    let max_bin = if params.max_freq > 0.0 {
        ((params.max_freq / sample_rate as f64) * fft_size as f64)
            .ceil() as usize
            + 1
    } else {
        n_bins
    }
    .min(n_bins);

    let hann = hann_window(fft_size);
    let mut magnitude = vec![vec![0.0f32; max_bin]; n_frames];

    for (frame_idx, frame_start) in (0..samples.len().saturating_sub(fft_size))
        .step_by(hop)
        .enumerate()
    {
        if frame_idx >= n_frames {
            break;
        }
        let mut buf: Vec<Complex<f32>> = samples[frame_start..frame_start + fft_size]
            .iter()
            .zip(hann.iter())
            .map(|(&s, &w)| Complex::new(s * w, 0.0))
            .collect();

        fft.process(&mut buf);

        for (bin, val) in buf.iter().take(max_bin).enumerate() {
            let mag = (val.re * val.re + val.im * val.im).sqrt();
            let db = 20.0 * (mag + 1e-10).log10();
            magnitude[frame_idx][bin] = db;
        }
    }

    // Normalise to 0..1
    let global_min = magnitude
        .iter()
        .flat_map(|row| row.iter())
        .cloned()
        .fold(f32::INFINITY, f32::min);
    let global_max = magnitude
        .iter()
        .flat_map(|row| row.iter())
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let range = (global_max - global_min).max(1e-6);

    for row in &mut magnitude {
        for val in row.iter_mut() {
            *val = (*val - global_min) / range;
        }
    }

    // Render to image
    let img_w = params.width;
    let img_h = params.height;
    let mut img = ImageBuffer::<Rgb<u8>, _>::new(img_w, img_h);

    for x in 0..img_w {
        let src_frame = (x as f64 / img_w as f64 * n_frames as f64) as usize;
        let src_frame = src_frame.min(n_frames.saturating_sub(1));

        for y in 0..img_h {
            let bin = ((img_h - 1 - y) as f64 / img_h as f64 * max_bin as f64) as usize;
            let bin = bin.min(max_bin.saturating_sub(1));

            let val = magnitude[src_frame][bin];
            let pixel = colormap(val);
            img.put_pixel(x, y, pixel);
        }
    }

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    img.save(out_path)
        .with_context(|| format!("Cannot write spectrogram to {}", out_path.display()))?;

    debug!("Spectrogram written to {}", out_path.display());
    Ok(())
}

/// Generate a spectrogram directly from a WAV file.
pub fn generate_from_wav(
    wav_path: &Path,
    out_path: &Path,
    params: &SpectrogramParams,
) -> Result<()> {
    let reader = hound::WavReader::open(wav_path)
        .with_context(|| format!("Cannot open {}", wav_path.display()))?;
    let spec = reader.spec();
    let n_ch = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .into_samples::<i32>()
            .map(|s| s.unwrap_or(0) as f32 / i32::MAX as f32)
            .collect(),
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.unwrap_or(0.0))
            .collect(),
    };

    let mono: Vec<f32> = if n_ch == 1 {
        samples
    } else {
        samples
            .chunks(n_ch)
            .map(|frame| frame.iter().sum::<f32>() / n_ch as f32)
            .collect()
    };

    // Downsample to 24 kHz for display
    let target_rate = 24000;
    let mono = if spec.sample_rate != target_rate {
        simple_downsample(&mono, spec.sample_rate, target_rate)
    } else {
        mono
    };

    generate(&mono, target_rate, out_path, params)
}

fn simple_downsample(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate <= to_rate {
        return input.to_vec();
    }
    let ratio = from_rate as f64 / to_rate as f64;
    let out_len = (input.len() as f64 / ratio) as usize;
    (0..out_len)
        .map(|i| {
            let src = (i as f64 * ratio) as usize;
            input.get(src).copied().unwrap_or(0.0)
        })
        .collect()
}

fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = std::f32::consts::PI * 2.0 * i as f32 / (n as f32 - 1.0);
            0.5 * (1.0 - x.cos())
        })
        .collect()
}

fn colormap(val: f32) -> Rgb<u8> {
    let v = val.clamp(0.0, 1.0);
    let r = (255.0 * (3.0 * v - 1.0).clamp(0.0, 1.0)) as u8;
    let g = (255.0
        * (3.0 * v - 0.0)
            .clamp(0.0, 1.0)
            .min((3.0 - 3.0 * v).clamp(0.0, 1.0))) as u8;
    let b = (255.0 * (1.0 - 3.0 * v + 1.0).clamp(0.0, 1.0)) as u8;
    Rgb([r, g, b])
}
