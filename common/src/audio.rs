//! Audio file reading, resampling, and chunking.
//!
//! Reused from `birdnet-server/src/audio.rs`.
//! Provides WAV I/O, mono conversion, rubato-based resampling, overlapping
//! chunking, and clip extraction.

use anyhow::{Context, Result};
use rubato::{FftFixedIn, Resampler};
use tracing::{debug, info};

/// Read a WAV file, convert to mono f32, resample to `target_sr`, and split
/// into overlapping chunks of `chunk_duration` seconds.
pub fn read_audio(
    path: &std::path::Path,
    target_sr: u32,
    chunk_duration: f64,
    overlap: f64,
) -> Result<Vec<Vec<f32>>> {
    info!("Reading audio: {}", path.display());

    let reader =
        hound::WavReader::open(path).with_context(|| format!("Cannot open {}", path.display()))?;
    let spec = reader.spec();
    let native_sr = spec.sample_rate;
    let n_channels = spec.channels as usize;

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

    // Convert to mono
    let mono: Vec<f32> = if n_channels == 1 {
        samples
    } else {
        samples
            .chunks(n_channels)
            .map(|frame| frame.iter().sum::<f32>() / n_channels as f32)
            .collect()
    };
    debug!("Read {} mono samples at {} Hz", mono.len(), native_sr);

    // Resample if needed
    let resampled = if native_sr == target_sr {
        mono
    } else {
        resample(&mono, native_sr, target_sr)?
    };
    info!(
        "Audio ready: {} samples at {} Hz",
        resampled.len(),
        target_sr
    );

    let chunks = split_signal(&resampled, target_sr, chunk_duration, overlap, 1.5);
    info!("Split into {} chunk(s)", chunks.len());
    Ok(chunks)
}

/// Resample a mono signal from `sr_in` to `sr_out` using rubato.
fn resample(input: &[f32], sr_in: u32, sr_out: u32) -> Result<Vec<f32>> {
    debug!("Resampling {} → {} Hz", sr_in, sr_out);

    let input_f64: Vec<f64> = input.iter().map(|&s| s as f64).collect();

    let chunk_size = 1024;
    let sub_chunks = 2;
    let mut resampler =
        FftFixedIn::<f64>::new(sr_in as usize, sr_out as usize, chunk_size, sub_chunks, 1)
            .context("Failed to create resampler")?;

    let mut output: Vec<f64> = Vec::with_capacity(
        (input_f64.len() as f64 * sr_out as f64 / sr_in as f64) as usize + chunk_size,
    );

    let frames_needed = resampler.input_frames_next();
    let mut pos = 0;
    while pos + frames_needed <= input_f64.len() {
        let chunk = &input_f64[pos..pos + frames_needed];
        let result = resampler
            .process(&[chunk.to_vec()], None)
            .context("Resampler error")?;
        output.extend_from_slice(&result[0]);
        pos += frames_needed;
    }

    if pos < input_f64.len() {
        let remaining = &input_f64[pos..];
        let result = resampler
            .process_partial(Some(&[remaining.to_vec()]), None)
            .context("Resampler partial error")?;
        output.extend_from_slice(&result[0]);
    }

    Ok(output.into_iter().map(|s| s as f32).collect())
}

/// Split a signal into overlapping chunks, zero-padding the last one if
/// shorter than `seconds`.  Chunks shorter than `min_len` are discarded.
pub fn split_signal(
    sig: &[f32],
    rate: u32,
    seconds: f64,
    overlap: f64,
    min_len: f64,
) -> Vec<Vec<f32>> {
    let chunk_samples = (seconds * rate as f64) as usize;
    let step = ((seconds - overlap) * rate as f64) as usize;
    let min_samples = (min_len * rate as f64) as usize;

    let mut chunks = Vec::new();
    let mut i = 0;
    while i < sig.len() {
        let end = (i + chunk_samples).min(sig.len());
        let split = &sig[i..end];

        if split.len() < min_samples {
            break;
        }

        let chunk = if split.len() < chunk_samples {
            let mut padded = vec![0.0f32; chunk_samples];
            padded[..split.len()].copy_from_slice(split);
            padded
        } else {
            split.to_vec()
        };

        chunks.push(chunk);
        i += step;
    }
    chunks
}

/// Extract a section of a WAV file and write it to `out_path`.
pub fn extract_clip(
    in_path: &std::path::Path,
    out_path: &std::path::Path,
    start_sec: f64,
    stop_sec: f64,
) -> Result<()> {
    let reader = hound::WavReader::open(in_path)
        .with_context(|| format!("Cannot open {}", in_path.display()))?;
    let spec = reader.spec();
    let sr = spec.sample_rate as f64;
    let ch = spec.channels as usize;

    let start_sample = (start_sec * sr) as usize * ch;
    let stop_sample = (stop_sec * sr) as usize * ch;

    let all_samples: Vec<i16> = match spec.sample_format {
        hound::SampleFormat::Int => reader.into_samples::<i16>().map(|s| s.unwrap_or(0)).collect(),
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| (s.unwrap_or(0.0) * i16::MAX as f32) as i16)
            .collect(),
    };

    let start = start_sample.min(all_samples.len());
    let stop = stop_sample.min(all_samples.len());
    let clip = &all_samples[start..stop];

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let out_spec = hound::WavSpec {
        channels: spec.channels,
        sample_rate: spec.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(out_path, out_spec)
        .with_context(|| format!("Cannot create {}", out_path.display()))?;
    for &sample in clip {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_signal() {
        // 48 kHz, 3-second chunks, 0 overlap
        let sig = vec![1.0f32; 48000 * 7]; // 7 seconds
        let chunks = split_signal(&sig, 48000, 3.0, 0.0, 1.5);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].len(), 48000 * 3);
    }

    #[test]
    fn test_split_with_overlap() {
        let sig = vec![1.0f32; 48000 * 6]; // 6 seconds
        let chunks = split_signal(&sig, 48000, 3.0, 1.0, 1.5);
        // step = 2s → chunks at 0s, 2s, 4s (last one 2s → padded)
        assert_eq!(chunks.len(), 3);
    }
}
