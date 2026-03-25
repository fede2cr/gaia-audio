//! Lightweight disk-usage helper (Linux / macOS).
//!
//! Calls `df` on the target path and parses the output.  This avoids
//! pulling in `nix` or `libc` just for `statvfs`.

use std::path::Path;
use std::process::Command;
use std::time::{Duration, SystemTime};

/// Result summary for an emergency WAV → Opus recode sweep.
#[derive(Debug, Default, Clone, Copy)]
pub struct RecodeSummary {
    pub scanned_wav: usize,
    pub converted: usize,
    pub freed_bytes: u64,
}

/// Human-readable disk space summary for the filesystem containing `path`.
///
/// Returns a formatted string like:
///   "Used 12.3 GB / 29.1 GB (42%), 16.8 GB available"
pub fn summary(path: &Path) -> Option<String> {
    // `df -B1 --output=used,avail,size <path>` prints bytes.
    let output = Command::new("df")
        .args(["-B1", "--output=used,avail,size"])
        .arg(path)
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.lines().nth(1)?; // skip header
    let cols: Vec<&str> = line.split_whitespace().collect();
    if cols.len() < 3 {
        return None;
    }
    let used: f64 = cols[0].parse().ok()?;
    let avail: f64 = cols[1].parse().ok()?;
    let total: f64 = cols[2].parse().ok()?;
    let pct = if total > 0.0 { used / total * 100.0 } else { 0.0 };

    Some(format!(
        "Used {:.1} GB / {:.1} GB ({:.1}%), {:.1} GB available",
        used / 1e9,
        total / 1e9,
        pct,
        avail / 1e9,
    ))
}

/// Return the disk usage percentage (0–100) for the filesystem that
/// contains `path`.  Returns `None` when the check cannot be performed.
pub fn usage_pct(path: &Path) -> Option<f64> {
    // `df --output=pcent <path>` prints e.g.:
    //   Use%
    //    42%
    let output = Command::new("df")
        .args(["--output=pcent"])
        .arg(path)
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Take the last non-empty line, strip the '%' and whitespace.
    stdout
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .and_then(|l| l.trim().trim_end_matches('%').trim().parse::<f64>().ok())
}

/// Recode settled `.wav` files in `dir` to `.opus` to free disk space.
///
/// Only files older than `min_age` are converted to avoid touching
/// segments that ffmpeg may still be writing.
pub fn recode_wav_to_opus(dir: &Path, min_age: Duration) -> RecodeSummary {
    let mut out = RecodeSummary::default();
    let now = SystemTime::now();

    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return out,
    };

    for entry in entries.flatten() {
        let wav_path = entry.path();
        if wav_path.extension().and_then(|e| e.to_str()) != Some("wav") {
            continue;
        }
        out.scanned_wav += 1;

        let meta = match std::fs::metadata(&wav_path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        if !meta.is_file() || meta.len() == 0 {
            continue;
        }

        let modified = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        if now.duration_since(modified).unwrap_or_default() < min_age {
            continue;
        }

        let src_size = meta.len();
        let opus_path = wav_path.with_extension("opus");
        let tmp_opus = wav_path.with_extension("opus.tmp");

        let status = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-i",
                wav_path.to_string_lossy().as_ref(),
                "-c:a",
                "libopus",
                "-b:a",
                "96k",
                tmp_opus.to_string_lossy().as_ref(),
            ])
            .status();

        match status {
            Ok(s) if s.success() => {
                let dst_meta = match std::fs::metadata(&tmp_opus) {
                    Ok(m) if m.len() > 0 => m,
                    _ => {
                        let _ = std::fs::remove_file(&tmp_opus);
                        continue;
                    }
                };

                if std::fs::rename(&tmp_opus, &opus_path).is_err() {
                    let _ = std::fs::remove_file(&tmp_opus);
                    continue;
                }
                if std::fs::remove_file(&wav_path).is_err() {
                    let _ = std::fs::remove_file(&opus_path);
                    continue;
                }

                out.converted += 1;
                let dst_size = dst_meta.len();
                if src_size > dst_size {
                    out.freed_bytes += src_size - dst_size;
                }
            }
            _ => {
                let _ = std::fs::remove_file(&tmp_opus);
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_test() {
        // Should succeed for the root filesystem at least.
        let pct = usage_pct(Path::new("/"));
        assert!(pct.is_some(), "df should work on /");
        let v = pct.unwrap();
        assert!((0.0..=100.0).contains(&v), "percentage out of range: {v}");
    }
}
