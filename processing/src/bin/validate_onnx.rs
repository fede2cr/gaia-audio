use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use tract_onnx::prelude::Framework;
use tract_onnx::prelude::InferenceModelExt as _;

fn main() {
    if let Err(e) = run() {
        eprintln!("  FAIL ✗  {e:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!(
            "Usage: validate_onnx <model.onnx> --shape 1,96,511,2 [--skip-ort-checks] [--skip-tract-checks] [--timeout-secs 120]"
        );
        std::process::exit(2);
    }

    let opts = Cli::parse(&args)?;

    if opts.worker {
        return run_inner(&opts);
    }

    run_with_timeout(&opts, &args)
}

struct Cli {
    model_path: String,
    shape: Vec<usize>,
    skip_ort: bool,
    skip_tract: bool,
    timeout_secs: u64,
    worker: bool,
}

impl Cli {
    fn parse(args: &[String]) -> Result<Self> {
        if args.len() < 3 {
            eprintln!(
                "Usage: validate_onnx <model.onnx> --shape 1,96,511,2 [--skip-ort-checks] [--skip-tract-checks] [--timeout-secs 120]"
            );
            std::process::exit(2);
        }

        let model_path = args[0].clone();
        let mut shape: Option<Vec<usize>> = None;
        let mut skip_ort = false;
        let mut skip_tract = false;
        let mut timeout_secs = 120u64;
        let mut worker = false;

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--shape" => {
                    let raw = args
                        .get(i + 1)
                        .context("--shape requires a value (e.g. 1,96,511,2)")?;
                    shape = Some(parse_shape(raw)?);
                    i += 2;
                }
                "--skip-tract-checks" => {
                    skip_tract = true;
                    i += 1;
                }
                "--skip-ort-checks" => {
                    skip_ort = true;
                    i += 1;
                }
                "--timeout-secs" => {
                    let raw = args
                        .get(i + 1)
                        .context("--timeout-secs requires a value (e.g. 120)")?;
                    timeout_secs = raw
                        .parse::<u64>()
                        .context("--timeout-secs must be a positive integer")?;
                    if timeout_secs == 0 {
                        bail!("--timeout-secs must be > 0");
                    }
                    i += 2;
                }
                "--worker" => {
                    worker = true;
                    i += 1;
                }
                other => bail!("Unknown argument: {other}"),
            }
        }

        let shape = shape.context("Missing required --shape argument")?;
        if skip_ort && skip_tract {
            bail!("Cannot skip both ORT and tract checks");
        }

        Ok(Self {
            model_path,
            shape,
            skip_ort,
            skip_tract,
            timeout_secs,
            worker,
        })
    }
}

fn run_with_timeout(opts: &Cli, original_args: &[String]) -> Result<()> {
    let mut child = Command::new(std::env::current_exe().context("Cannot resolve current executable")?)
        .args(original_args)
        .arg("--worker")
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .context("Failed to spawn validate_onnx worker process")?;

    let start = Instant::now();
    let timeout = std::time::Duration::from_secs(opts.timeout_secs);

    loop {
        if let Some(status) = child.try_wait().context("Failed to check worker status")? {
            if status.success() {
                return Ok(());
            }
            bail!(
                "Validation worker exited with status {} for {}",
                status,
                opts.model_path
            );
        }

        if start.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            bail!(
                "Validation timed out after {}s for {}. Likely ORT/tract stall in model load or inference.",
                opts.timeout_secs,
                opts.model_path
            );
        }

        std::thread::sleep(std::time::Duration::from_millis(200));
    }
}

fn run_inner(opts: &Cli) -> Result<()> {
    println!(
        "Validating {} (expected input: {:?})",
        opts.model_path, opts.shape
    );

    if opts.skip_ort {
        println!("  ORT checks skipped (--skip-ort-checks)");
    } else {
        init_ort()?;
        validate_ort(&opts.model_path, &opts.shape)?;
    }

    if opts.skip_tract {
        println!("  tract-onnx compatibility checks skipped (--skip-tract-checks)");
    } else {
        validate_tract(&opts.model_path)?;
    }

    println!("  PASS ✓");
    Ok(())
}

fn parse_shape(raw: &str) -> Result<Vec<usize>> {
    let dims: Result<Vec<_>, _> = raw
        .split(',')
        .map(|d| d.trim().parse::<usize>())
        .collect();
    let dims = dims.context("Shape must be comma-separated positive integers")?;
    if dims.is_empty() {
        bail!("Shape cannot be empty");
    }
    if dims.iter().any(|&d| d == 0) {
        bail!("Shape dimensions must be > 0")
    }
    Ok(dims)
}

fn init_ort() -> Result<()> {
    if let Ok(path) = std::env::var("ORT_DYLIB_PATH") {
        let p = Path::new(&path);
        if p.is_file() {
            ort::init_from(p)
                .context("Failed to load ORT from ORT_DYLIB_PATH")?
                .commit();
            if let Ok(env) = ort::environment::Environment::current() {
                env.set_log_level(ort::logging::LogLevel::Warning);
            }
            return Ok(());
        }
    }

    for candidate in ["/usr/lib/libonnxruntime.so", "/usr/local/lib/libonnxruntime.so"] {
        let p = Path::new(candidate);
        if p.is_file() {
            ort::init_from(p)
                .with_context(|| format!("Failed to load ORT from {}", p.display()))?
                .commit();
            if let Ok(env) = ort::environment::Environment::current() {
                env.set_log_level(ort::logging::LogLevel::Warning);
            }
            return Ok(());
        }
    }

    ort::init().commit();
    if let Ok(env) = ort::environment::Environment::current() {
        env.set_log_level(ort::logging::LogLevel::Warning);
    }
    Ok(())
}

fn validate_ort(model_path: &str, shape: &[usize]) -> Result<()> {
    let t0 = Instant::now();
    let mut session = ort::session::Session::builder()
        .context("Failed to create ORT session builder")?
        .commit_from_file(model_path)
        .with_context(|| format!("Failed to load ONNX model: {model_path}"))?;
    println!("  Loaded in {:.2}s", t0.elapsed().as_secs_f64());

    let input = session
        .inputs()
        .first()
        .context("Model has no inputs")?;
    let input_name = input.name().to_string();
    println!("  Input: name='{}'", input_name);

    let dummy = make_deterministic_noise(shape, 0.01);
    let shape_i64: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
    let tensor = ort::value::Tensor::from_array((shape_i64, dummy.into_boxed_slice()))
        .context("Cannot build input tensor")?;

    let t1 = Instant::now();
    let outputs = session
        .run(ort::inputs![input_name => tensor])
        .context("Dummy inference failed")?;
    let infer_dt = t1.elapsed().as_secs_f64();

    if outputs.len() == 0 {
        bail!("Model has no outputs");
    }

    for i in 0..outputs.len() {
        let (out_shape, data) = outputs[i]
            .try_extract_tensor::<f32>()
            .with_context(|| format!("Output[{i}] is not extractable as f32 tensor"))?;

        if data.iter().any(|v| v.is_nan()) {
            bail!("Output[{i}] contains NaN values");
        }
        if data.iter().any(|v| v.is_infinite()) {
            bail!("Output[{i}] contains Inf values");
        }

        println!("  Output[{i}]: shape={out_shape:?}, len={}, dtype=float32", data.len());
    }

    let (out0_shape, out0_data) = outputs[0]
        .try_extract_tensor::<f32>()
        .context("Cannot inspect Output[0] as f32 tensor")?;
    println!(
        "  Inference: {:.3}s, output shape={out0_shape:?}, dtype=float32",
        infer_dt
    );
    println!("  Classes: {}", out0_data.len());

    Ok(())
}

fn validate_tract(model_path: &str) -> Result<()> {
    tract_onnx::onnx()
        .model_for_path(model_path)
        .with_context(|| format!("tract-onnx failed to load: {model_path}"))?
        .into_optimized()
        .context("tract-onnx optimization failed")?
        .into_runnable()
        .context("tract-onnx runnable conversion failed")?;

    println!("  tract-onnx compatibility: all checks passed ✓");
    Ok(())
}

fn make_deterministic_noise(shape: &[usize], scale: f32) -> Vec<f32> {
    let len = shape.iter().product();
    let mut x: u64 = 0x9E3779B97F4A7C15;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        let y = x.wrapping_mul(0x2545F4914F6CDD1D);
        let unit = ((y >> 40) as u32) as f32 / ((1u32 << 24) as f32); // [0,1)
        let centered = (unit * 2.0) - 1.0; // [-1,1)
        out.push(centered * scale);
    }
    out
}
