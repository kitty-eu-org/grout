use anyhow::{Result, ensure};
use clap::Parser;
use grout::model::{GenerationOutput, Qwen3Engine};
use serde_json::json;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Qwen3-4B inference prototype on cuda-tile")]
struct Args {
    #[arg(long)]
    model: PathBuf,

    #[arg(long)]
    prompt: String,

    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,

    #[arg(long)]
    max_seq_len: Option<usize>,

    #[arg(long, default_value_t = false)]
    sample: bool,

    #[arg(long, default_value_t = false)]
    raw_prompt: bool,

    #[arg(long, default_value_t = false)]
    device_argmax: bool,

    #[arg(long, default_value_t = false)]
    profile: bool,

    #[arg(long, default_value_t = 1)]
    runs: usize,

    #[arg(long, default_value_t = 0)]
    warmup_runs: usize,

    /// Write benchmark metrics (same schema as bench/bench_transformers.py) to a JSON file.
    #[arg(long)]
    json_out: Option<PathBuf>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.runs > 0, "--runs must be greater than 0");

    let mut engine = Qwen3Engine::load(&args.model, args.max_seq_len).await?;
    engine.set_sampling_enabled(args.sample);
    engine.set_chat_template_enabled(!args.raw_prompt);
    engine.set_device_argmax_enabled(args.device_argmax);
    engine.set_profile_enabled(args.profile);

    println!("Loaded model from {}", engine.model_dir().display());
    println!("Prompt: {}", args.prompt);
    println!("Generating {} tokens...", args.max_new_tokens);

    if args.runs == 1 && args.warmup_runs == 0 && args.json_out.is_none() {
        let output = engine.generate(&args.prompt, args.max_new_tokens).await?;
        print_single_output(output);
        return Ok(());
    }

    println!(
        "Benchmark mode: warmup_runs={}, measured_runs={}",
        args.warmup_runs, args.runs
    );

    for run_idx in 0..args.warmup_runs {
        let output = engine.generate(&args.prompt, args.max_new_tokens).await?;
        println!(
            "warmup {}/{}: {:.2} prompt, {:.2} decode, {:.2} end-to-end (prompt_tokens={}, generated_tokens={}, prompt_s={:.3}, decode_s={:.3}, total_s={:.3})",
            run_idx + 1,
            args.warmup_runs,
            output.prompt_tps(),
            output.decode_tps(),
            output.total_tps(),
            output.prompt_tokens,
            output.generated_tokens,
            output.prompt_elapsed.as_secs_f64(),
            output.decode_elapsed.as_secs_f64(),
            output.total_elapsed.as_secs_f64(),
        );
    }

    let mut prompt_tps_sum = 0.0;
    let mut decode_tps_sum = 0.0;
    let mut total_tps_sum = 0.0;
    let mut prompt_s_sum = 0.0;
    let mut decode_s_sum = 0.0;
    let mut total_s_sum = 0.0;
    let mut prompt_tokens_sum = 0usize;
    let mut generated_tokens_sum = 0usize;

    // Per-run records for JSON serialization.
    let mut json_runs: Vec<serde_json::Value> = Vec::with_capacity(args.runs);
    // Last captured op-profile (parsed from profile_report), if profiling enabled.
    let mut last_op_profile: Vec<serde_json::Value> = Vec::new();

    for run_idx in 0..args.runs {
        let output = engine.generate(&args.prompt, args.max_new_tokens).await?;
        let prompt_s = output.prompt_elapsed.as_secs_f64();
        let decode_s = output.decode_elapsed.as_secs_f64();
        let total_s = output.total_elapsed.as_secs_f64();
        let prompt_tps = output.prompt_tps();
        let decode_tps = output.decode_tps();
        let total_tps = output.total_tps();

        println!(
            "run {}/{}: {:.2} prompt, {:.2} decode, {:.2} end-to-end (prompt_tokens={}, generated_tokens={}, prompt_s={:.3}, decode_s={:.3}, total_s={:.3})",
            run_idx + 1,
            args.runs,
            prompt_tps,
            decode_tps,
            total_tps,
            output.prompt_tokens,
            output.generated_tokens,
            prompt_s,
            decode_s,
            total_s,
        );

        if let Some(report) = output.profile_report.as_deref() {
            println!();
            println!("{report}");
            // Parse the op-profile table once for the JSON output.
            if last_op_profile.is_empty() {
                last_op_profile = parse_op_profile(report);
            }
        }

        json_runs.push(json!({
            "run": run_idx + 1,
            "prompt_tokens": output.prompt_tokens,
            "generated_tokens": output.generated_tokens,
            "prompt_s": prompt_s,
            "decode_s": decode_s,
            "total_s": total_s,
            "prompt_tps": prompt_tps,
            "decode_tps": decode_tps,
            "total_tps": total_tps,
        }));

        prompt_tps_sum += prompt_tps;
        decode_tps_sum += decode_tps;
        total_tps_sum += total_tps;
        prompt_s_sum += prompt_s;
        decode_s_sum += decode_s;
        total_s_sum += total_s;
        prompt_tokens_sum += output.prompt_tokens;
        generated_tokens_sum += output.generated_tokens;
    }

    let runs = args.runs as f64;
    println!();
    println!(
        "summary avg over {} runs: {:.2} prompt, {:.2} decode, {:.2} end-to-end (avg_prompt_tokens={:.1}, avg_generated_tokens={:.1}, avg_prompt_s={:.3}, avg_decode_s={:.3}, avg_total_s={:.3})",
        args.runs,
        prompt_tps_sum / runs,
        decode_tps_sum / runs,
        total_tps_sum / runs,
        prompt_tokens_sum as f64 / runs,
        generated_tokens_sum as f64 / runs,
        prompt_s_sum / runs,
        decode_s_sum / runs,
        total_s_sum / runs,
    );

    if let Some(path) = args.json_out.as_deref() {
        let cuda_graph_decode = env_bool_or("GROUT_CUDA_GRAPH_DECODE", false);
        let cublas_compute16 = env_bool_or("GROUT_CUBLAS_COMPUTE16", false);
        let profile_ops = env_bool_or("GROUT_PROFILE_OPS", false);
        let doc = json!({
            "engine": "grout",
            "model": engine.model_dir().display().to_string(),
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "dtype": "float16",
            "device": device_name(),
            "config": {
                "warmup_runs": args.warmup_runs,
                "runs": args.runs,
                "raw_prompt": args.raw_prompt,
                "sample": args.sample,
                "device_argmax": args.device_argmax,
                "profile": args.profile,
                "cuda_graph_decode": cuda_graph_decode,
                "cublas_compute16": cublas_compute16,
                "profile_ops": profile_ops,
            },
            "runs": json_runs,
            "summary": {
                "avg_prompt_tps": prompt_tps_sum / runs,
                "avg_decode_tps": decode_tps_sum / runs,
                "avg_total_tps": total_tps_sum / runs,
                "avg_prompt_s": prompt_s_sum / runs,
                "avg_decode_s": decode_s_sum / runs,
                "avg_total_s": total_s_sum / runs,
                "avg_prompt_tokens": prompt_tokens_sum as f64 / runs,
                "avg_generated_tokens": generated_tokens_sum as f64 / runs,
            },
            // Only present when --profile + GROUT_PROFILE_OPS=1.
            "op_profile": last_op_profile,
        });
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(path, serde_json::to_string_pretty(&doc)?)?;
        println!("\nWrote JSON metrics to {}", path.display());
    }

    Ok(())
}

fn print_single_output(output: GenerationOutput) {
    println!();
    println!("{}", output.text);
    println!(
        "t/s: {:.2} prompt, {:.2} decode, {:.2} end-to-end (prompt_tokens={}, generated_tokens={}, prompt_s={:.3}, decode_s={:.3}, total_s={:.3})",
        output.prompt_tps(),
        output.decode_tps(),
        output.total_tps(),
        output.prompt_tokens,
        output.generated_tokens,
        output.prompt_elapsed.as_secs_f64(),
        output.decode_elapsed.as_secs_f64(),
        output.total_elapsed.as_secs_f64(),
    );
    if let Some(report) = output.profile_report {
        println!();
        println!("{report}");
    }
}

fn env_bool_or(var: &str, default: bool) -> bool {
    std::env::var(var)
        .ok()
        .map(|v| v != "0")
        .unwrap_or(default)
}

/// Best-effort GPU name via nvidia-smi (returns "unknown" on failure).
fn device_name() -> String {
    std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name",
            "--format=csv,noheader",
        ])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Parse grout's `RunProfile::render()` op-profile lines into structured records.
///
/// The table looks like:
/// ```text
/// op profile (total_ms, avg_us, calls):
///   MatMul          total_ms=  12.345 avg_us= 1234.56 calls=10
/// ```
fn parse_op_profile(report: &str) -> Vec<serde_json::Value> {
    let mut out = Vec::new();
    let mut in_table = false;
    for line in report.lines() {
        if line.contains("op profile") {
            in_table = true;
            continue;
        }
        if !in_table || line.trim().is_empty() {
            continue;
        }
        if let Some(rec) = parse_op_line(line) {
            out.push(rec);
        }
    }
    out
}

fn parse_op_line(line: &str) -> Option<serde_json::Value> {
    // The profile line looks like:
    //   "  MatMul         total_ms=   2.740 avg_us=   10.87 calls=252"
    // Note the value may be separated from `key=` by whitespace (right-aligned),
    // so we scan the whole line rather than splitting on whitespace first.
    let trimmed = line.trim();
    let op = trimmed.split_whitespace().next()?;
    let total_ms = extract_field_f64(trimmed, "total_ms")?;
    let avg_us = extract_field_f64(trimmed, "avg_us");
    let calls = extract_field_u64(trimmed, "calls");
    Some(json!({
        "op": op,
        "total_ms": total_ms,
        "avg_us": avg_us.unwrap_or(0.0),
        "calls": calls.unwrap_or(0),
    }))
}

/// Extract the number following `name=` in `s`, tolerating whitespace after `=`.
fn extract_field_f64(s: &str, name: &str) -> Option<f64> {
    let idx = s.find(&format!("{name}="))?;
    let rest = &s[idx + name.len() + 1..];
    rest.split_whitespace().next()?.parse::<f64>().ok()
}

fn extract_field_u64(s: &str, name: &str) -> Option<u64> {
    let idx = s.find(&format!("{name}="))?;
    let rest = &s[idx + name.len() + 1..];
    rest.split_whitespace().next()?.parse::<u64>().ok()
}
