//! HumanEval benchmark with ensemble voting — compare baseline vs ensemble.

use std::env;

use decentral_ai_core::rwkv_model::RwkvModel;
use decentral_ai_core::tokenizer::BpeTokenizer;
use decentral_ai_core::speculative::{SpeculativeEngine, SpeculativeConfig, VoteStrategy, CellRole};

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut model_path = String::new();
    let data_dir = format!("{}/data", env!("CARGO_MANIFEST_DIR"));
    let mut data_path = format!("{}/HumanEval.jsonl", data_dir);
    let mut output_baseline = String::from("humaneval_baseline.jsonl");
    let mut output_ensemble = String::from("humaneval_ensemble.jsonl");
    let mut max_tokens: usize = 128;
    let mut n_cells: usize = 3;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { model_path = args.get(i+1).cloned().unwrap_or_default(); i += 2; }
            "--data" => { data_path = args.get(i+1).cloned().unwrap_or_default(); i += 2; }
            "--output-baseline" => { output_baseline = args.get(i+1).cloned().unwrap_or_default(); i += 2; }
            "--output-ensemble" => { output_ensemble = args.get(i+1).cloned().unwrap_or_default(); i += 2; }
            "--max-tokens" => { max_tokens = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(128); i += 2; }
            "--cells" => { n_cells = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(3); i += 2; }
            _ => { i += 1; }
        }
    }

    if model_path.is_empty() {
        eprintln!("Usage: humaneval_ensemble --model <path> [--data path] [--max-tokens N] [--cells N]");
        eprintln!("");
        eprintln!("Runs HumanEval with both baseline (single cell) and ensemble (N cells).");
        eprintln!("Outputs: humaneval_baseline.jsonl, humaneval_ensemble.jsonl");
        std::process::exit(1);
    }

    let data_dir = format!("{}/data", env!("CARGO_MANIFEST_DIR"));

    eprintln!("[Tokenizer] Loading from {}", data_dir);
    let tokenizer = BpeTokenizer::load(&data_dir).expect("Failed to load tokenizer");

    eprintln!("[Model] Loading {}", model_path);
    let model = RwkvModel::load_from_file(&model_path).expect("Failed to load model");
    eprintln!("  Params: {}", model.total_params());

    // Run baseline
    eprintln!("\n========================================");
    eprintln!("  Phase 1: Baseline (single cell)");
    eprintln!("========================================\n");
    decentral_ai_core::humaneval::run_humaneval(
        &model, &tokenizer, &data_path, &output_baseline, max_tokens,
    ).expect("Baseline HumanEval failed");

    // Run ensemble
    eprintln!("\n========================================");
    eprintln!("  Phase 2: Ensemble ({} cells)", n_cells);
    eprintln!("========================================\n");

    let config = SpeculativeConfig {
        n_cells,
        temperatures: (0..n_cells).map(|i| 0.5 + 0.3 * i as f32).collect(),
        cell_roles: (0..n_cells).map(|_| CellRole::general()).collect(),
        strategy: VoteStrategy::Majority,
        min_consensus: 0.0,
        draft_tokens: 0,
        top_p: 0.9,
    };

    // Note: We need to re-load the model because SpeculativeEngine takes ownership
    let model2 = RwkvModel::load_from_file(&model_path).expect("Failed to reload model");
    let mut engine = SpeculativeEngine::new(model2, config);

    run_humaneval_ensemble(
        &mut engine, &tokenizer, &data_path, &output_ensemble, max_tokens,
    ).expect("Ensemble HumanEval failed");
}

/// Run HumanEval with ensemble voting
fn run_humaneval_ensemble(
    engine: &mut SpeculativeEngine,
    tokenizer: &BpeTokenizer,
    data_path: &str,
    output_path: &str,
    max_tokens: usize,
) -> std::io::Result<()> {
    use std::fs::{File, OpenOptions};
    use std::io::{BufRead, BufReader, Write};

    let file = File::open(data_path)?;
    let reader = BufReader::new(file);

    // Resume support
    let completed = load_completed_task_ids(output_path);

    let out_file = OpenOptions::new()
        .write(true)
        .create(true)
        .append(true)
        .open(output_path)?;

    let mut writer = std::io::BufWriter::new(out_file);
    let mut total = 0u32;
    let mut skipped = 0u32;
    let mut total_tokens = 0usize;
    let mut total_consensus = 0.0f32;
    let start = std::time::Instant::now();

    eprintln!("  Data:    {}", data_path);
    eprintln!("  Output:  {}", output_path);
    eprintln!("  Max tok: {}", max_tokens);
    if !completed.is_empty() {
        eprintln!("  RESUME:  {} tasks already done", completed.len());
    }
    eprintln!();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() { continue; }

        let entry: serde_json::Value = serde_json::from_str(&line)?;
        let task_id = entry["task_id"].as_str().unwrap_or("unknown").to_string();
        let prompt = entry["prompt"].as_str().unwrap_or("");
        let entry_point = entry["entry_point"].as_str().unwrap_or("");
        let test_code = entry["test"].as_str().unwrap_or("");
        let canonical = entry["canonical_solution"].as_str().unwrap_or("");

        total += 1;

        if completed.contains(&task_id) {
            skipped += 1;
            continue;
        }

        eprintln!("[{:3}/164] {} ({}) ...", total, task_id, entry_point);

        // Encode prompt
        let token_ids = tokenizer.encode(prompt);
        let vocab_size = engine.vocab_size();
        let token_ids: Vec<usize> = token_ids.into_iter()
            .map(|id| if id >= vocab_size { 0 } else { id })
            .collect();

        // Generate with ensemble voting (Phase 1 backward-compat)
        let gen_start = std::time::Instant::now();
        let (generated_ids, stats) = engine.generate_phase1(&token_ids, max_tokens);
        let gen_time = gen_start.elapsed();

        total_tokens += generated_ids.len();
        total_consensus += stats.avg_consensus();

        let speed = generated_ids.len() as f64 / gen_time.as_secs_f64().max(0.001);

        eprintln!("         {} tokens in {:.1}s ({:.1} tok/s), consensus: {:.1}%",
            generated_ids.len(), gen_time.as_secs_f32(), speed, stats.avg_consensus() * 100.0);

        // Decode
        let completion = tokenizer.decode(&generated_ids);

        // Record
        let record = serde_json::json!({
            "task_id": task_id,
            "entry_point": entry_point,
            "prompt_tokens": token_ids.len(),
            "completion_tokens": generated_ids.len(),
            "completion_text": completion.clone(),
            "full_code": completion,
            "test_code": test_code,
            "canonical_solution": canonical,
            "speed_tok_per_s": speed,
            "avg_consensus": stats.avg_consensus(),
            "cell_hits": stats.cell_hits,
        });

        writeln!(writer, "{}", serde_json::to_string(&record)?)?;
        writer.flush()?;
    }

    let elapsed = start.elapsed();
    eprintln!("\n========================================");
    eprintln!("  Summary");
    eprintln!("========================================");
    eprintln!("  Total scanned:  {}", total);
    eprintln!("  Resumed (skip): {}", skipped);
    eprintln!("  Completed:      {}", total - skipped);
    eprintln!("  Total tokens:   {}", total_tokens);
    eprintln!("  Avg consensus:  {:.1}%", total_consensus / (total - skipped) as f32 * 100.0);
    eprintln!("  Total time:     {:.1}s", elapsed.as_secs_f32());
    eprintln!();

    Ok(())
}

fn load_completed_task_ids(output_path: &str) -> std::collections::HashSet<String> {
    use std::collections::HashSet;
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let mut set = HashSet::new();
    if let Ok(file) = File::open(output_path) {
        let reader = BufReader::new(file);
        for line in reader.lines() {
            if let Ok(l) = line {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&l) {
                    if let Some(id) = v["task_id"].as_str() {
                        set.insert(id.to_string());
                    }
                }
            }
        }
    }
    set
}
