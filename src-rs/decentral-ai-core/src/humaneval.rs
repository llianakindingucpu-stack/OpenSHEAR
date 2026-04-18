//! HumanEval benchmark runner — with crash recovery and per-problem error isolation.
//!
//! Loads RWKV model once, generates completions for all 164 problems.
//! Skips already-completed task_ids (resume from crash).
//! Each problem is isolated — one crash doesn't kill the rest.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};

use crate::rwkv_model::RwkvModel;
use crate::tokenizer::BpeTokenizer;

fn load_completed_task_ids(output_path: &str) -> HashSet<String> {
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

pub fn run_humaneval(
    model: &RwkvModel,
    tokenizer: &BpeTokenizer,
    data_path: &str,
    output_path: &str,
    max_tokens: usize,
) -> std::io::Result<()> {
    let file = File::open(data_path)?;
    let reader = BufReader::new(file);

    // Resume: check already-completed tasks
    let completed = load_completed_task_ids(output_path);
    let is_resume = !completed.is_empty();

    let out_file = OpenOptions::new()
        .write(true)
        .create(true)
        .append(true)  // <-- append mode for resume
        .open(output_path)?;

    let mut writer = std::io::BufWriter::new(out_file);
    let mut total = 0u32;
    let mut skipped = 0u32;
    let mut errors = 0u32;
    let mut total_tokens = 0usize;
    let start = std::time::Instant::now();

    eprintln!("\n========================================");
    eprintln!("  HumanEval Benchmark (RWKV-4-169M)");
    eprintln!("========================================\n");
    eprintln!("  Data:    {}", data_path);
    eprintln!("  Output:  {}", output_path);
    eprintln!("  Max tok: {}", max_tokens);
    if is_resume {
        eprintln!("  RESUME:  {} tasks already done", completed.len());
    }
    eprintln!();

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("  [WARN] Failed to read line: {}", e);
                continue;
            }
        };
        if line.trim().is_empty() { continue; }

        let entry: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("  [WARN] Failed to parse JSON: {}", e);
                continue;
            }
        };

        let task_id = entry["task_id"].as_str().unwrap_or("unknown").to_string();
        let prompt = entry["prompt"].as_str().unwrap_or("");
        let entry_point = entry["entry_point"].as_str().unwrap_or("");
        let test_code = entry["test"].as_str().unwrap_or("");
        let canonical = entry["canonical_solution"].as_str().unwrap_or("");

        total += 1;

        // Skip already completed
        if completed.contains(&task_id) {
            skipped += 1;
            continue;
        }

        eprintln!("[{:3}/164] {} ({}) ...", total, task_id, entry_point);

        // Encode prompt
        let token_ids = tokenizer.encode(prompt);

        // Clamp token IDs to valid range
        let vocab_size = model.emb.nrows();
        let token_ids: Vec<usize> = token_ids.into_iter()
            .map(|id| if id >= vocab_size { 0 } else { id })
            .collect();

        // Generate completion — wrapped in per-problem error isolation
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let gen_start = std::time::Instant::now();
            let (generated_ids, _gen_duration) = model.generate(&token_ids, max_tokens, 0.8);
            let gen_time = gen_start.elapsed();
            let speed = if gen_time.as_secs_f64() > 0.0 {
                generated_ids.len() as f64 / gen_time.as_secs_f64()
            } else {
                0.0
            };
            (generated_ids, speed, gen_time)
        }));

        match result {
            Ok((generated_ids, speed, gen_time)) => {
                total_tokens += generated_ids.len();

                // Decode completion
                let completion = tokenizer.decode(&generated_ids);

                eprintln!("         {} tokens in {:.1}s ({:.1} tok/s)",
                    generated_ids.len(), gen_time.as_secs_f32(), speed);

                // Build output record
                let full_code = completion.clone();
                let completion_text = completion;
                let record = serde_json::json!({
                    "task_id": task_id,
                    "entry_point": entry_point,
                    "prompt_tokens": token_ids.len(),
                    "completion_tokens": generated_ids.len(),
                    "completion_text": completion_text,
                    "full_code": full_code,
                    "test_code": test_code,
                    "canonical_solution": canonical,
                    "speed_tok_per_s": speed,
                });

                if let Err(e) = writeln!(writer, "{}", serde_json::to_string(&record).unwrap()) {
                    eprintln!("  [ERROR] Failed to write result: {}", e);
                    errors += 1;
                }
            }
            Err(_) => {
                eprintln!("         *** PANIC on this problem, skipping ***");
                errors += 1;

                // Write error record so we don't retry infinitely
                let record = serde_json::json!({
                    "task_id": task_id,
                    "entry_point": entry_point,
                    "error": "panic",
                    "prompt_tokens": token_ids.len(),
                    "completion_tokens": 0,
                    "completion_text": "",
                    "full_code": prompt.to_string(),
                    "test_code": test_code,
                    "canonical_solution": canonical,
                    "speed_tok_per_s": 0.0,
                });

                let _ = writeln!(writer, "{}", serde_json::to_string(&record).unwrap());
            }
        }

        // Flush after each problem to survive crashes
        let _ = writer.flush();
    }

    let elapsed = start.elapsed();
    let completed_count = completed.len() as u32 + (total - skipped - errors);
    let avg_speed = if elapsed.as_secs_f64() > 0.0 {
        total_tokens as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    eprintln!("\n========================================");
    eprintln!("  Summary");
    eprintln!("========================================");
    eprintln!("  Total scanned:  {}", total);
    eprintln!("  Resumed (skip): {}", skipped);
    eprintln!("  Completed:      {}", completed_count);
    eprintln!("  Errors/Panics:  {}", errors);
    eprintln!("  Total tokens:   {}", total_tokens);
    eprintln!("  Total time:     {:.1}s", elapsed.as_secs_f32());
    eprintln!("  Avg speed:      {:.1} tok/s", avg_speed);
    eprintln!();

    Ok(())
}
