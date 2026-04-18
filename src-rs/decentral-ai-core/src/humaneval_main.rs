use std::env;

use decentral_ai_core::rwkv_model::RwkvModel;
use decentral_ai_core::tokenizer::BpeTokenizer;

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut model_path = String::new();
    let mut data_path = String::from("data/HumanEval.jsonl");
    let mut output_path = String::from("humaneval_results.jsonl");
    let mut max_tokens: usize = 128;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { model_path = args.get(i+1).cloned().unwrap_or_default(); i += 2; }
            "--data" => { data_path = args.get(i+1).cloned().unwrap_or_default(); i += 2; }
            "--output" => { output_path = args.get(i+1).cloned().unwrap_or_default(); i += 2; }
            "--max-tokens" => { max_tokens = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(128); i += 2; }
            _ => { i += 1; }
        }
    }

    if model_path.is_empty() {
        eprintln!("Usage: humaneval --model <path> [--data path] [--output path] [--max-tokens N]");
        eprintln!("");
        eprintln!("Resume: automatically skips already-completed tasks in output file.");
        std::process::exit(1);
    }

    let data_dir = format!("{}/data", env!("CARGO_MANIFEST_DIR"));

    eprintln!("[Tokenizer] Loading from {}", data_dir);
    let tokenizer = BpeTokenizer::load(&data_dir).expect("Failed to load tokenizer");

    eprintln!("[Model] Loading {}", model_path);
    let model = RwkvModel::load_from_file(&model_path).expect("Failed to load model");
    eprintln!("  Params: {}", model.total_params());

    decentral_ai_core::humaneval::run_humaneval(
        &model, &tokenizer, &data_path, &output_path, max_tokens,
    ).expect("HumanEval run failed");
}
