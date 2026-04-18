//! SHEAR Engine + RWKV Inference CLI
//!
//! Mode 1 — Synthetic demo (default):
//!   cargo run --bin shear -- --shear
//!
//! Mode 2 — Real RWKV-4-169M inference:
//!   cargo run --bin shear -- --model ../data/rwkv4_169m.bin --prompt "hello world"
//!
//! Mode 3 — List options:
//!   cargo run --bin shear -- --help

use decentral_ai_core::rwkv_model::{self, RwkvModel, RwkvModelState, VOCAB};
use rand::Rng;
use std::time::Instant;

// ===================== Simple Tokenizer =====================

struct Tokenizer {
    word_to_id: std::collections::HashMap<String, usize>,
    id_to_word: std::collections::HashMap<usize, String>,
    next_id: usize,
}

impl Tokenizer {
    fn new() -> Self {
        let mut t = Tokenizer {
            word_to_id: std::collections::HashMap::new(),
            id_to_word: std::collections::HashMap::new(),
            next_id: 4,  // 0=PAD, 1=BOS, 2=EOS, 3=UNK
        };
        // Special tokens
        for (id, name) in [(0, "<PAD>"), (1, "<BOS>"), (2, "<EOS>"), (3, "<UNK>")] {
            t.word_to_id.insert(name.to_string(), id);
            t.id_to_word.insert(id, name.to_string());
        }
        t
    }

    fn fit(&mut self, text: &str) {
        for word in text.split_whitespace() {
            if !self.word_to_id.contains_key(word) {
                let id = self.next_id;
                self.next_id += 1;
                self.word_to_id.insert(word.to_string(), id);
                self.id_to_word.insert(id, word.to_string());
            }
        }
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids = vec![1usize; 1]; // BOS
        for word in text.split_whitespace() {
            ids.push(*self.word_to_id.get(word).unwrap_or(&3)); // UNK
        }
        ids
    }

    fn decode(&self, ids: &[usize]) -> String {
        let mut words = Vec::new();
        for &id in ids {
            if let Some(w) = self.id_to_word.get(&id) {
                if !w.starts_with('<') {
                    words.push(w.clone());
                }
            }
        }
        words.join(" ")
    }
}

// ===================== Args =====================

struct Args {
    mode: String,          // "shear" or "model"
    model_path: String,
    prompt: String,
    max_new: usize,
    temperature: f32,
    n_cells: usize,
    // shear-only params
    d_model: usize,
    n_layers: usize,
    warmup: bool,
}

impl Args {
    fn parse() -> Self {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let mut mode = "shear".to_string();
        let mut model_path = String::new();
        let mut prompt = "hello world this is a test".to_string();
        let mut max_new = 20;
        let mut temperature = 0.8;
        let mut n_cells = 3;
        let mut d_model = 128;
        let mut n_layers = 2;
        let mut warmup = false;

        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--model" => { mode = "model".to_string(); model_path = args.get(i+1).cloned().unwrap_or_default(); i += 1; }
                "--shear" => { mode = "shear".to_string(); }
                "--prompt" => { if let Some(p) = args.get(i+1) { prompt = p.clone(); } i += 1; }
                "--max-new" => { if let Some(v) = args.get(i+1).and_then(|s| s.parse().ok()) { max_new = v; } i += 1; }
                "--temp" => { if let Some(v) = args.get(i+1).and_then(|s| s.parse().ok()) { temperature = v; } i += 1; }
                "--cells" => { if let Some(v) = args.get(i+1).and_then(|s| s.parse().ok()) { n_cells = v; } i += 1; }
                "--dmodel" => { if let Some(v) = args.get(i+1).and_then(|s| s.parse().ok()) { d_model = v; } i += 1; }
                "--layers" => { if let Some(v) = args.get(i+1).and_then(|s| s.parse().ok()) { n_layers = v; } i += 1; }
                "--warmup" => { warmup = true; }
                "-h" | "--help" => {
                    println!("SHEAR + RWKV CLI");
                    println!("  --shear         Use random-weight SHEAR demo (default)");
                    println!("  --model <path>  Load RWKV-4-169M binary, run inference");
                    println!("  --prompt TXT    Input prompt (default: 'hello world...')");
                    println!("  --max-new N     Max new tokens (default: 20)");
                    println!("  --temp T        Temperature (default: 0.8)");
                    println!("  --warmup        Warmup before timing");
                    println!();
                    println!("SHEAR mode extra:");
                    println!("  --cells N       Number of parallel Cells (default: 3)");
                    println!("  --dmodel N     Hidden dim (default: 128)");
                    println!("  --layers N     Layers per cell (default: 2)");
                    std::process::exit(0);
                }
                _ => {}
            }
            i += 1;
        }

        // Default RWKV model path
        if model_path.is_empty() {
            model_path = "D:/IdeaProjects/decentral-ai/data/rwkv4_169m.bin".to_string();
        }

        Args { mode, model_path, prompt, max_new, temperature, n_cells, d_model, n_layers, warmup }
    }
}

// ===================== SHEAR Demo (random weights) =====================

#[allow(dead_code)]
fn run_shear(args: &Args) {
    use decentral_ai_core::aggregator::{ShearEngine, Aggregator, AggregatorConfig};
    use decentral_ai_core::cell::{Cell, CellConfig, CellState};

    println!();
    println!("╔══════════════════════════════════════════════════╗");
    println!("║         SHEAR Engine — Random Cell Demo        ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();

    let vocab_size = 1024;
    let mut tokenizer = Tokenizer::new();
    tokenizer.fit(&args.prompt);
    while tokenizer.next_id < vocab_size {
        let id = tokenizer.next_id;
        tokenizer.next_id += 1;
        tokenizer.id_to_word.insert(id, format!("<extra_{}>", id));
    }

    let input_ids = tokenizer.encode(&args.prompt);
    println!("[Input]  \"{}\"", args.prompt);
    println!("[Tokens] {:?}", input_ids);
    println!();

    let cell_cfg = CellConfig {
        vocab_size,
        d_model: args.d_model,
        d_ffn: args.d_model * 4,
        n_layers: args.n_layers,
        head_size: args.d_model / 4,
        n_heads: 4,
        max_seq_len: 128,
    };

    println!("[Init]   {} Cell(s), {} params each", args.n_cells, cell_cfg.total_params());
    let cells: Vec<_> = (0..args.n_cells)
        .map(|_| Cell::random(cell_cfg.clone()))
        .collect();

    let mut states: Vec<_> = cells.iter().map(|c| CellState::new(&c.config)).collect();
    let aggregator = Aggregator::with_weights(
        vocab_size, args.n_cells, vec![1.0f32; args.n_cells]
    ).with_tags((0..args.n_cells).map(|i| format!("cell-{}", i)).collect::<Vec<_>>());
    let engine = ShearEngine::new(aggregator, (0..args.n_cells).map(|i| format!("cell-{}", i)).collect::<Vec<_>>());

    println!();
    println!("[Prompt] Processing {} tokens...", input_ids.len());
    let t0 = Instant::now();
    for &id in &input_ids {
        let logits: Vec<_> = cells.iter().enumerate()
            .map(|(i, c)| c.forward_token(id, &mut states[i]))
            .collect();
        let _ = engine.step(&logits, args.temperature);
    }
    println!("  Time: {:?}", t0.elapsed());

    println!();
    println!("[Generate] {} tokens, temp={}", args.max_new, args.temperature);
    let mut all = input_ids.clone();
    let t1 = Instant::now();
    for step in 0..args.max_new {
        let logits: Vec<_> = cells.iter().enumerate()
            .map(|(i, c)| c.forward_token(*all.last().unwrap(), &mut states[i]))
            .collect();
        let next = engine.step(&logits, args.temperature);
        all.push(next);
        let word = tokenizer.id_to_word.get(&next).cloned().unwrap_or_default();
        println!("  step {:3}: → \"{}\"", step, word);
    }

    let elapsed = t1.elapsed();
    let new_count = all.len() - input_ids.len();
    println!();
    println!("[Result] {} tokens in {:?} = {:.1} tok/s", new_count, elapsed, new_count as f64 / elapsed.as_secs_f64());
    println!("  Output: \"{}\"", tokenizer.decode(&all[input_ids.len()..]));
}

// ===================== RWKV Real Inference =====================

fn run_rwkv(args: &Args) {
    println!();
    println!("╔══════════════════════════════════════════════════╗");
    println!("║       RWKV-4-169M Real Inference Mode         ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();

    // 1. Load model
    println!("[Model]  Loading {}", args.model_path);
    let load_start = Instant::now();
    let model = match RwkvModel::load_from_file(&args.model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[ERROR] Failed to load model: {}", e);
            std::process::exit(1);
        }
    };
    println!("  Loaded in {:?}", load_start.elapsed());
    println!("  Params: {} ({:.1}M)",
             model.total_params(), model.total_params() as f64 / 1_000_000.0);
    println!();

    // 2. Simple tokenizer (word-level, vocabulary built from prompt)
    let mut tokenizer = Tokenizer::new();
    tokenizer.fit(&args.prompt);
    // Extend to full vocab (for demo: only map words that appear)
    // Note: real tokenizer would use the model's built-in tokenizer
    // For this demo, we'll use a simple approach: map common subwords to IDs
    // The actual RWKV model needs token IDs from its vocabulary

    // 3. Encode prompt — for demo, use first-token BOS + hash of each word
    let input_ids: Vec<usize> = {
        let mut ids = vec![1usize]; // BOS
        for word in args.prompt.split_whitespace() {
            // Map word to a pseudo-token (for demo; real use needs proper tokenizer)
            let hash = word.bytes().fold(0usize, |acc, b| acc.wrapping_mul(31).wrapping_add(b as usize));
            ids.push((hash % (VOCAB - 10)) + 4); // Avoid special tokens
        }
        ids
    };

    println!("[Prompt]  \"{}\"", args.prompt);
    println!("  Tokens: {:?}", input_ids);
    println!();

    // 4. Warmup
    if args.warmup {
        println!("[Warmup] Running 3 warmup tokens...");
        let mut state = RwkvModelState::new();
        for &id in &[0usize, 1, 2] {
            let _ = model.forward_token(id, &mut state);
        }
        println!("  Done.");
        println!();
    }

    // 5. Generate
    println!("[Generate] max_new={}, temp={}", args.max_new, args.temperature);
    let (all_ids, gen_time) = model.generate(&input_ids, args.max_new, args.temperature);

    let new_ids = &all_ids[input_ids.len()..];
    let total_new = new_ids.len();

    println!();
    println!("╔══════════════════════════════════════════════════╗");
    println!("║                   Results                      ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();
    println!("  Prompt tokens: {}", input_ids.len());
    println!("  Generated:     {} tokens", total_new);
    println!("  Time:          {:?}", gen_time);
    if gen_time.as_secs_f64() > 0.0 {
        println!("  Speed:        {:.1} tok/s", total_new as f64 / gen_time.as_secs_f64());
    }
    println!();

    // 6. Show generated token IDs (for debugging)
    println!("  New token IDs: {:?}", new_ids);
    println!();
    println!("[Done]");
}

// ===================== Main =====================

fn main() {
    let args = Args::parse();

    match args.mode.as_str() {
        "model" => run_rwkv(&args),
        _ => run_shear(&args),
    }
}
