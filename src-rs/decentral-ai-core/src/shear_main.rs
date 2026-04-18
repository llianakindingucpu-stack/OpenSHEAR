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

use decentral_ai_core::rwkv_model::{self, RwkvModel, RwkvModelState};
use decentral_ai_core::tokenizer::BpeTokenizer;
use std::time::Instant;

// ===================== Args =====================

struct Args {
    mode: String,
    model_path: String,
    data_dir: String,
    prompt: String,
    max_new: usize,
    temperature: f32,
    // shear-only
    n_cells: usize,
    d_model: usize,
    n_layers: usize,
    warmup: bool,
}

impl Args {
    fn parse() -> Self {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let mut mode = "shear".to_string();
        let mut model_path = "D:/IdeaProjects/decentral-ai/data/rwkv4_169m.bin".to_string();
        let mut data_dir = format!("{}/data", env!("CARGO_MANIFEST_DIR"));
        let mut prompt = "Hello, how are you?".to_string();
        let mut max_new = 30;
        let mut temperature = 0.8;
        let mut n_cells = 3;
        let mut d_model = 128;
        let mut n_layers = 2;
        let mut warmup = false;

        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--model" => { mode = "model".to_string(); model_path = args.get(i+1).cloned().unwrap_or_default(); i += 1; }
                "--data-dir" => { data_dir = args.get(i+1).cloned().unwrap_or_default(); i += 1; }
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
                    println!("  --shear         Random-weight SHEAR demo (default)");
                    println!("  --model <path>  Load RWKV-4-169M binary, run inference");
                    println!("  --data-dir <d>  Tokenizer data dir (default: data/)");
                    println!("  --prompt TXT    Input prompt (default: 'Hello, how are you?')");
                    println!("  --max-new N     Max new tokens (default: 30)");
                    println!("  --temp T        Temperature (default: 0.8)");
                    println!("  --warmup        Warmup before timing");
                    println!("  --cells N       [shear] Number of cells (default: 3)");
                    println!("  --dmodel N     [shear] Hidden dim (default: 128)");
                    println!("  --layers N     [shear] Layers per cell (default: 2)");
                    std::process::exit(0);
                }
                _ => {}
            }
            i += 1;
        }
        Args { mode, model_path, data_dir, prompt, max_new, temperature, n_cells, d_model, n_layers, warmup }
    }
}

// ===================== SHEAR Demo (random weights) =====================

#[allow(dead_code)]
fn run_shear(args: &Args) {
    use decentral_ai_core::aggregator::{ShearEngine, Aggregator};
    use decentral_ai_core::cell::{Cell, CellConfig, CellState};

    println!();
    println!("╔══════════════════════════════════════════════════╗");
    println!("║         SHEAR Engine — Random Cell Demo        ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();

    let vocab_size = 1024;
    let cell_cfg = CellConfig {
        vocab_size,
        d_model: args.d_model,
        d_ffn: args.d_model * 4,
        n_layers: args.n_layers,
        head_size: args.d_model / 4,
        n_heads: 4,
        max_seq_len: 128,
    };

    // Simple word-level tokenizer (random vocab)
    #[derive(Debug)]
    struct SimpleTok {
        word_to_id: std::collections::HashMap<String, usize>,
        id_to_word: Vec<String>,
        next_id: usize,
    }
    impl SimpleTok {
        fn new() -> Self {
            let mut t = SimpleTok { word_to_id: Default::default(), id_to_word: vec!["<PAD>".into(), "<BOS>".into(), "<EOS>".into(), "<UNK>".into()], next_id: 4 };
            t
        }
        fn fit(&mut self, text: &str) {
            for word in text.split_whitespace() {
                if !self.word_to_id.contains_key(word) {
                    let id = self.next_id;
                    self.next_id += 1;
                    self.word_to_id.insert(word.to_string(), id);
                    if id >= self.id_to_word.len() { self.id_to_word.resize(id+1, String::new()); }
                    self.id_to_word[id] = word.to_string();
                }
            }
            while self.id_to_word.len() < 1024 {
                let id = self.id_to_word.len();
                self.id_to_word.push(format!("<{}>", id));
            }
        }
        fn encode(&self, text: &str) -> Vec<usize> {
            let mut ids = vec![1usize; 1];
            for word in text.split_whitespace() {
                ids.push(*self.word_to_id.get(word).unwrap_or(&3));
            }
            ids
        }
        fn decode(&self, ids: &[usize]) -> String {
            ids.iter().filter_map(|&i| self.id_to_word.get(i).cloned()).filter(|w| !w.starts_with('<')).collect::<Vec<_>>().join(" ")
        }
    }

    let mut tokenizer = SimpleTok::new();
    tokenizer.fit(&args.prompt);
    tokenizer.fit("the quick brown fox jumps over the lazy dog");

    let input_ids = tokenizer.encode(&args.prompt);
    println!("[Tokens] {:?}", input_ids);
    println!("[Prompt] \"{}\"", args.prompt);
    println!();

    println!("[Init]   {} Cell(s), {} params each", args.n_cells, cell_cfg.total_params());
    let cells: Vec<_> = (0..args.n_cells).map(|_| Cell::random(cell_cfg.clone())).collect();
    let mut states: Vec<_> = cells.iter().map(|c| CellState::new(&c.config)).collect();
    let aggregator = Aggregator::with_weights(
        vocab_size, args.n_cells, vec![1.0f32; args.n_cells]
    ).with_tags((0..args.n_cells).map(|i| format!("cell-{}", i)).collect::<Vec<_>>());
    let engine = ShearEngine::new(aggregator, (0..args.n_cells).map(|i| format!("cell-{}", i)).collect::<Vec<_>>());

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
        let word = tokenizer.id_to_word.get(next).cloned().unwrap_or_default();
        println!("  step {:3}: → \"{}\"", step, word);
    }
    let elapsed = t1.elapsed();
    println!();
    println!("[Result] {} tokens in {:?} = {:.1} tok/s", all.len() - input_ids.len(), elapsed, (all.len()-input_ids.len()) as f64 / elapsed.as_secs_f64());
    println!("  Output: \"{}\"", tokenizer.decode(&all[input_ids.len()..]));
}

// ===================== RWKV Real Inference =====================

fn run_rwkv(args: &Args) {
    println!();
    println!("╔══════════════════════════════════════════════════╗");
    println!("║       RWKV-4-169M Real Inference Mode         ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();

    // 1. Load tokenizer
    println!("[Tokenizer] Loading BPE tokenizer from {}", args.data_dir);
    let tokenizer = match BpeTokenizer::load(&args.data_dir) {
        Ok(t) => t,
        Err(e) => { eprintln!("[ERROR] Failed to load tokenizer: {}", e); std::process::exit(1); }
    };
    println!("  Vocab size: {}", tokenizer.vocab_size());

    // 2. Load model
    println!("[Model]  Loading {}", args.model_path);
    let load_start = Instant::now();
    let model = match RwkvModel::load_from_file(&args.model_path) {
        Ok(m) => m,
        Err(e) => { eprintln!("[ERROR] Failed to load model: {}", e); std::process::exit(1); }
    };
    println!("  Loaded in {:?}", load_start.elapsed());
    println!("  Params: {} ({:.1}M)", model.total_params(), model.total_params() as f64 / 1_000_000.0);
    println!();

    // 3. Encode prompt
    let input_ids = tokenizer.encode_with_bos(&args.prompt);
    println!("[Prompt]  \"{}\"", args.prompt);
    println!("  Tokens: {:?}", input_ids);
    for (i, &id) in input_ids.iter().enumerate().take(10) {
        println!("    {:3}: id={:5} → {:?}", i, id, tokenizer.id_to_token(id));
    }
    if input_ids.len() > 10 { println!("    ... ({} more)", input_ids.len() - 10); }
    println!();

    // 4. Warmup
    if args.warmup {
        println!("[Warmup] Running 3 warmup tokens...");
        let mut state = RwkvModelState::new();
        for &id in &[0usize, 1, 2] {
            let _ = model.forward_token(id, &mut state);
        }
        println!("  Done.\n");
    }

    // 5. Generate
    println!("[Generate] max_new={}, temp={}", args.max_new, args.temperature);
    let (all_ids, gen_time) = model.generate(&input_ids, args.max_new, args.temperature);

    let new_ids = &all_ids[input_ids.len()..];

    println!();
    println!("╔══════════════════════════════════════════════════╗");
    println!("║                   Results                      ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();
    println!("  Prompt tokens: {}", input_ids.len());
    println!("  Generated:     {} tokens", new_ids.len());
    println!("  Time:          {:?}", gen_time);
    if gen_time.as_secs_f64() > 0.0 {
        println!("  Speed:        {:.1} tok/s", new_ids.len() as f64 / gen_time.as_secs_f64());
    }
    println!();

    // 6. Decode output
    let decoded = tokenizer.decode(new_ids);
    println!("[Output Text]");
    println!("{}", decoded);
    println!();

    // 7. Show tokens
    println!("[Output Tokens]");
    for (i, &id) in new_ids.iter().enumerate().take(20) {
        println!("  {:3}: id={:5} → {:?}", i, id, tokenizer.id_to_token(id));
    }
    if new_ids.len() > 20 { println!("  ... ({} more)", new_ids.len() - 20); }
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
