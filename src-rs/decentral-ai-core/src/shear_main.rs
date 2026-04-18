//! SHEAR Engine CLI — end-to-end inference demo
//! 
//! Usage: cargo run --bin shear -- [OPTIONS]
//!
//! Example: cargo run --bin shear -- --cells 3 --layers 4 --prompt "Hello world" --max-new 20
//!
//! This demo:
//!   1. Initializes N Cells with random weights
//!   2. Tokenizes input prompt
//!   3. Runs forward pass for each token through all Cells
//!   4. Aggregates Cell outputs (WeightedSum)
//!   5. Samples next token and generates text

use decentral_ai_core::{aggregator::{ShearEngine, Aggregator, CombineStrategy, AggregatorConfig}, cell::{Cell, CellConfig, CellState}};
use std::time::Instant;

// ===================== Simple tokenizer =====================

/// Minimal word-level tokenizer for demo purposes.
/// Maps each word to a unique token ID, with special tokens for BOS/EOS/PAD.
struct SimpleTokenizer {
    vocab: Vec<String>,
}

impl SimpleTokenizer {
    fn new() -> Self {
        // Special tokens first
        let special: Vec<String> = vec!["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
            .into_iter().map(|s| s.to_string()).collect();
        Self { vocab: special }
    }

    /// Build vocab from corpus
    fn fit(&mut self, corpus: &str) {
        for word in corpus.split_whitespace() {
            if !self.vocab.contains(&word.to_string()) {
                self.vocab.push(word.to_string());
            }
        }
    }

    /// Encode text → token IDs
    fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids = vec![1usize; 1]; // BOS
        for word in text.split_whitespace() {
            if let Some(idx) = self.vocab.iter().position(|w| w == word) {
                ids.push(idx);
            } else {
                ids.push(3); // UNK
            }
        }
        ids
    }

    /// Decode token IDs → text
    fn decode(&self, ids: &[usize]) -> String {
        let mut words = Vec::new();
        for &id in ids {
            if id < self.vocab.len() {
                let w = &self.vocab[id];
                if !w.starts_with('<') {
                    words.push(w.clone());
                }
            }
        }
        words.join(" ")
    }

    fn vocab_size(&self) -> usize { self.vocab.len() }
}

// ===================== Arguments =====================

#[derive(Debug)]
struct Args {
    n_cells: usize,
    vocab_size: usize,
    d_model: usize,
    d_ffn: usize,
    n_layers: usize,
    n_heads: usize,
    max_seq_len: usize,
    prompt: String,
    max_new: usize,
    temperature: f32,
    warmup: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            n_cells: 3,
            vocab_size: 1024,
            d_model: 128,
            d_ffn: 512,
            n_layers: 2,
            n_heads: 4,
            max_seq_len: 128,
            prompt: "hello world this is a test".to_string(),
            max_new: 20,
            temperature: 0.8,
            warmup: false,
        }
    }
}

impl Args {
    fn from_env() -> Self {
        let args: Vec<String> = std::env::args().collect();
        let mut a = Args::default();

        for i in 1..args.len() {
            match args[i].as_str() {
                "--cells"    => { if let Some(v) = args.get(i+1).and_then(|s| s.parse().ok()) { a.n_cells = v; } }
                "--vocab"    => { if let Some(v) = args.get(i+1).and_then(|s| s.parse().ok()) { a.vocab_size = v; } }
                "--dmodel"   => { if let Some(v) = args.get(i+1).and_then(|s| s.parse().ok()) { a.d_model = v; } }
                "--dffn"     => { if let Some(v) = args.get(i+1).and_then(|s| s.parse().ok()) { a.d_ffn = v; } }
                "--layers"   => { if let Some(v) = args.get(i+1).and_then(|s| s.parse().ok()) { a.n_layers = v; } }
                "--heads"    => { if let Some(v) = args.get(i+1).and_then(|s| s.parse().ok()) { a.n_heads = v; } }
                "--prompt"   => { if let Some(p) = args.get(i+1) { a.prompt = p.clone(); } }
                "--max-new"  => { if let Some(v) = args.get(i+1).and_then(|s| s.parse().ok()) { a.max_new = v; } }
                "--temp"     => { if let Some(v) = args.get(i+1).and_then(|s| s.parse().ok()) { a.temperature = v; } }
                "--warmup"   => { a.warmup = true; }
                "-h" | "--help" => {
                    println!("SHEAR Engine CLI — usage:");
                    println!("  cargo run --bin shear -- [OPTIONS]");
                    println!();
                    println!("Options:");
                    println!("  --cells N      Number of parallel Cells (default: 3)");
                    println!("  --vocab N      Vocabulary size (default: 1024)");
                    println!("  --dmodel N     Hidden dimension (default: 128)");
                    println!("  --dffn N       FFN intermediate size (default: 512)");
                    println!("  --layers N     Number of layers (default: 2)");
                    println!("  --heads N      Number of attention heads (default: 4)");
                    println!("  --prompt TXT   Input prompt (default: 'hello world...')");
                    println!("  --max-new N    Max new tokens to generate (default: 20)");
                    println!("  --temp T       Sampling temperature (default: 0.8)");
                    println!("  --warmup       Warmup run before timing");
                    println!("  -h, --help     Show this help");
                    std::process::exit(0);
                }
                _ => {}
            }
        }

        // Validate
        if a.d_model % a.n_heads != 0 {
            eprintln!("d_model {} must be divisible by n_heads {}", a.d_model, a.n_heads);
            std::process::exit(1);
        }
        if a.n_cells == 0 {
            eprintln!("n_cells must be > 0");
            std::process::exit(1);
        }

        a
    }
}

// ===================== Stats =====================

fn format_params(cfg: &CellConfig) -> String {
    let m = cfg.total_params();
    if m >= 1_000_000 {
        format!("{:.1}M", m as f64 / 1_000_000.0)
    } else {
        format!("{}", m)
    }
}

// ===================== Main =====================

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════╗");
    println!("║         SHEAR Engine — End-to-End Demo          ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();

    let args = Args::from_env();

    // 1. Build tokenizer
    let mut tokenizer = SimpleTokenizer::new();
    tokenizer.fit(&args.prompt);
    // Expand vocab to target size
    while tokenizer.vocab_size() < args.vocab_size {
        tokenizer.vocab.push(format!("<extra_{}>", tokenizer.vocab_size()));
    }
    let vocab_size = args.vocab_size;
    println!("[Tokenizer] word-level, vocab_size={}", vocab_size);

    // 2. Encode prompt
    let input_ids = tokenizer.encode(&args.prompt);
    println!("[Input]    \"{}\"", args.prompt);
    println!("[Tokens]   {:?}", input_ids);

    // 3. Cell config
    let cell_cfg = CellConfig {
        vocab_size,
        d_model: args.d_model,
        d_ffn: args.d_ffn,
        n_layers: args.n_layers,
        head_size: args.d_model / args.n_heads,
        n_heads: args.n_heads,
        max_seq_len: args.max_seq_len,
    };
    println!();
    println!("[Config]");
    println!("  N cells:        {}", args.n_cells);
    println!("  Vocab:          {}", vocab_size);
    println!("  d_model:        {}", cell_cfg.d_model);
    println!("  d_ffn:          {}", cell_cfg.d_ffn);
    println!("  Layers:         {}", cell_cfg.n_layers);
    println!("  Heads:          {}", cell_cfg.n_heads);
    println!("  Head size:      {}", cell_cfg.head_size);
    println!("  Params/cell:    {} params", format_params(&cell_cfg));
    println!("  Total params:   {} params", format_params(&cellConfig_total(&cell_cfg, args.n_cells)));
    println!();

    // 4. Build N Cells
    println!("[Init]     Initializing {} Cell(s)...", args.n_cells);
    let start_init = Instant::now();
    let mut cells = Vec::with_capacity(args.n_cells);
    for i in 0..args.n_cells {
        let init_start = Instant::now();
        let cell = Cell::random(cell_cfg.clone());
        println!("  Cell {}: {} params, init in {:?}",
                 i, format_params(&cell_cfg), init_start.elapsed());
        cells.push(cell);
    }
    println!("  Total init time: {:?}", start_init.elapsed());

    // 5. Build aggregator
    let _agg_cfg = AggregatorConfig {
        vocab_size,
        n_cells: args.n_cells,
        d_model: args.d_model,
        strategy: CombineStrategy::WeightedSum,
    };
    let aggregator = Aggregator::with_weights(
        vocab_size,
        args.n_cells,
        vec![1.0f32; args.n_cells], // equal weights
    ).with_tags((0..args.n_cells).map(|i| format!("cell-{}", i)).collect());

    let engine = ShearEngine::new(aggregator, (0..args.n_cells).map(|i| format!("cell-{}", i)).collect::<Vec<String>>());

    println!();
    println!("[Aggregator] strategy=WeightedSum, weights=[{}]",
             (0..args.n_cells).map(|_| "1/N".to_string()).collect::<Vec<_>>().join(", "));

    // 6. Warmup (optional)
    if args.warmup {
        println!();
        println!("[Warmup]   Running 3 warmup tokens...");
        let mut warmup_states: Vec<CellState> = cells.iter().map(|c| CellState::new(&c.config)).collect();
        let warmup_tokens = &[0usize, 1, 2];
        for &t in warmup_tokens {
            let logits: Vec<_> = cells.iter().enumerate()
                .map(|(i, c)| c.forward_token(t, &mut warmup_states[i]))
                .collect();
            let _ = engine.step(&logits, 0.8);
        }
        println!("[Warmup]   Done.");
    }

    // 7. Forward pass through prompt tokens
    println!();
    println!("[Prompt]   Processing {} prompt tokens...", input_ids.len());
    let mut cell_states: Vec<CellState> = cells.iter().map(|c| CellState::new(&c.config)).collect();
    let start_forward = Instant::now();

    for (pos, &token_id) in input_ids.iter().enumerate() {
        let t_start = Instant::now();
        let logits: Vec<_> = cells.iter().enumerate()
            .map(|(i, c)| c.forward_token(token_id, &mut cell_states[i]))
            .collect();

        // Show top token per cell
        let top_ids: Vec<usize> = logits.iter().map(|l| {
            l.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i).unwrap_or(0)
        }).collect();

        println!("  step {:3}: token={:5} → top per cell {:?} [{:?}]",
                 pos, token_id, top_ids, t_start.elapsed());
    }

    println!("  Prompt time: {:?}", start_forward.elapsed());

    // 8. Generate new tokens
    println!();
    println!("[Generate] max_new={}, temp={}", args.max_new, args.temperature);
    let mut all_tokens = input_ids.clone();
    let start_gen = Instant::now();

    for step in 0..args.max_new {
        let t_start = Instant::now();

        // Forward through all cells
        let logits: Vec<_> = cells.iter().enumerate()
            .map(|(i, c)| c.forward_token(*all_tokens.last().unwrap(), &mut cell_states[i]))
            .collect();

        // Aggregate
        let next_token = engine.step(&logits, args.temperature);
        all_tokens.push(next_token);

        let word = tokenizer.vocab.get(next_token).cloned().unwrap_or_else(|| "<UNK>".to_string());
        println!("  step {:3}: token={:5} → \"{}\" [{:?}]",
                 step, next_token, word, t_start.elapsed());

        // Stop on EOS
        if next_token == 0 || word.starts_with("<EOS>") {
            println!("  [EOS] Stopped.");
            break;
        }
    }

    let total_time = start_gen.elapsed();
    let total_new = all_tokens.len() - input_ids.len();
    let tokens_per_sec = if total_time.as_secs_f64() > 0.0 {
        total_new as f64 / total_time.as_secs_f64()
    } else { 0.0 };

    println!();
    println!("╔══════════════════════════════════════════════════╗");
    println!("║                    Results                        ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();
    println!("  Generated:   {} tokens in {:?}", total_new, total_time);
    println!("  Speed:      {:.1} tokens/sec", tokens_per_sec);
    println!();

    let output_text = tokenizer.decode(&all_tokens[input_ids.len()..]);
    println!("  Output:     \"{}\"", output_text);
    println!();
    println!("  Full text:  \"{}\"", tokenizer.decode(&all_tokens));
    println!();
    println!("[Done]     SHEAR Engine demo complete.");
}

fn cellConfig_total(cfg: &CellConfig, n_cells: usize) -> CellConfig {
    let _total = cfg.total_params() * n_cells;
    cfg.clone()
}
