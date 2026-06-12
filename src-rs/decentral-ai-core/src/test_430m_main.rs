//! Full test for RWKV-4-World-430M model with longer generation.
use decentral_ai_core::rwkv_model::RwkvModel;
use decentral_ai_core::world_tokenizer::WorldTokenizer;

fn main() {
    let model_path = r"D:\IdeaProjects\SHEAR\network\src-rs\decentral-ai-core\data\rwkv4_430m.bin";
    let data_dir = r"D:\IdeaProjects\SHEAR\network\src-rs\decentral-ai-core\data";

    let tok = WorldTokenizer::load(data_dir).expect("tokenizer fail");
    let model = RwkvModel::load_from_file(model_path).expect("model fail");
    println!("Model: {} params, vocab={}\n", model.total_params(), model.vocab);

    let prompts = [
        "def fibonacci",
        "The meaning of life is",
        "In the year 2026,",
        "Hello, my name is",
    ];

    for prompt in &prompts {
        let ids = tok.encode(prompt);
        let (gen, dur) = model.generate(&ids, 50, 0.8);
        let new_toks = &gen[ids.len()..];
        let decoded = tok.decode(new_toks);
        let speed = new_toks.len() as f32 / dur.as_secs_f32().max(0.001);
        println!("Prompt: {:?}", prompt);
        println!("Output: {}{}", prompt, decoded);
        println!("Speed: {:.1} tok/s\n", speed);
    }

    // Longer generation
    println!("=== Long generation (100 tokens) ===");
    let prompt = "Once upon a time";
    let ids = tok.encode(prompt);
    let (gen, dur) = model.generate(&ids, 100, 0.7);
    let new_toks = &gen[ids.len()..];
    let decoded = tok.decode(new_toks);
    let speed = new_toks.len() as f32 / dur.as_secs_f32().max(0.001);
    println!("{}{}", prompt, decoded);
    println!("\nSpeed: {:.1} tok/s ({} tokens in {:.1}s)", speed, new_toks.len(), dur.as_secs_f32());
}
