//! Debug test for RWKV-4-World-430M model.
use decentral_ai_core::rwkv_model::RwkvModel;
use decentral_ai_core::world_tokenizer::WorldTokenizer;

fn main() {
    let model_path = r"D:\IdeaProjects\SHEAR\network\src-rs\decentral-ai-core\data\rwkv4_430m.bin";
    let data_dir = r"D:\IdeaProjects\SHEAR\network\src-rs\decentral-ai-core\data";

    let tok = WorldTokenizer::load(data_dir).expect("tokenizer fail");
    let model = RwkvModel::load_from_file(model_path).expect("model fail");
    println!("Model: {} params, vocab={}\n", model.total_params(), model.vocab);

    let prompt = "def fibonacci";
    let ids = tok.encode(prompt);
    println!("Prompt: {:?}", prompt);
    println!("Token IDs: {:?}", ids);
    println!("Decoded: {:?}\n", tok.decode(&ids));

    let logits = model.debug_logits(&ids);
    let nan = logits.iter().filter(|v| v.is_nan()).count();
    let inf = logits.iter().filter(|v| v.is_infinite()).count();

    let min_logit = logits.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean_logit = logits.iter().sum::<f32>() / logits.len() as f32;
    println!("NaN: {}, Inf: {}", nan, inf);
    println!("Logit stats: min={:.3}, max={:.3}, mean={:.6}", min_logit, max_logit, mean_logit);

    // Top 10
    let mut with_idx: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    with_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\nTop 10 tokens:");
    for (i, &(id, logit)) in with_idx.iter().take(10).enumerate() {
        println!("  {}: id={} logit={:.3}  token={:?}", i+1, id, logit, tok.decode(&[id]));
    }

    println!("\nFirst 10 tokens by ID:");
    for i in 0..10 {
        println!("  {}: id={} logit={:.3}  token={:?}", i+1, i, logits[i], tok.decode(&[i]));
    }

    println!("\n=== Generation (temp=0, 10 tokens) ===");
    let (gen, _) = model.generate(&ids, 10, 0.0);
    let new_toks = &gen[ids.len()..];
    println!("New tokens: {:?}", new_toks);
    println!("Decoded: {:?}", tok.decode(new_toks));

    println!("\n=== Generation (temp=0.8, 20 tokens) ===");
    let (gen2, _) = model.generate(&ids, 20, 0.8);
    let new_toks2 = &gen2[ids.len()..];
    println!("New tokens: {:?}", new_toks2);
    println!("Decoded: {:?}", tok.decode(new_toks2));
}
