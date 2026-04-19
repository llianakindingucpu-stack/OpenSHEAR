//! Final test: verify RWKV generates actual NEW content after NaN fix.
use decentral_ai_core::rwkv_model::RwkvModel;
use decentral_ai_core::tokenizer::BpeTokenizer;

fn main() {
    let model_path = r"D:\IdeaProjects\decentral-ai\data\rwkv4_169m.bin";
    let data_dir = format!("{}/data", env!("CARGO_MANIFEST_DIR"));

    let tok = BpeTokenizer::load(&data_dir).expect("tok fail");
    let model = RwkvModel::load_from_file(model_path).expect("model fail");
    println!("Model: {} params\n", model.total_params());

    // Test 1: Short prompt
    let prompt1 = "def add(a, b):\n    \"\"\"";
    let ids1 = tok.encode(prompt1);
    let (gen1, dur1) = model.generate(&ids1, 30, 0.8);
    let new1 = &gen1[ids1.len()..];
    let dec1 = tok.decode(new1);
    println!("Test 1: Short prompt");
    println!("  Prompt: {:?}", prompt1);
    println!("  Generated {} new tokens, decoded:", new1.len());
    println!("  {:?}", &dec1[..dec1.len().min(100)]);
    println!("  Speed: {:.1} tok/s\n", new1.len() as f32 / dur1.as_secs_f32().max(0.001));

    // Test 2: HumanEval prompt
    let prompt2 = "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\"";
    let ids2 = tok.encode(prompt2);
    let (gen2, dur2) = model.generate(&ids2, 50, 0.8);
    let new2 = &gen2[ids2.len()..];
    let dec2 = tok.decode(new2);
    let _nan2 = false; // already confirmed 0 above
    println!("Test 2: HumanEval prompt");
    println!("  Generated {} new tokens", new2.len());
    println!("  Speed: {:.1} tok/s", new2.len() as f32 / dur2.as_secs_f32().max(0.001));
    println!("  Decoded: {:?}", &dec2[..dec2.len().min(120)]);
    println!("  Is empty? {}", dec2.trim().is_empty());

    // Test 3: Sample logits for first gen step
    let logits = model.debug_logits(&ids2);
    let nan = logits.iter().filter(|v| v.is_nan()).count();
    let (max_id, max_val) = logits.iter().enumerate().fold((0, f32::NEG_INFINITY),
        |m, (i, &v)| if v > m.1 { (i, v) } else { m });
    println!("\nLogits check: NaN={}, max_id={} val={:.3}", nan, max_id, max_val);
    println!("  vocab[{}] = {:?}", max_id, tok.id_to_token(max_id));
}
