//! Standalone tokenizer test - no P2P, no network, just load & test World tokenizer.

fn main() {
    println!("Loading World tokenizer...");

    let data_dir = std::path::Path::new(
        "D:\\IdeaProjects\\decentral-ai\\src-rs\\decentral-ai-core\\data",
    );

    use decentral_ai_core::world_tokenizer::WorldTokenizer;

    match WorldTokenizer::load_from_file(&data_dir.join("id2token_world.txt")) {
        Ok(tokenizer) => {
            println!("Loaded {} tokens", tokenizer.vocab_size());

            // Spot checks
            println!("\n=== Spot checks ===");

            // ID 33 should be space
            if let Some(token) = tokenizer.get_token(33) {
                println!("ID 33: {:?}", std::str::from_utf8(token).unwrap_or("?"));
                println!("ID 33 bytes: {:?}", token);
            }

            // ID 1 should be \x00
            if let Some(token) = tokenizer.get_token(1) {
                println!("ID 1: {:?}", token);
            }

            // Test encode: "from typing import List"
            println!("\n=== Encoding test ===");
            let test_str = "from typing import List";
            let ids = tokenizer.encode(test_str);
            println!("'{}' => {} tokens", test_str, ids.len());
            let show = ids.len().min(20);
            println!("First {} IDs: {:?}", show, &ids[..show]);

            // Decode back
            let decoded = tokenizer.decode(&ids);
            println!("Decoded: '{}'", decoded);

            // Test with spaces
            println!("\n=== Space test ===");
            let space_ids = tokenizer.encode("a b c");
            println!("'a b c' => {} tokens: {:?}", space_ids.len(), space_ids);
            let space_decoded = tokenizer.decode(&space_ids);
            println!("Decoded: '{}'", space_decoded);

            // Verify space token id
            println!("\n=== Finding space token id ===");
            let space_id: Option<usize> = (0..tokenizer.vocab_size())
                .find(|&i| tokenizer.get_token(i).map(|t| t.as_slice() == b" ").unwrap_or(false));
            println!("Space token id: {:?}", space_id);
        }
        Err(e) => {
            eprintln!("ERROR loading tokenizer: {}", e);
        }
    }
}
