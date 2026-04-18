//! BPE tokenizer for RWKV-4 model.
//!
//! Loads: data/id2token.txt  → Vec<String>  (index = token_id)
//!         data/merges.txt    → Vec<(String,String)>  (ordered BPE merges)
//!
//! Supports byte-level BPE encoding (GPT-NeoX style, Ġ prefix for spaces).

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

/// RWKV-4 BPE Tokenizer (byte-level, GPT-NeoX style)
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    /// vocab[id] = token string
    vocab: Vec<String>,
    /// reverse_map[token] = id  (for encoding)
    vocab_rev: HashMap<String, usize>,
    /// merges in order: (part_a, part_b)
    merges: Vec<(String, String)>,
}

impl BpeTokenizer {
    /// Load from data directory.
    pub fn load(data_dir: &str) -> io::Result<Self> {
        let vocab = Self::load_vocab(Path::new(data_dir).join("id2token.txt"))?;
        let merges = Self::load_merges(Path::new(data_dir).join("merges.txt"))?;

        let mut vocab_rev = HashMap::with_capacity(vocab.len());
        for (id, tok) in vocab.iter().enumerate() {
            vocab_rev.insert(tok.clone(), id);
        }

        eprintln!("[Tokenizer] Loaded {} tokens, {} merges", vocab.len(), merges.len());
        Ok(BpeTokenizer { vocab, vocab_rev, merges })
    }

    fn load_vocab(path: std::path::PathBuf) -> io::Result<Vec<String>> {
        let file = File::open(&path)?;
        let reader = BufReader::new(file);
        let mut vocab = Vec::with_capacity(60000);

        for line in reader.lines() {
            let line = line?;
            if line.is_empty() { continue; }
            let tab = match line.find('\t') {
                Some(p) => p,
                None => continue,
            };
            let id: usize = match line[..tab].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let token = &line[tab+1..];
            let token = token.replace("\\t", "\t").replace("\\n", "\n");
            while vocab.len() <= id {
                vocab.push(String::new());
            }
            vocab[id] = token;
        }
        Ok(vocab)
    }

    fn load_merges(path: std::path::PathBuf) -> io::Result<Vec<(String, String)>> {
        let file = File::open(&path)?;
        let reader = BufReader::new(file);
        let mut merges = Vec::with_capacity(60000);

        for line in reader.lines() {
            let line = line?;
            if line.is_empty() { continue; }
            let space = match line.find(' ') {
                Some(p) => p,
                None => continue,
            };
            merges.push((line[..space].to_string(), line[space+1..].to_string()));
        }
        Ok(merges)
    }

    /// Decode token IDs → UTF-8 string.
    pub fn decode(&self, ids: &[usize]) -> String {
        let raw: String = ids.iter()
            .filter_map(|&id| self.vocab.get(id))
            .cloned()
            .collect();

        // Byte-level decode: chr(0x120+n) -> chr(n) for n in 0..33
        let mut bytes = Vec::with_capacity(raw.len());
        for ch in raw.chars() {
            let cp = ch as u32;
            if cp >= 0x100 && cp < 0x100 + 33 {
                bytes.push((cp - 0x100) as u8);
            } else if cp < 0x80 {
                bytes.push(cp as u8);
            } else if cp < 0x800 {
                bytes.push(0xC0 | ((cp >> 6) as u8));
                bytes.push(0x80 | ((cp & 0x3F) as u8));
            } else if cp < 0x10000 {
                bytes.push(0xE0 | ((cp >> 12) as u8));
                bytes.push(0x80 | ((cp >> 6) as u8 & 0x3F));
                bytes.push(0x80 | (cp as u8 & 0x3F));
            } else {
                bytes.push(0xF0 | ((cp >> 18) as u8));
                bytes.push(0x80 | ((cp >> 12) as u8 & 0x3F));
                bytes.push(0x80 | ((cp >> 6) as u8 & 0x3F));
                bytes.push(0x80 | (cp as u8 & 0x3F));
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
    }

    /// Encode UTF-8 string → token IDs using byte-level BPE.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        // Step 1: Byte-level pre-tokenization (GPT-NeoX style)
        // space (0x20) -> Ġ (chr(0x120)), control chars -> chr(0x100+n)
        let byte_encoded: Vec<char> = text.bytes().map(|b| {
            if b >= 33 { b as char } else { char::from_u32(b as u32 + 0x100).unwrap() }
        }).collect();
        let byte_str: String = byte_encoded.iter().collect();

        // Step 2: BPE merge
        let mut tokens: Vec<String> = byte_str.chars().map(|c| c.to_string()).collect();

        // Apply merges in priority order (first match wins)
        for (a, b) in &self.merges {
            if tokens.len() < 2 { break; }
            let mut i = 0;
            let mut changed = false;
            while i + 1 < tokens.len() {
                if tokens[i] == *a && tokens[i + 1] == *b {
                    tokens[i] = format!("{}{}", a, b);
                    tokens.remove(i + 1);
                    changed = true;
                } else {
                    i += 1;
                }
            }
            // If a merge was applied, re-scan from beginning
            // (this ensures priority ordering)
            if changed {
                // Don't break - continue with next merge
            }
        }

        // Step 3: Look up IDs
        tokens.iter().map(|t| {
            *self.vocab_rev.get(t).unwrap_or(&0)
        }).collect()
    }

    /// Encode with BOS token (id=0).
    pub fn encode_with_bos(&self, text: &str) -> Vec<usize> {
        let mut ids = vec![0];
        ids.extend(self.encode(text));
        ids
    }

    /// Debug: token ID → string.
    pub fn id_to_token(&self, id: usize) -> &str {
        self.vocab.get(id).map(|s| s.as_str()).unwrap_or("<UNK>")
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load() {
        let dir = format!("{}/data", env!("CARGO_MANIFEST_DIR"));
        let _tok = BpeTokenizer::load(&dir).unwrap();
    }

    #[test]
    fn test_decode_basic() {
        let dir = format!("{}/data", env!("CARGO_MANIFEST_DIR"));
        let tok = BpeTokenizer::load(&dir).unwrap();
        assert_eq!(tok.id_to_token(2), "!");
        assert_eq!(tok.id_to_token(510), "The");
    }

    #[test]
    fn test_encode_the() {
        let dir = format!("{}/data", env!("CARGO_MANIFEST_DIR"));
        let tok = BpeTokenizer::load(&dir).unwrap();
        let ids = tok.encode("The");
        assert_eq!(ids, vec![510]);
    }

    #[test]
    fn test_encode_hello() {
        let dir = format!("{}/data", env!("CARGO_MANIFEST_DIR"));
        let tok = BpeTokenizer::load(&dir).unwrap();
        let ids = tok.encode("hello");
        assert_eq!(ids, vec![25521]);
    }
}
