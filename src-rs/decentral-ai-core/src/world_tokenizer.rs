//! World tokenizer for RWKV-4-World models.
//!
//! Simple vocabulary mapping (no BPE merges).
//! Loads: data/id2token_world.txt

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

/// Convert a hex digit byte ('0'-'9', 'A'-'F') to its numeric value. Returns 0 for invalid.
fn hex_val(b: u8) -> u8 {
    match b {
        b'0'..=b'9' => b - b'0',
        b'A'..=b'F' => b - b'A' + 10,
        b'a'..=b'f' => b - b'a' + 10,
        _ => 0,
    }
}

/// RWKV World Tokenizer (simple vocab, no BPE)
#[derive(Debug, Clone)]
pub struct WorldTokenizer {
    /// vocab[id] = token bytes
    vocab: Vec<Vec<u8>>,
    /// reverse_map[token_bytes] = id
    vocab_rev: HashMap<Vec<u8>, usize>,
}

impl WorldTokenizer {
    /// Load from data directory.
    pub fn load(data_dir: &str) -> io::Result<Self> {
        let vocab_path = Path::new(data_dir).join("id2token_world.txt");
        Self::load_from_file(&vocab_path)
    }

    pub fn load_from_file(path: &Path) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;

        let mut vocab = Vec::with_capacity(65536);

        // Split on \n manually to avoid UTF-8 decoding issues
        let mut line_start = 0;
        for (i, &byte) in buf.iter().enumerate() {
            if byte == b'\n' {
                let line_bytes = &buf[line_start..i];
                // Skip \r if present
                let end = if line_bytes.ends_with(&[b'\r']) {
                    line_bytes.len() - 1
                } else {
                    line_bytes.len()
                };
                let line = &line_bytes[..end];

                if !line.is_empty() {
                    if let Some(tab_pos) = line.iter().position(|&b| b == b'\t') {
                        let id_bytes = &line[..tab_pos];
                        let token_part = &line[tab_pos + 1..];

                        if let Ok(id_str) = std::str::from_utf8(id_bytes) {
                            if let Ok(id) = id_str.parse::<usize>() {
                                        // Decode percent-escapes in raw token bytes
                                let token_bytes = Self::unescape_bytes(token_part);

                                while vocab.len() <= id {
                                    vocab.push(vec![]);
                                }
                                if id < vocab.len() {
                                    vocab[id] = token_bytes;
                                }
                            }
                        }
                    }
                }
                line_start = i + 1;
            }
        }

        // Build reverse map
        let mut vocab_rev = HashMap::with_capacity(vocab.len());
        for (id, token) in vocab.iter().enumerate() {
            if !token.is_empty() {
                vocab_rev.insert(token.clone(), id);
            }
        }

        eprintln!("[WorldTokenizer] Loaded {} tokens", vocab.len());
        Ok(WorldTokenizer { vocab, vocab_rev })
    }

    /// Decode percent-encoded bytes in token data.
    /// Works on raw bytes, not UTF-8 strings — handles non-UTF-8 tokens like \xf0\x9f.
    fn unescape_bytes(raw: &[u8]) -> Vec<u8> {
        let mut result = Vec::with_capacity(raw.len());
        let mut i = 0;
        while i < raw.len() {
            if raw[i] == b'%' && i + 2 < raw.len() {
                let hex = [raw[i + 1], raw[i + 2]];
                let hi = hex[0].to_ascii_uppercase();
                let lo = hex[1].to_ascii_uppercase();
                let val = (hex_val(hi) << 4) | hex_val(lo);
                // hex_val returns 0 for invalid digits; check validity via range
                let hi_valid = raw[i+1].is_ascii_hexdigit();
                let lo_valid = raw[i+2].is_ascii_hexdigit();
                if hi_valid && lo_valid {
                    result.push(val);
                    i += 3;
                    continue;
                }
            }
            result.push(raw[i]);
            i += 1;
        }
        result
    }

    #[cfg(test)]
    fn unescape_token(s: &str) -> Vec<u8> {
        // Kept for tests — delegates to byte version
        Self::unescape_bytes(s.as_bytes())
    }

    /// Encode text to token ids
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let text_bytes = text.as_bytes();
        let mut ids = Vec::new();
        let mut pos = 0;

        while pos < text_bytes.len() {
            // Try to match longest token
            let mut best_len = 0;
            let mut best_id = 0;

            for (token, &id) in &self.vocab_rev {
                if pos + token.len() <= text_bytes.len() && token.len() > best_len {
                    if &text_bytes[pos..pos + token.len()] == token.as_slice() {
                        best_len = token.len();
                        best_id = id;
                    }
                }
            }

            if best_len > 0 {
                ids.push(best_id);
                pos += best_len;
            } else {
                // Unknown byte, skip (or use byte fallback)
                ids.push(0);  // or could use byte as id if vocab includes all bytes
                pos += 1;
            }
        }

        ids
    }

    /// Decode token ids to text
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if id < self.vocab.len() {
                bytes.extend(&self.vocab[id]);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get the token bytes for a given id.
    pub fn get_token(&self, id: usize) -> Option<&Vec<u8>> {
        self.vocab.get(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unescape() {
        // Actual format in id2token_world.txt uses %XX percent-encoding
        assert_eq!(WorldTokenizer::unescape_bytes(b"%0A"), vec![0x0A]);
        assert_eq!(WorldTokenizer::unescape_bytes(b"%0D"), vec![0x0D]);
        assert_eq!(WorldTokenizer::unescape_bytes(b"%00"), vec![0x00]);
        assert_eq!(WorldTokenizer::unescape_bytes(b"%09"), vec![0x09]);
        assert_eq!(WorldTokenizer::unescape_bytes(b"abc"), vec![b'a', b'b', b'c']);
        // Literal bytes pass through
        assert_eq!(WorldTokenizer::unescape_bytes(b"hello world"), b"hello world".to_vec());
    }
}
