#!/usr/bin/env python3
"""Export RWKV tokenizer to plain text files for Rust."""

import json, sys, os

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'src-rs', 'decentral-ai-core', 'data')
os.makedirs(OUT_DIR, exist_ok=True)

tokenizer_path = os.path.join(os.path.dirname(__file__), '..', '..', 'pylib', 'tokenizer.json')
if not os.path.exists(tokenizer_path):
    tokenizer_path = r'D:/pylib/tokenizer.json'

with open(tokenizer_path, 'r', encoding='utf-8') as f:
    d = json.load(f)

# --- 1. Export id→token (for decode) ---
id2token_path = os.path.join(OUT_DIR, 'id2token.txt')
with open(id2token_path, 'w', encoding='utf-8') as f:
    # Read special tokens from added_tokens
    special = {t['id']: t['content'] for t in d['added_tokens']}
    # Read vocab (token → id)
    vocab = d['model']['vocab']
    # Find max id
    max_id = max(v for _, v in vocab.items())
    # Include special tokens
    for i in range(max_id + 1):
        if i in special:
            token = special[i]
        else:
            # Reverse lookup in vocab
            token = next((t for t, vid in vocab.items() if vid == i), None)
            if token is None:
                token = f'<UNK:{i}>'
        f.write(f"{i}\t{token}\n")
print(f"Wrote id2token: {id2token_path}")

# --- 2. Export merges (for BPE encode) ---
merges_path = os.path.join(OUT_DIR, 'merges.txt')
with open(merges_path, 'w', encoding='utf-8') as f:
    merges = d['model'].get('merges', [])
    for m in merges:
        f.write(m + '\n')
print(f"Wrote merges: {merges_path} ({len(merges)} merges)")

# --- 3. Show sample id→token ---
print("\nSample id2token:")
with open(id2token_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 10: break
        print(f"  {line.rstrip()}")

# --- 4. Verify decode works for known tokens ---
print("\nVerify decode:")
special = {t['id']: t['content'] for t in d['added_tokens']}
vocab_rev = {v: k for k, v in vocab.items()}
for tid, expected in [(0, '<|endoftext|>'), (1, '<|padding|>'), (2, '!'), (3, '"'), (50277, None)]:
    token = special.get(tid) or vocab_rev.get(tid, f'<UNK:{tid}>')
    print(f"  id={tid} → '{token}'")
