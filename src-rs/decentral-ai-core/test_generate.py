"""
Minimal test: encode prompt → generate → decode
Verifies that the RWKV generation produces actual NEW content.
"""
import sys
sys.path.insert(0, r'D:\IdeaProjects\decentral-ai\src-rs\decentral-ai-core')

import time
from rwkv_model import RwkvModel
from tokenizer import BpeTokenizer

MODEL_PATH = r'D:\IdeaProjects\decentral-ai\data\rwkv4_169m.bin'
DATA_DIR = r'D:\IdeaProjects\decentral-ai\src-rs\decentral-ai-core\data'

print("Loading tokenizer...")
tokenizer = BpeTokenizer.load(DATA_DIR)
print(f"  Vocab: {tokenizer.id2token.__len__()} tokens")

print("\nLoading model...")
model = RwkvModel.load_from_file(MODEL_PATH)
print(f"  Params: {model.total_params()}")

# Test 1: Encode then decode (should be lossless)
test_str = "def hello():\n    return 42\n"
ids = tokenizer.encode(test_str)
decoded = tokenizer.decode(ids)
print(f"\nTest 1: Encode/Decode")
print(f"  Original: {repr(test_str)}")
print(f"  Decoded:  {repr(decoded)}")
print(f"  Match: {test_str == decoded}")

# Test 2: Generate from a simple prompt
prompt = "def add(a, b):\n    \"\"\""
ids = tokenizer.encode(prompt)
print(f"\nTest 2: Generate from prompt")
print(f"  Prompt: {repr(prompt)}")
print(f"  Token IDs: {ids[:10]}... ({len(ids)} tokens)")

# Generate
gen_ids, duration = model.generate(ids, max_new=20, temperature=0.8)
print(f"  Generated {len(gen_ids) - len(ids)} new tokens in {duration:.2f}s")
print(f"  Speed: {(len(gen_ids) - len(ids)) / duration:.1f} tok/s")

# Decode ALL tokens (like the benchmark does)
all_decoded = tokenizer.decode(gen_ids)
# Decode ONLY new tokens
new_ids = gen_ids[len(ids):]
new_decoded = tokenizer.decode(new_ids)

print(f"\n  ALL decoded ({len(gen_ids)} tok): {repr(all_decoded[:100])}...")
print(f"  NEW  decoded ({len(new_ids)} tok): {repr(new_decoded)}")

# Test 3: Compare with full prompt + generate
prompt_full = "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\""
ids_full = tokenizer.encode(prompt_full)
gen_full, _ = model.generate(ids_full, max_new=50, temperature=0.8)
new_full = gen_full[len(ids_full):]
decoded_new = tokenizer.decode(new_full)
decoded_all = tokenizer.decode(gen_full)
prompt_decoded = tokenizer.decode(ids_full)

print(f"\nTest 3: HumanEval-style prompt")
print(f"  Prompt decoded: {repr(prompt_decoded[:60])}...")
print(f"  New tokens decoded: {repr(decoded_new[:100])}")
print(f"  Is new_decoded == prompt_decoded? {decoded_new == prompt_decoded}")
print(f"  Is new content empty? {len(decoded_new.strip()) == 0}")
