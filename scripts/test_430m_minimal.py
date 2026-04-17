"""Minimal test: RWKV-4-World-430M single inference"""
import sys, os, time
sys.path.insert(0, r'D:\pylib')
os.environ['PYTHONIOENCODING'] = 'utf-8'

from rwkv.model import RWKV
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
import torch

print('Loading RWKV-4-World-430M (fp32)...')
t0 = time.time()
model = RWKV(model=r'D:\pylib\rwkv-4-world-430m-native.pth', strategy='cpu fp32')
print(f'Loaded in {time.time()-t0:.1f}s')

vocab_path = r'D:\pylib\rwkv4-world-tok\rwkv_vocab_v20230424.txt'
tokenizer = TRIE_TOKENIZER(vocab_path)
print('Tokenizer ready')

# Single generation: 20 tokens only
ctx = "def add(a, b):"
tokens = tokenizer.encode(ctx)
print(f'Prompt: "{ctx}" -> {tokens[:5]}...')

state = None
out, state = model.forward(tokens, state)
generated = []
t0 = time.time()
for i in range(20):
    token = out.float().argmax().item()
    generated.append(token)
    out, state = model.forward([token], state)
elapsed = time.time() - t0
result = tokenizer.decode(tokens + generated)
print(f'Output: {result}')
print(f'Speed: {20/elapsed:.1f} tok/s ({elapsed:.1f}s)')
print('DONE')
