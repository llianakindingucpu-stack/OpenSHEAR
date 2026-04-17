"""Test RWKV-4-World-430M inference"""
import sys, os, time
sys.path.insert(0, r'D:\pylib')
os.environ['PYTHONIOENCODING'] = 'utf-8'

from rwkv.model import RWKV
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER

print('Loading RWKV-4-World-430M...')
t0 = time.time()
model = RWKV(model=r'D:\pylib\rwkv-4-world-430m-native.pth', strategy='cpu fp32')
print(f'Model loaded in {time.time()-t0:.1f}s')

print('Loading tokenizer...')
vocab_path = r'D:\pylib\rwkv4-world-tok\rwkv_vocab_v20230424.txt'
tokenizer = TRIE_TOKENIZER(vocab_path)
print(f'Tokenizer loaded!')

# Test 1: General text
ctx = "The meaning of life is"
tokens = tokenizer.encode(ctx)
state = None
out, state = model.forward(tokens, state)
generated = []
t0 = time.time()
for i in range(50):
    import torch
    logits = out.float() / 0.8
    probs = torch.softmax(logits, dim=-1)
    token = torch.multinomial(probs, 1).item()
    generated.append(token)
    out, state = model.forward([token], state)
elapsed = time.time() - t0
result = tokenizer.decode(tokens + generated)
print(f'\n[General] "{result}"')
print(f'Speed: {50/elapsed:.1f} tok/s')

# Test 2: Code generation
ctx = "def add(a, b):\n    "
tokens = tokenizer.encode(ctx)
state = None
out, state = model.forward(tokens, state)
generated = []
t0 = time.time()
for i in range(50):
    logits = out.float() / 0.2  # low temp for code
    token = logits.argmax().item()
    generated.append(token)
    out, state = model.forward([token], state)
elapsed = time.time() - t0
result = tokenizer.decode(tokens + generated)
print(f'\n[Code] {result}')
print(f'Speed: {50/elapsed:.1f} tok/s')

# Test 3: Chinese
ctx = "\u4eba\u5de5\u667a\u80fd\u7684\u672a\u6765"
tokens = tokenizer.encode(ctx)
state = None
out, state = model.forward(tokens, state)
generated = []
t0 = time.time()
for i in range(30):
    logits = out.float() / 0.8
    probs = torch.softmax(logits, dim=-1)
    token = torch.multinomial(probs, 1).item()
    generated.append(token)
    out, state = model.forward([token], state)
elapsed = time.time() - t0
result = tokenizer.decode(tokens + generated)
print(f'\n[Chinese] {result}')
print(f'Speed: {30/elapsed:.1f} tok/s')

print('\nSUCCESS: RWKV-4-World-430M works on this machine!')
