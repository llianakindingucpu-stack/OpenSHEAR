"""Quick test: RWKV inference on this machine"""
import sys, os, time
sys.path.insert(0, r'D:\pylib')
os.environ['PYTHONIOENCODING'] = 'utf-8'

from rwkv.model import RWKV
from tokenizers import Tokenizer

print('Loading model...')
t0 = time.time()
model = RWKV(model=r'D:\pylib\rwkv-4-169m-native.pth', strategy='cpu fp32')
print(f'Model loaded in {time.time()-t0:.1f}s')

print('Loading tokenizer...')
tok = Tokenizer.from_file(r'D:\pylib\tokenizer.json')
print(f'Tokenizer loaded, vocab size: {tok.get_vocab_size()}')

# Simple inference test
ctx = "The meaning of life is"
encoded = tok.encode(ctx)
tokens = encoded.ids
print(f'Input: "{ctx}" -> {tokens}')

# Generate tokens
state = None
current_tokens = tokens
generated = []
t0 = time.time()
for i in range(30):
    out, state = model.forward(current_tokens, state)
    token = out.argmax().item()
    generated.append(token)
    current_tokens = [token]

elapsed = time.time() - t0
all_ids = tokens + generated
result = tok.decode(all_ids)
print(f'Output: {result}')
print(f'Generation: 30 tokens in {elapsed:.1f}s ({30/elapsed:.1f} tok/s)')
print('SUCCESS: RWKV inference works on this machine!')
