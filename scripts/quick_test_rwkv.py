"""Quick check: can we load the RWKV model?"""
import sys
sys.path.insert(0, r'D:\pylib')

print('[1] Importing RWKV...')
from rwkv.model import RWKV
print('[2] Loading model...')
model = RWKV(model=r'D:\pylib\rwkv-4-169m-native.pth', strategy='cpu fp32')
print('[3] Model loaded OK')
print(f'  Model keys: {list(model.w.keys())[:5]}...')