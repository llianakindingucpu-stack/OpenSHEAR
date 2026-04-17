import sys, json, os
sys.path.insert(0, r'D:\pylib')

# Check data files
data_dir = r'D:\IdeaProjects\decentral-ai\data'
for f in ['unified_train.jsonl', 'unified_val.jsonl', 'HumanEval.jsonl']:
    fp = os.path.join(data_dir, f)
    if os.path.exists(fp):
        n = sum(1 for _ in open(fp, 'rb'))
        print(f'{f}: {n} lines')

# Check model files
model_dir = r'D:\pylib'
print('\nModels in D:\\pylib:')
for f in os.listdir(model_dir):
    if f.endswith('.pth') or f.endswith('.gguf'):
        sz = os.path.getsize(os.path.join(model_dir, f)) / 1024 / 1024
        print(f'  {f}: {sz:.1f} MB')