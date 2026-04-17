import json, sys
sys.path.insert(0, r'D:\pylib')

# Read a few samples
with open(r'D:\IdeaProjects\decentral-ai\data\unified_train.jsonl', 'rb') as f:
    lines = f.readlines()[:3]

for i, line in enumerate(lines):
    rec = json.loads(line)
    print(f'=== Sample {i+1} ===')
    print('  source:', rec.get('source'))
    print('  prompt:', rec.get('prompt', '')[:120])
    print('  solution:', rec.get('solution', '')[:180])
    print()