import sys, json
sys.path.insert(0, r'D:\pylib')

files = [
    r'D:\IdeaProjects\decentral-ai\data\HumanEval.jsonl',
    r'D:\IdeaProjects\decentral-ai\data\synthetic_variations.jsonl',
    r'D:\IdeaProjects\decentral-ai\data\APPS_train_sample.jsonl',
]
for fp in files:
    fn = fp.split('\\')[-1]
    with open(fp, 'rb') as f:
        lines = f.readlines()
    print(f'\n=== {fn} ===')
    print(f'  Total: {len(lines)}')
    first = json.loads(lines[0].decode('utf-8', 'replace'))
    print(f'  Keys: {list(first.keys())}')
    snippet = json.dumps(first, ensure_ascii=False)[:280]
    print(f'  Sample: {snippet}')
