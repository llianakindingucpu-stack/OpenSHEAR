import json

with open(r'D:\IdeaProjects\decentral-ai\data\HumanEval.jsonl', 'r', encoding='utf-8') as f:
    line = f.readline()

d = json.loads(line)
print("Keys:", list(d.keys()))
print()
print("=== prompt (first 400 chars) ===")
print(d['prompt'][:400])
print()
print("=== test (first 300 chars) ===")
print(d['test'][:300])
print()
print("=== entry_point ===")
print(d['entry_point'])
