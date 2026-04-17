import json
with open(r'D:\IdeaProjects\decentral-ai\data\unified_train.jsonl', 'rb') as f:
    lines = f.readlines()

# Find some humaneval and synthetic samples
count = 0
for line in lines:
    rec = json.loads(line)
    if rec['source'] in ['humaneval', 'synthetic_variations']:
        print(f'=== {rec["source"]} ===')
        print('  prompt:', rec['prompt'][:80])
        print('  solution:', rec['solution'][:150])
        print()
        count += 1
        if count >= 2:
            break