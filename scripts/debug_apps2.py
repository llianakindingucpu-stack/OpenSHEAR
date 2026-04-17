import json

with open(r'D:\IdeaProjects\decentral-ai\data\APPS_train_sample.jsonl', 'r', encoding='utf-8') as f:
    line = f.readline()
    d = json.loads(line)

print('solutions type:', type(d['solutions']))
print('solutions first 500 chars:')
print(repr(d['solutions'][:500]))