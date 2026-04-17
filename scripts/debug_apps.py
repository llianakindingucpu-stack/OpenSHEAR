import json

with open(r'D:\IdeaProjects\decentral-ai\data\APPS_train_sample.jsonl', 'r', encoding='utf-8') as f:
    line = f.readline()
    d = json.loads(line)

print('solutions type:', type(d['solutions']))
print('solutions is list:', isinstance(d['solutions'], list))
if isinstance(d['solutions'], list) and len(d['solutions']) > 0:
    print('solutions[0] type:', type(d['solutions'][0]))
    print('First 300 chars of solutions[0]:')
    print(repr(d['solutions'][0][:300]))