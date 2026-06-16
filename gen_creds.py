import json
with open('medusa-firebase-key.json', 'r') as f:
    creds = json.load(f)
lines = ['firebase_creds = {']
for k, v in creds.items():
    lines.append(f'    \"{k}\": \"{v}\",')
lines.append('}')
print('\n'.join(lines))
