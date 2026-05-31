import json

data = '{"test": "val"}'
payload = json.loads(data)
print("Payload loaded successfully:", payload)
