import sys, time, json, urllib.request

# Test direct worker call
url = "http://127.0.0.1:8081/v1/chat/completions"
data = {
    "model": "rwkv-4-169m",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 10,
    "temperature": 0.7
}
req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"}, method="POST")
try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
        print(f"SUCCESS: {result}")
except Exception as e:
    print(f"ERROR: {e}")
    if hasattr(e, 'read'):
        print(f"Body: {e.read().decode()}")
