"""
DecentralAI 推理 Worker（内置 http.server，无需额外依赖）
"""
import os, sys, time, json, torch
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(__file__))
from rwkv_engine import RWKV4, load_model, RWKVTokenizer

MODEL_PATH = os.environ.get("RWKV_MODEL", "D:/pylib/rwkv-4-169m-native.pth")

# 全局
model = None
tokenizer = None
model_loaded = False

def load_engine():
    global model, tokenizer, model_loaded
    if model_loaded:
        return
    print("[Worker] Loading RWKV...")
    t0 = time.time()
    model, _ = load_model(MODEL_PATH)
    tokenizer = RWKVTokenizer()
    model_loaded = True
    print(f"[Worker] Ready in {time.time()-t0:.1f}s")

def run_inference(prompt: str, max_new: int = 50, temperature: float = 0.7, vocab_size: int = 65536) -> tuple:
    global model, tokenizer
    tokens = tokenizer.encode(prompt)
    # Clip to valid vocab range (tokenizer may produce IDs > model vocab size)
    tokens = [t for t in tokens if 0 <= t < vocab_size]
    if not tokens:
        tokens = [0]  # fallback to BOS
    tok_tensor = torch.tensor([tokens], dtype=torch.long)
    state = None
    with torch.no_grad():
        logits, state = model(tok_tensor, state)
    output = list(tokens)
    for _ in range(max_new):
        lp = logits[0, -1, :].float() / max(temperature, 0.01)
        s, idx = torch.sort(lp, descending=True)
        p = torch.softmax(s, dim=-1)
        cs = torch.cumsum(p, dim=-1)
        m = cs > 0.9
        m[0] = False
        s2 = s.masked_fill(m, float("-inf"))
        p2 = torch.softmax(s2, dim=-1)
        try:
            nt = idx[torch.multinomial(p2, 1).item()].item()
        except Exception:
            nt = idx[0].item()
        output.append(nt)
        tok_tensor = torch.tensor([[nt]], dtype=torch.long)
        with torch.no_grad():
            logits, state = model(tok_tensor, state)
        if nt == 0:
            break
    return tokenizer.decode(output[len(tokens):]), len(tokens)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # 静默

    def do_GET(self):
        if self.path == "/health":
            self.send_json({"status": "ok", "model_loaded": model_loaded,
                           "model": MODEL_PATH.split("/")[-1] if model_loaded else None})
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return

        if not model_loaded:
            load_engine()

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
        except Exception:
            data = {}

        messages = data.get("messages", [])
        prompt = ""
        if messages:
            prompt = "\n".join(
                f"{'User' if m.get('role')=='user' else 'Assistant'}: {m.get('content','')}"
                for m in messages
            )
            prompt += "\nAssistant: "
        else:
            prompt = data.get("prompt", "")

        max_tokens = min(data.get("max_tokens", 64), 256)
        temperature = max(data.get("temperature", 0.7), 0.1)

        t0 = time.time()
        try:
            content, pt = run_inference(prompt, max_tokens, temperature)
            elapsed = time.time() - t0
            ct = len(tokenizer.encode(content))
            resp = {
                "id": f"rwkv-{int(t0*1000)}",
                "object": "chat.completion",
                "created": int(t0),
                "model": MODEL_PATH.split("/")[-1],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
                "latency_ms": round(elapsed * 1000),
            }
            self.send_json(resp)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.send_json({"error": str(e)}, code=500)

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8081)
    args = p.parse_args()

    load_engine()
    server = HTTPServer((args.host, args.port), Handler)
    print(f"[Worker] RWKV serving on {args.host}:{args.port}")
    server.serve_forever()
