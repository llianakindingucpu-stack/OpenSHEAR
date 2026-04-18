"""
DecentralAI OpenAI-Compatible API Server
==========================================
Every node exposes an OpenAI-compatible API, so existing tools
(ChatGPT clients, LangChain, etc.) can use DecentralAI without changes.

Endpoints:
  POST /v1/chat/completions
  POST /v1/completions
  GET  /v1/models
  GET  /health
  GET  /status

Run:
  python api_server.py --port 8000
"""

import argparse
import json
import os
import sys
import time
import uuid
import threading
import mimetypes
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, r'D:\pylib')

from core import (
    Node, NodeIdentity, NodeLevel, NodeCapabilities,
    ExpertModel, ModelArchitecture,
    InferenceRequest, InferenceResponse, RequestType,
    Router, Verifier, CreditLedger,
)


# ============================================================
# 1. Request/Response Models (OpenAI format)
# ============================================================

def make_chat_completion(model: str, content: str, prompt_tokens: int = 0,
                         completion_tokens: int = 0, finish_reason: str = "stop") -> Dict:
    """Create OpenAI-compatible chat completion response"""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
            },
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    }


def make_completion(model: str, text: str, prompt_tokens: int = 0,
                    completion_tokens: int = 0, finish_reason: str = "stop") -> Dict:
    """Create OpenAI-compatible completion response"""
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:12]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "text": text,
            "index": 0,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    }


def make_models_list(experts: List[ExpertModel]) -> Dict:
    """Create OpenAI-compatible models list"""
    models = []
    for e in experts:
        models.append({
            "id": e.base_model,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "decentral-ai",
            "permission": [],
        })
    # Always include the network model
    models.append({
        "id": "decentral-ai",
        "object": "model",
        "created": int(time.time()),
        "owned_by": "decentral-ai",
        "permission": [],
    })
    return {
        "object": "list",
        "data": models,
    }


# ============================================================
# 2. Inference Engine (connects API to Node)
# ============================================================

class InferenceEngine:
    """
    Bridges the API layer to the node's inference pipeline.
    
    For now, uses the RWKV model directly.
    In production, this routes through the Router to find the best expert.
    """
    
    def __init__(self, node: Node):
        self.node = node
        self.rwkv_model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load RWKV model if available"""
        model_path = r'D:\pylib\rwkv-4-169m-native.pth'
        tokenizer_path = r'D:\pylib\tokenizer.json'
        
        if not os.path.exists(model_path):
            print(f"  Model not found: {model_path}")
            print(f"  API will return mock responses")
            return
        
        try:
            from rwkv.model import RWKV
            
            print(f"  Loading model: {model_path}")
            self.rwkv_model = RWKV(model=model_path, strategy='cpu fp32')
            print(f"  Model loaded")
            
            # Load tokenizer
            if os.path.exists(tokenizer_path):
                try:
                    from tokenizers import Tokenizer
                    self.tokenizer = Tokenizer.from_file(tokenizer_path)
                    print(f"  Tokenizer loaded (HF tokenizers)")
                except ImportError:
                    # Fallback to TRIE tokenizer
                    try:
                        from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
                        vocab_path = r'D:\pylib\rwkv4-world-tok\rwkv_vocab_v20230424.txt'
                        if os.path.exists(vocab_path):
                            self.tokenizer = TRIE_TOKENIZER(vocab_path)
                            print(f"  Tokenizer loaded (TRIE)")
                    except Exception as e2:
                        print(f"  Tokenizer fallback failed: {e2}")
        except Exception as e:
            print(f"  Model load failed: {e}")
            print(f"  API will return mock responses")
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7,
                 top_p: float = 0.9) -> tuple:
        """
        Generate text from prompt.
        Returns (generated_text, prompt_tokens, completion_tokens)
        """
        if self.rwkv_model is None:
            # Mock response
            return f"[DecentralAI Mock] Prompt received: {prompt[:50]}...", len(prompt) // 4, 20
        
        try:
            # Tokenize
            if self.tokenizer is not None:
                if hasattr(self.tokenizer, 'encode'):
                    if hasattr(self.tokenizer, 'decode'):
                        # HF Tokenizer
                        encoded = self.tokenizer.encode(prompt)
                        tokens = encoded.ids if hasattr(encoded, 'ids') else encoded
                    else:
                        # TRIE tokenizer
                        tokens = self.tokenizer.encode(prompt)
                else:
                    tokens = [0]  # Fallback
            else:
                tokens = [0]
            
            prompt_token_count = len(tokens)
            
            # Generate
            generated_tokens = []
            state = None
            
            # Feed prompt
            out, state = self.rwkv_model.forward(tokens, state)
            
            # Generate continuation
            for _ in range(max_tokens):
                # Sample next token
                logits = out.float().numpy()
                
                # Temperature
                if temperature > 0:
                    logits = logits / temperature
                
                # Top-p sampling
                import numpy as np
                probs = np.exp(logits - np.max(logits))
                probs = probs / probs.sum()
                sorted_indices = np.argsort(-probs)
                cumulative = np.cumsum(probs[sorted_indices])
                cutoff = sorted_indices[cumulative > top_p]
                probs[cutoff] = 0
                probs = probs / probs.sum()
                
                next_token = int(np.random.choice(len(probs), p=probs))
                generated_tokens.append(next_token)
                
                # Stop on EOS
                if next_token == 0:
                    break
                
                # Continue generation
                out, state = self.rwkv_model.forward([next_token], state)
            
            # Decode
            if self.tokenizer is not None and hasattr(self.tokenizer, 'decode'):
                if hasattr(self.tokenizer, 'decode'):
                    if isinstance(self.tokenizer, type) or hasattr(self.tokenizer, 'decode'):
                        try:
                            # HF Tokenizer
                            text = self.tokenizer.decode(generated_tokens)
                        except:
                            # TRIE tokenizer - decode tokens one by one
                            text = self.tokenizer.decode(generated_tokens)
                    else:
                        text = str(generated_tokens)
                else:
                    text = str(generated_tokens)
            else:
                text = str(generated_tokens)
            
            return text, prompt_token_count, len(generated_tokens)
            
        except Exception as e:
            return f"[Error: {e}]", 0, 0


# ============================================================
# 3. HTTP Handler
# ============================================================

class APIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for OpenAI-compatible API"""
    
    node: Node = None  # Set by server
    engine: InferenceEngine = None
    
    def _send_json(self, data: Dict, status: int = 200):
        """Send JSON response"""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        response = json.dumps(data, ensure_ascii=False)
        self.wfile.write(response.encode('utf-8'))
    
    def _read_body(self) -> Dict:
        """Read and parse JSON body"""
        length = int(self.headers.get('Content-Length', 0))
        if length == 0:
            return {}
        body = self.rfile.read(length)
        return json.loads(body.decode('utf-8'))
    
    def do_OPTIONS(self):
        """CORS preflight"""
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()
    
    def do_GET(self):
        # Serve dashboard static files
        if self.path == '/' or self.path == '/dashboard':
            self._serve_file('/index.html')
        elif self.path.startswith('/static/'):
            self._serve_file(self.path)
        elif self.path == '/health':
            self._send_json({'status': 'ok', 'node': self.node.identity.node_id})
        elif self.path == '/v1/models':
            self._send_json(make_models_list(self.node.experts))
        elif self.path == '/status':
            self._send_json(self.node.get_status())
        else:
            self._send_json({'error': f'Not found: {self.path}'}, 404)
    
    def _serve_file(self, path: str):
        """Serve a static file from the dashboard directory"""
        # Resolve file path
        if path == '/index.html':
            path = '/dashboard/index.html'
        
        base_dir = os.path.join(os.path.dirname(__file__))
        file_path = base_dir + path.replace('/', '\\')
        
        if not os.path.exists(file_path):
            self._send_json({'error': f'Not found: {path}'}, 404)
            return
        
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'
        
        with open(file_path, 'rb') as f:
            content = f.read()
        
        self.send_response(200)
        self.send_header('Content-Type', mime_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(content)
    
    def do_POST(self):
        try:
            if self.path == '/v1/chat/completions':
                self._handle_chat_completions()
            elif self.path == '/v1/completions':
                self._handle_completions()
            else:
                self._send_json({'error': f'Not found: {self.path}'}, 404)
        except Exception as e:
            self._send_json({'error': str(e)}, 500)
    
    def _handle_chat_completions(self):
        """Handle /v1/chat/completions"""
        body = self._read_body()
        
        messages = body.get('messages', [])
        model = body.get('model', 'decentral-ai')
        max_tokens = body.get('max_tokens', 512)
        temperature = body.get('temperature', 0.7)
        top_p = body.get('top_p', 0.9)
        stream = body.get('stream', False)
        
        # Extract prompt from messages
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        prompt = "\n".join(prompt_parts)
        
        # Detect request type
        request_type = RequestType.GENERAL_CHAT
        if any(kw in prompt.lower() for kw in ['code', 'function', 'def ', 'class ', 'import ']):
            request_type = RequestType.CODE_GENERATION
        elif any(kw in prompt.lower() for kw in ['math', 'calculate', 'solve']):
            request_type = RequestType.MATH_REASONING
        
        # Route through node
        request = InferenceRequest(
            request_type=request_type,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        
        # Generate
        text, prompt_tokens, completion_tokens = self.engine.generate(
            prompt, max_tokens, temperature, top_p
        )
        
        # Update node stats
        if self.node.experts:
            self.node.experts[0].total_requests += 1
        
        # Format response
        if stream:
            # SSE stream (simplified - just send complete response)
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            chunk = make_chat_completion(model, text, prompt_tokens, completion_tokens)
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode('utf-8'))
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            response = make_chat_completion(model, text, prompt_tokens, completion_tokens)
            self._send_json(response)
    
    def _handle_completions(self):
        """Handle /v1/completions"""
        body = self._read_body()
        
        prompt = body.get('prompt', '')
        model = body.get('model', 'decentral-ai')
        max_tokens = body.get('max_tokens', 512)
        temperature = body.get('temperature', 0.7)
        top_p = body.get('top_p', 0.9)
        
        text, prompt_tokens, completion_tokens = self.engine.generate(
            prompt, max_tokens, temperature, top_p
        )
        
        response = make_completion(model, text, prompt_tokens, completion_tokens)
        self._send_json(response)
    
    def log_message(self, format, *args):
        """Override to reduce noise"""
        pass  # Silent


# ============================================================
# 4. Server
# ============================================================

def run_server(node: Node, port: int = 8000):
    """Start the API server"""
    
    print("=" * 60)
    print("DecentralAI API Server")
    print("=" * 60)
    
    # Create inference engine
    print("\n[1/3] Initializing inference engine...")
    engine = InferenceEngine(node)
    
    # Configure handler
    APIHandler.node = node
    APIHandler.engine = engine
    
    # Start server
    print(f"\n[2/3] Node: {node.identity.node_id} ({node.identity.capabilities.level.name})")
    print(f"      Experts: {len(node.experts)}")
    for e in node.experts:
        print(f"        - {e.base_model} ({e.architecture.value}, {e.param_count_mb}MB)")
    
    server = HTTPServer(('0.0.0.0', port), APIHandler)
    
    print(f"\n[3/3] Server started on port {port}")
    print(f"\nEndpoints:")
    print(f"  GET  http://localhost:{port}/          Dashboard")
    print(f"  POST http://localhost:{port}/v1/chat/completions")
    print(f"  POST http://localhost:{port}/v1/completions")
    print(f"  GET  http://localhost:{port}/v1/models")
    print(f"  GET  http://localhost:{port}/health")
    print(f"  GET  http://localhost:{port}/status")
    print(f"\nCompatible with OpenAI SDK:")
    print(f'  openai.api_base = "http://localhost:{port}/v1"')
    print(f'  openai.api_key = "any"  # DecentralAI is open')
    print(f"\nPress Ctrl+C to stop\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


def main():
    parser = argparse.ArgumentParser(description="DecentralAI API Server")
    parser.add_argument('--port', type=int, default=8000, help='Port to listen on')
    parser.add_argument('--level', choices=['L0', 'L1', 'L2', 'L3', 'L4'],
                       default='L1', help='Node level')
    args = parser.parse_args()
    
    # Create node
    level_map = {
        'L0': NodeLevel.L0_COLLECTOR, 'L1': NodeLevel.L1_LIGHT_INFERENCE,
        'L2': NodeLevel.L2_STANDARD_INFERENCE, 'L3': NodeLevel.L3_HEAVY_INFERENCE,
        'L4': NodeLevel.L4_DATA_CENTER
    }
    level = level_map[args.level]
    
    caps = NodeCapabilities(
        level=level,
        architectures=[ModelArchitecture.RWKV],
        max_memory_mb=8192,
        compute_flops=50e9,
        bandwidth_mbps=100,
        storage_gb=100,
    )
    node = Node(NodeIdentity(capabilities=caps))
    
    # Add expert
    node.add_expert(ExpertModel(
        base_model="rwkv-4-169m-pile",
        architecture=ModelArchitecture.RWKV,
        domain="general",
        param_count_mb=646,
        avg_tokens_per_sec=5.8,
    ))
    
    run_server(node, args.port)


if __name__ == "__main__":
    main()
