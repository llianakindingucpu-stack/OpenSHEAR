"""
DecentralAI - 推理路由桥接
将 Rust API 请求转发到 Python Worker，同时完成 Credits 结算

架构：
  Client → Rust API (:8080) → Router → Python Worker (:8081) → Model
                                           ↓
                                      Credits 结算
                                           ↓
                                      区块链记录
"""

import os
import sys
import json
import time
import sqlite3
import argparse
import logging
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

# ============================================================
# 配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("router")

# Python Worker 地址
PYTHON_WORKER_URL = os.environ.get("PYTHON_WORKER_URL", "http://127.0.0.1:8081")

# 数据库
DB_PATH = Path(__file__).parent / "data" / "router.db"


# ============================================================
# 数据结构
# ============================================================
@dataclass
class InferenceJob:
    id: str
    model: str
    prompt: str
    max_tokens: int
    temperature: float
    credits_cost: float
    status: str  # pending / running / completed / failed
    result: Optional[str]
    latency_ms: float
    timestamp: int


# ============================================================
# 数据库
# ============================================================
def init_db():
    """初始化数据库"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            model TEXT,
            prompt TEXT,
            max_tokens INTEGER,
            temperature REAL,
            credits_cost REAL,
            status TEXT,
            result TEXT,
            latency_ms REAL,
            timestamp INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS credits_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT,
            user_id TEXT,
            amount REAL,
            reason TEXT,
            timestamp INTEGER
        )
    """)
    conn.commit()
    return conn


def get_user_credits(conn: sqlite3.Connection, user_id: str) -> float:
    """获取用户余额"""
    # 简化：每个用户初始 100 Credits
    result = conn.execute(
        "SELECT SUM(amount) FROM credits_log WHERE user_id = ?",
        (user_id,)
    ).fetchone()[0]
    return result if result else 100.0


def deduct_credits(conn: sqlite3.Connection, user_id: str, amount: float, reason: str, job_id: str = ""):
    """扣除 Credits"""
    conn.execute(
        "INSERT INTO credits_log (job_id, user_id, amount, reason, timestamp) VALUES (?, ?, ?, ?, ?)",
        (job_id, user_id, -amount, reason, int(time.time()))
    )
    conn.commit()


# ============================================================
# 推理路由
# ============================================================
class InferenceRouter:
    """推理请求路由器"""
    
    # Credits 费率表（per 1K tokens）
    RATE_TABLE = {
        "rwkv-4-169m": 0.5,   # L1 节点
        "rwkv-4-430m": 0.8,
        "qwen-0.5b": 1.0,
        "qwen-1.5b": 1.5,
        "qwen-7b": 2.5,       # L2 节点
        "default": 1.0,
    }
    
    def __init__(self, worker_url: str):
        self.worker_url = worker_url
        self.conn = init_db()
        log.info(f"路由初始化: Worker={worker_url}, DB={DB_PATH}")
        
    def estimate_cost(self, model: str, max_tokens: int) -> float:
        """估算费用"""
        rate = self.RATE_TABLE.get(model, self.RATE_TABLE["default"])
        return rate * max_tokens / 1000.0
        
    def route(self, messages: list, model: str = "rwkv-4-169m",
              max_tokens: int = 128, temperature: float = 0.7,
              user_id: str = "anonymous") -> Dict[str, Any]:
        """
        路由推理请求
        返回 OpenAI 格式的 response
        """
        job_id = f"job-{int(time.time()*1000)}-{id(messages)}"
        start = time.perf_counter()
        
        # 1. 费用估算
        estimated_cost = self.estimate_cost(model, max_tokens)
        log.info(f"[{job_id}] 用户={user_id}, 模型={model}, 预估费用={estimated_cost:.3f} credits")
        
        # 2. 扣除预授权（简化：先扣后返差）
        self.deduct_credits(user_id, estimated_cost, f"推理预授权: {model}", job_id)
        
        # 3. 转发到 Python Worker
        try:
            # 拼接 prompt
            prompt = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}"
                for m in messages
            )
            
            # 调用 Python Worker
            resp = requests.post(
                f"{self.worker_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                timeout=60
            )
            resp.raise_for_status()
            result = resp.json()
            
            # 4. 计算实际费用
            usage = result.get("usage", {})
            actual_tokens = usage.get("completion_tokens", 0)
            actual_cost = self.estimate_cost(model, actual_tokens)
            
            # 差额退还
            refund = estimated_cost - actual_cost
            if refund > 0:
                self.conn.execute(
                    "INSERT INTO credits_log (job_id, user_id, amount, reason, timestamp) VALUES (?, ?, ?, ?, ?)",
                    (job_id, user_id, refund, f"推理退款: {actual_tokens} tokens", int(time.time()))
                )
                self.conn.commit()
            
            # 5. 记录任务
            latency_ms = (time.perf_counter() - start) * 1000
            self.conn.execute("""
                INSERT INTO jobs (id, model, prompt, max_tokens, temperature, credits_cost, status, result, latency_ms, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, 'completed', ?, ?, ?)
            """, (job_id, model, prompt[:500], max_tokens, temperature, actual_cost,
                  result["choices"][0]["message"]["content"][:500], latency_ms, int(time.time())))
            self.conn.commit()
            
            log.info(f"[{job_id}] 完成: {actual_tokens} tokens, 费用={actual_cost:.3f} credits, 延迟={latency_ms:.0f}ms")
            return result
            
        except requests.exceptions.RequestException as e:
            # Worker 请求失败，退款
            self.conn.execute(
                "INSERT INTO credits_log (job_id, user_id, amount, reason, timestamp) VALUES (?, ?, ?, ?, ?)",
                (job_id, user_id, estimated_cost, f"推理失败退款: {str(e)[:100]}", int(time.time()))
            )
            self.conn.commit()
            
            log.error(f"[{job_id}] Worker 请求失败: {e}")
            return {
                "error": {
                    "message": f"Worker unavailable: {str(e)}",
                    "type": "server_error"
                }
            }
            
    def deduct_credits(self, user_id: str, amount: float, reason: str, job_id: str = ""):
        """扣除 credits"""
        self.conn.execute(
            "INSERT INTO credits_log (job_id, user_id, amount, reason, timestamp) VALUES (?, ?, ?, ?, ?)",
            (job_id, user_id, -amount, reason, int(time.time()))
        )
        self.conn.commit()
            
    def get_balance(self, user_id: str = "anonymous") -> Dict[str, float]:
        """查询余额"""
        result = self.conn.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM credits_log WHERE user_id = ?",
            (user_id,)
        ).fetchone()[0]
        return {"user_id": user_id, "balance": float(result)}
        
    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        total_jobs = self.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        completed = self.conn.execute("SELECT COUNT(*) FROM jobs WHERE status='completed'").fetchone()[0]
        total_cost = self.conn.execute("SELECT COALESCE(SUM(credits_cost), 0) FROM jobs WHERE status='completed'").fetchone()[0]
        avg_latency = self.conn.execute("SELECT COALESCE(AVG(latency_ms), 0) FROM jobs WHERE status='completed'").fetchone()[0]
        
        return {
            "total_jobs": total_jobs,
            "completed_jobs": completed,
            "total_credits_earned": float(total_cost),
            "avg_latency_ms": float(avg_latency)
        }


# ============================================================
# CLI 入口（用于独立测试）
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="DecentralAI Router Bridge")
    parser.add_argument("--worker-url", default=PYTHON_WORKER_URL)
    args = parser.parse_args()
    
    router = InferenceRouter(args.worker_url)
    
    # 演示路由
    log.info("=" * 50)
    log.info("  DecentralAI Router Bridge")
    log.info(f"  Worker: {args.worker_url}")
    log.info("=" * 50)
    
    # 测试请求
    result = router.route(
        messages=[{"role": "user", "content": "def hello():"}],
        model="rwkv-4-169m",
        max_tokens=50,
        user_id="test-user"
    )
    
    print(f"\n路由结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
    print(f"余额: {router.get_balance('test-user')}")
    print(f"统计: {router.get_stats()}")


if __name__ == "__main__":
    main()
