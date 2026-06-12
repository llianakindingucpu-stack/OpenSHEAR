"""
DecentralAI 多节点路由器
=========================

核心能力：
1. 节点发现 + 健康检查
2. 智能负载均衡（基于能力/信誉/延迟）
3. 请求冗余 + 结果聚合
4. Credits 实时结算
5. 故障自动切换

架构：
  Client → Router (:8082) → Node A (:8081) → Model
                         ↘ Node B (:8083)
                         ↘ Node C (:8084)
"""

import os
import sys
import time
import json
import random
import logging
import sqlite3
import threading
import requests
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, Future

# ============================================================
# 配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("router")

DB_PATH = Path(__file__).parent.parent / "data" / "router.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ============================================================
# 数据结构
# ============================================================

class NodeStatus(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    UNHEALTHY = "unhealthy"


@dataclass
class NodeInfo:
    """节点信息"""
    node_id: str
    url: str
    level: int = 1  # L0-L4
    model: str = "unknown"
    status: NodeStatus = NodeStatus.OFFLINE
    capabilities: Dict = field(default_factory=dict)
    
    # 性能指标（指数移动平均）
    avg_latency_ms: float = 1000.0
    avg_tokens_per_sec: float = 0.0
    success_rate: float = 0.0
    
    # 统计
    total_requests: int = 0
    failed_requests: int = 0
    
    # 心跳
    last_heartbeat: float = 0
    registered_at: float = 0
    
    def score(self) -> float:
        """计算节点分数（用于负载均衡）"""
        if self.status != NodeStatus.ONLINE:
            return 0.0
        
        # 综合评分：成功率(40%) + 速度(30%) + 延迟(20%) + 级别(10%)
        success_score = self.success_rate
        speed_score = min(self.avg_tokens_per_sec / 20, 1.0)  # 20 tok/s = 满分
        latency_score = max(0, 1 - self.avg_latency_ms / 5000)  # 5s = 0分
        level_score = self.level / 4
        
        return success_score * 0.4 + speed_score * 0.3 + latency_score * 0.2 + level_score * 0.1
    
    def update_metrics(self, latency_ms: float, tokens: int, success: bool):
        """更新性能指标（EMA）"""
        alpha = 0.3  # 平滑系数
        
        self.total_requests += 1
        if not success:
            self.failed_requests += 1
        
        # 更新延迟
        self.avg_latency_ms = alpha * latency_ms + (1 - alpha) * self.avg_latency_ms
        
        # 更新速度
        if latency_ms > 0 and tokens > 0:
            tps = tokens / (latency_ms / 1000)
            self.avg_tokens_per_sec = alpha * tps + (1 - alpha) * self.avg_tokens_per_sec
        
        # 更新成功率
        self.success_rate = 1 - (self.failed_requests / self.total_requests)


@dataclass
class InferenceTask:
    """推理任务"""
    task_id: str
    prompt: str
    model: str
    max_tokens: int
    temperature: float
    redundancy: int = 1  # 冗余数
    timeout_ms: float = 30000
    
    user_id: str = "anonymous"
    created_at: float = 0
    
    # 结果
    responses: List[Dict] = field(default_factory=list)
    aggregated_result: Optional[str] = None
    final_latency_ms: float = 0


# ============================================================
# 节点注册表
# ============================================================

class NodeRegistry:
    """
    节点注册表 - 维护所有节点的状态
    """
    
    def __init__(self):
        self.nodes: Dict[str, NodeInfo] = {}
        self._lock = threading.RLock()
        self._health_thread: Optional[threading.Thread] = None
        self._running = False
        
    def register(self, node_id: str, url: str, level: int = 1, model: str = "unknown") -> NodeInfo:
        """注册节点"""
        with self._lock:
            node = NodeInfo(
                node_id=node_id,
                url=url,
                level=level,
                model=model,
                status=NodeStatus.OFFLINE,
                registered_at=time.time()
            )
            self.nodes[node_id] = node
            log.info(f"节点注册: {node_id} @ {url} (L{level})")
            return node
    
    def unregister(self, node_id: str):
        """注销节点"""
        with self._lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                log.info(f"节点注销: {node_id}")
    
    def get(self, node_id: str) -> Optional[NodeInfo]:
        """获取节点"""
        with self._lock:
            return self.nodes.get(node_id)
    
    def get_all(self) -> List[NodeInfo]:
        """获取所有节点"""
        with self._lock:
            return list(self.nodes.values())
    
    def get_healthy(self) -> List[NodeInfo]:
        """获取健康节点"""
        with self._lock:
            return [n for n in self.nodes.values() if n.status == NodeStatus.ONLINE]
    
    def get_best_for_model(self, model: str, count: int = 1) -> List[NodeInfo]:
        """
        获取指定模型的最佳节点
        按分数排序，返回前 count 个
        """
        with self._lock:
            candidates = [
                n for n in self.nodes.values()
                if n.status == NodeStatus.ONLINE and (n.model == model or model == "any")
            ]
            # 按分数排序
            candidates.sort(key=lambda n: n.score(), reverse=True)
            return candidates[:count]
    
    def update_heartbeat(self, node_id: str):
        """更新心跳"""
        with self._lock:
            if node_id in self.nodes:
                self.nodes[node_id].last_heartbeat = time.time()
                if self.nodes[node_id].status == NodeStatus.UNHEALTHY:
                    self.nodes[node_id].status = NodeStatus.ONLINE
                    log.info(f"节点恢复: {node_id}")
    
    def start_health_check(self, interval_sec: float = 10.0):
        """启动健康检查线程"""
        self._running = True
        self._health_thread = threading.Thread(
            target=self._health_check_loop,
            args=(interval_sec,),
            daemon=True
        )
        self._health_thread.start()
    
    def stop_health_check(self):
        """停止健康检查"""
        self._running = False
        if self._health_thread:
            self._health_thread.join(timeout=5)
    
    def _health_check_loop(self, interval_sec: float):
        """健康检查循环"""
        while self._running:
            self._check_all_nodes()
            time.sleep(interval_sec)
    
    def _check_all_nodes(self):
        """检查所有节点健康状态"""
        now = time.time()
        timeout = 30  # 30秒无心跳视为不健康
        
        for node in self.get_all():
            # 检查心跳超时
            if now - node.last_heartbeat > timeout:
                if node.status != NodeStatus.OFFLINE:
                    node.status = NodeStatus.UNHEALTHY
                    log.warning(f"节点超时: {node.node_id}")
            else:
                # 主动探测
                try:
                    resp = requests.get(
                        f"{node.url}/health",
                        timeout=5
                    )
                    if resp.status_code == 200:
                        node.status = NodeStatus.ONLINE
                        data = resp.json()
                        node.model = data.get("model", node.model)
                    else:
                        node.status = NodeStatus.UNHEALTHY
                except Exception:
                    if node.status == NodeStatus.ONLINE:
                        node.status = NodeStatus.UNHEALTHY
                        log.warning(f"节点无响应: {node.node_id}")


# ============================================================
# 路由器核心
# ============================================================

class InferenceRouter:
    """
    推理路由器 - 网络的心脏
    
    职责：
    1. 接收客户端请求
    2. 选择最佳节点执行
    3. 处理冗余请求（多节点并行）
    4. 聚合结果
    5. 结算 Credits
    """
    
    # Credits 费率（per 1K tokens）
    RATE_TABLE = {
        "rwkv-4-169m": 0.5,
        "rwkv-4-430m": 0.8,
        "qwen-0.5b": 1.0,
        "qwen-1.5b": 1.5,
        "qwen-7b": 2.5,
        "default": 1.0,
    }
    
    def __init__(self):
        self.registry = NodeRegistry()
        self.executor = ThreadPoolExecutor(max_workers=20)
        self.pending_tasks: Dict[str, InferenceTask] = {}
        self._lock = threading.RLock()
        
        # 初始化数据库
        self._init_db()
        
        # 启动健康检查
        self.registry.start_health_check(interval_sec=15)
    
    def _init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                user_id TEXT,
                model TEXT,
                prompt TEXT,
                max_tokens INTEGER,
                redundancy INTEGER,
                status TEXT,
                result TEXT,
                latency_ms REAL,
                credits_cost REAL,
                created_at INTEGER,
                completed_at INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS credits (
                user_id TEXT PRIMARY KEY,
                balance REAL DEFAULT 100.0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS credits_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                amount REAL,
                reason TEXT,
                task_id TEXT,
                timestamp INTEGER
            )
        """)
        conn.commit()
        conn.close()
    
    def register_node(self, node_id: str, url: str, level: int = 1, model: str = "unknown"):
        """注册新节点"""
        return self.registry.register(node_id, url, level, model)
    
    def route_and_execute(self, task: InferenceTask) -> Dict:
        """
        路由并执行任务
        
        流程：
        1. 选择最佳节点（redundancy 个）
        2. 并行发送请求
        3. 等待结果（最多 timeout_ms）
        4. 聚合结果
        5. 结算 Credits
        """
        task.created_at = time.time()
        start_time = time.perf_counter()
        
        # 1. 选择节点
        nodes = self.registry.get_best_for_model(task.model, task.redundancy)
        
        if not nodes:
            return {"error": {"message": "No available nodes", "type": "router_error"}}
        
        log.info(f"[{task.task_id}] 路由到 {len(nodes)} 个节点: {[n.node_id for n in nodes]}")
        
        # 2. 预扣 Credits
        estimated_cost = self._estimate_cost(task.model, task.max_tokens) * task.redundancy
        self._deduct_credits(task.user_id, estimated_cost, "预授权", task.task_id)
        
        # 3. 并行请求
        futures: Dict[str, Future] = {}
        for node in nodes:
            future = self.executor.submit(
                self._execute_on_node,
                task,
                node
            )
            futures[node.node_id] = future
        
        # 4. 等待结果
        results = []
        for node_id, future in futures.items():
            try:
                result = future.result(timeout=task.timeout_ms / 1000)
                results.append(result)
                self.registry.update_heartbeat(node_id)
            except Exception as e:
                log.warning(f"[{task.task_id}] 节点 {node_id} 执行失败: {e}")
        
        # 5. 聚合结果
        if results:
            aggregated = self._aggregate_results(results)
            task.aggregated_result = aggregated["text"]
            task.final_latency_ms = (time.perf_counter() - start_time) * 1000
            
            # 计算实际费用并退款
            actual_tokens = aggregated.get("tokens", 0)
            actual_cost = self._estimate_cost(task.model, actual_tokens)
            refund = estimated_cost - actual_cost
            if refund > 0:
                self._refund_credits(task.user_id, refund, "退款", task.task_id)
            
            # 记录任务
            self._record_task(task, actual_cost, "completed")
            
            return {
                "task_id": task.task_id,
                "model": task.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": aggregated["text"]},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(task.prompt) // 4,
                    "completion_tokens": actual_tokens,
                    "total_tokens": len(task.prompt) // 4 + actual_tokens
                },
                "latency_ms": task.final_latency_ms,
                "nodes_used": len(results),
                "credits_cost": actual_cost
            }
        else:
            # 全部失败，退款
            self._refund_credits(task.user_id, estimated_cost, "执行失败退款", task.task_id)
            self._record_task(task, 0, "failed")
            return {"error": {"message": "All nodes failed", "type": "execution_error"}}
    
    def _execute_on_node(self, task: InferenceTask, node: NodeInfo) -> Dict:
        """在单个节点上执行任务"""
        start = time.perf_counter()
        
        try:
            resp = requests.post(
                f"{node.url}/v1/chat/completions",
                json={
                    "model": task.model,
                    "messages": [{"role": "user", "content": task.prompt}],
                    "max_tokens": task.max_tokens,
                    "temperature": task.temperature
                },
                timeout=task.timeout_ms / 1000
            )
            resp.raise_for_status()
            data = resp.json()
            
            latency_ms = (time.perf_counter() - start) * 1000
            tokens = data.get("usage", {}).get("completion_tokens", 0)
            
            # 更新节点指标
            node.update_metrics(latency_ms, tokens, True)
            
            return {
                "node_id": node.node_id,
                "text": data["choices"][0]["message"]["content"],
                "tokens": tokens,
                "latency_ms": latency_ms
            }
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            node.update_metrics(latency_ms, 0, False)
            raise
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """
        聚合多个节点的结果
        
        策略：
        1. 如果结果一致（高相似度），任选一个
        2. 如果不一致，选择最快节点的结果（可扩展为投票）
        """
        if len(results) == 1:
            return results[0]
        
        # 按延迟排序
        results.sort(key=lambda r: r["latency_ms"])
        
        # 计算文本相似度
        texts = [r["text"] for r in results]
        if self._all_similar(texts, threshold=0.7):
            # 结果一致，选最快的
            return results[0]
        else:
            # 结果不一致，暂时选最快的（未来可改为投票）
            log.warning(f"结果不一致: {[t[:50] for t in texts]}")
            return results[0]
    
    def _all_similar(self, texts: List[str], threshold: float = 0.7) -> bool:
        """检查所有文本是否相似"""
        if len(texts) < 2:
            return True
        
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if self._similarity(texts[i], texts[j]) < threshold:
                    return False
        return True
    
    def _similarity(self, a: str, b: str) -> float:
        """Jaccard 相似度"""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)
    
    def _estimate_cost(self, model: str, tokens: int) -> float:
        """估算费用"""
        rate = self.RATE_TABLE.get(model, self.RATE_TABLE["default"])
        return rate * tokens / 1000
    
    def _deduct_credits(self, user_id: str, amount: float, reason: str, task_id: str):
        """扣除 Credits"""
        conn = sqlite3.connect(str(DB_PATH))
        # 确保用户存在
        conn.execute("INSERT OR IGNORE INTO credits (user_id) VALUES (?)", (user_id,))
        # 扣除
        conn.execute("UPDATE credits SET balance = balance - ? WHERE user_id = ?", (amount, user_id))
        # 记录日志
        conn.execute(
            "INSERT INTO credits_log (user_id, amount, reason, task_id, timestamp) VALUES (?, ?, ?, ?, ?)",
            (user_id, -amount, reason, task_id, int(time.time()))
        )
        conn.commit()
        conn.close()
    
    def _refund_credits(self, user_id: str, amount: float, reason: str, task_id: str):
        """退还 Credits"""
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("UPDATE credits SET balance = balance + ? WHERE user_id = ?", (amount, user_id))
        conn.execute(
            "INSERT INTO credits_log (user_id, amount, reason, task_id, timestamp) VALUES (?, ?, ?, ?, ?)",
            (user_id, amount, reason, task_id, int(time.time()))
        )
        conn.commit()
        conn.close()
    
    def _record_task(self, task: InferenceTask, cost: float, status: str):
        """记录任务"""
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("""
            INSERT INTO tasks (task_id, user_id, model, prompt, max_tokens, redundancy, 
                              status, result, latency_ms, credits_cost, created_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task.task_id, task.user_id, task.model, task.prompt[:500], 
            task.max_tokens, task.redundancy, status,
            task.aggregated_result[:500] if task.aggregated_result else None,
            task.final_latency_ms, cost, int(task.created_at), int(time.time())
        ))
        conn.commit()
        conn.close()
    
    def get_balance(self, user_id: str) -> float:
        """查询余额"""
        conn = sqlite3.connect(str(DB_PATH))
        result = conn.execute(
            "SELECT balance FROM credits WHERE user_id = ?", (user_id,)
        ).fetchone()
        conn.close()
        return result[0] if result else 100.0
    
    def get_stats(self) -> Dict:
        """获取统计"""
        conn = sqlite3.connect(str(DB_PATH))
        total_tasks = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        completed = conn.execute("SELECT COUNT(*) FROM tasks WHERE status='completed'").fetchone()[0]
        total_credits = conn.execute(
            "SELECT COALESCE(SUM(credits_cost), 0) FROM tasks WHERE status='completed'"
        ).fetchone()[0]
        avg_latency = conn.execute(
            "SELECT COALESCE(AVG(latency_ms), 0) FROM tasks WHERE status='completed'"
        ).fetchone()[0]
        conn.close()
        
        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed,
            "total_credits_earned": float(total_credits),
            "avg_latency_ms": float(avg_latency),
            "active_nodes": len(self.registry.get_healthy()),
            "total_nodes": len(self.registry.get_all())
        }
    
    def shutdown(self):
        """关闭路由器"""
        self.registry.stop_health_check()
        self.executor.shutdown(wait=True)


# ============================================================
# HTTP 服务
# ============================================================

def create_http_server(router: InferenceRouter, host: str = "0.0.0.0", port: int = 8082):
    """创建 HTTP 服务"""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from urllib.parse import urlparse, parse_qs
    import uuid
    
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # 静默
        
        def do_GET(self):
            path = urlparse(self.path).path
            
            if path == "/health":
                self.send_json({"status": "ok", "role": "router", "nodes": len(router.registry.get_healthy())})
            
            elif path == "/stats":
                self.send_json(router.get_stats())
            
            elif path == "/nodes":
                nodes = [
                    {
                        "node_id": n.node_id,
                        "url": n.url,
                        "level": n.level,
                        "model": n.model,
                        "status": n.status.value,
                        "score": round(n.score(), 3),
                        "avg_latency_ms": round(n.avg_latency_ms, 1),
                        "success_rate": round(n.success_rate, 3)
                    }
                    for n in router.registry.get_all()
                ]
                self.send_json({"nodes": nodes, "count": len(nodes)})
            
            elif path == "/balance":
                query = parse_qs(urlparse(self.path).query)
                user_id = query.get("user_id", ["anonymous"])[0]
                self.send_json({"user_id": user_id, "balance": router.get_balance(user_id)})
            
            else:
                self.send_error(404)
        
        def do_POST(self):
            path = urlparse(self.path).path
            
            if path == "/register":
                data = self.read_json()
                router.register_node(
                    data.get("node_id", str(uuid.uuid4())[:8]),
                    data.get("url", "http://127.0.0.1:8081"),
                    data.get("level", 1),
                    data.get("model", "unknown")
                )
                self.send_json({"status": "registered"})
            
            elif path == "/v1/chat/completions":
                data = self.read_json()
                
                # 提取消息
                messages = data.get("messages", [])
                prompt = "\n".join(
                    f"{m.get('role', 'user')}: {m.get('content', '')}"
                    for m in messages
                ) if messages else data.get("prompt", "")
                
                task = InferenceTask(
                    task_id=f"task-{int(time.time()*1000)}-{random.randint(1000,9999)}",
                    prompt=prompt,
                    model=data.get("model", "rwkv-4-169m"),
                    max_tokens=min(data.get("max_tokens", 128), 512),
                    temperature=data.get("temperature", 0.7),
                    redundancy=data.get("redundancy", 1),
                    timeout_ms=data.get("timeout_ms", 30000),
                    user_id=data.get("user_id", "anonymous")
                )
                
                result = router.route_and_execute(task)
                self.send_json(result)
            
            else:
                self.send_error(404)
        
        def read_json(self):
            length = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(length))
        
        def send_json(self, obj, code=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
    
    server = HTTPServer((host, port), Handler)
    log.info(f"Router HTTP 服务启动: {host}:{port}")
    return server


# ============================================================
# CLI
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="DecentralAI Router")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--register", action="append", nargs=4,
                        metavar=("NODE_ID", "URL", "LEVEL", "MODEL"),
                        help="预注册节点")
    args = parser.parse_args()
    
    router = InferenceRouter()
    
    # 预注册节点
    if args.register:
        for node_id, url, level, model in args.register:
            router.register_node(node_id, url, int(level), model)
    
    server = create_http_server(router, args.host, args.port)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("关闭路由器...")
        router.shutdown()
        server.shutdown()


if __name__ == "__main__":
    main()
