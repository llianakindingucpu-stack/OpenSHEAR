"""
DecentralAI - 统一启动入口
一键启动：Rust API + Python Worker + Router Bridge

使用方式：
    python run.py                    # 启动完整栈
    python run.py --rust-only        # 只启动 Rust
    python run.py --python-only      # 只启动 Python Worker
    python run.py --router-only       # 只启动 Router（测试模式）
"""

import os
import sys
import time
import signal
import argparse
import logging
import subprocess
import threading
import socket
from pathlib import Path
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("run")

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
RUST_DIR = PROJECT_ROOT / "src-rs" / "decentral-ai-core"
PYTHON_SCRIPTS = PROJECT_ROOT / "scripts"


def is_port_available(port: int) -> bool:
    """检查端口是否可用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def is_port_in_use(port: int) -> bool:
    """检查端口是否已被占用"""
    return not is_port_available(port)


class ProcessManager:
    """进程管理器"""
    
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        
    def add(self, name: str, proc: subprocess.Popen):
        log.info(f"启动: {name} (PID: {proc.pid})")
        self.processes.append(proc)
        
    def stop_all(self):
        log.info("停止所有服务...")
        for proc in self.processes:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        log.info("所有服务已停止")


def start_rust(manager: ProcessManager, mode: str = "release"):
    """启动 Rust API 服务"""
    exe_path = RUST_DIR / "target" / mode / "decentral-ai-core.exe"
    
    if not exe_path.exists():
        log.warning(f"Rust 二进制不存在: {exe_path}，跳过")
        log.info("运行 'cargo build' 编译")
        return
    
    log.info(f"启动 Rust API 服务: {exe_path}")
    env = os.environ.copy()
    proc = subprocess.Popen(
        [str(exe_path)],
        cwd=str(RUST_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    manager.add("Rust API (:8080)", proc)


def start_python_worker(manager: ProcessManager, port: int = 8081, mock: bool = True):
    """启动 Python 推理 Worker"""
    log.info(f"启动 Python Worker (端口 {port}, mock={mock})")
    args = [sys.executable, "scripts/inference_worker.py", "--port", str(port)]
    if mock:
        args.append("--mock")
    
    proc = subprocess.Popen(
        args,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    manager.add(f"Python Worker (:{port})", proc)


def start_router(manager: ProcessManager):
    """启动 Router Bridge"""
    log.info("启动 Router Bridge")
    # Router 是同步的，只用于测试，不后台运行
    proc = subprocess.Popen(
        [sys.executable, "scripts/router_bridge.py"],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    # Router 同步执行一个测试后退出
    stdout, stderr = proc.communicate(timeout=30)
    if proc.returncode == 0:
        log.info("Router 测试通过")
        log.info(stdout.decode("utf-8", errors="replace"))
    else:
        log.error(f"Router 测试失败: {stderr.decode('utf-8', errors='replace')}")


def status_check():
    """状态检查"""
    log.info("=" * 50)
    log.info("  DecentralAI 服务状态")
    log.info("=" * 50)
    
    ports = {
        8080: "Rust API",
        8081: "Python Worker"
    }

    for port, name in ports.items():
        if is_port_in_use(port):
            log.info(f"  {name:20s} ✅ 运行中 (端口 {port})")
        else:
            log.info(f"  {name:20s} ❌ 未运行 (端口 {port} 空闲)")
    
    log.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="DecentralAI 统一启动器")
    parser.add_argument("--rust-only", action="store_true", help="只启动 Rust API")
    parser.add_argument("--python-only", action="store_true", help="只启动 Python Worker")
    parser.add_argument("--router-only", action="store_true", help="只测试 Router Bridge")
    parser.add_argument("--status", action="store_true", help="查看服务状态")
    parser.add_argument("--mock", action="store_true", default=True, help="Python Worker 使用 Mock 模式")
    args = parser.parse_args()
    
    if args.status:
        status_check()
        return
    
    if args.router_only:
        start_router(ProcessManager())
        return
    
    manager = ProcessManager()
    
    try:
        if args.rust_only:
            start_rust(manager)
        elif args.python_only:
            start_python_worker(manager, mock=args.mock)
        else:
            # 启动完整栈
            start_python_worker(manager, port=8081, mock=args.mock)
            
            # Rust 可能已在运行，跳过
            if is_port_in_use(8080):
                log.info("Rust API 已在运行，跳过")
            else:
                start_rust(manager)
            
            log.info("")
            log.info("=" * 50)
            log.info("  DecentralAI 完整栈已启动")
            log.info("  Rust API:      http://127.0.0.1:8080")
            log.info("  Python Worker: http://127.0.0.1:8081")
            log.info("")
            log.info("  测试命令:")
            log.info("    curl http://127.0.0.1:8080/health")
            log.info("    curl http://127.0.0.1:8081/health")
            log.info("=" * 50)
        
        # 保持运行
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        log.info("\n收到中断信号")
    finally:
        manager.stop_all()


if __name__ == "__main__":
    main()
