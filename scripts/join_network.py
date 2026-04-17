#!/usr/bin/env python3
"""
DecentralAI - 节点一键加入脚本
自动检测硬件 → 生成配置 → 启动节点
"""

import os
import sys
import json
import socket
import platform
import subprocess
import secrets
import hashlib
from datetime import datetime

def get_node_id():
    """生成唯一节点ID"""
    hostname = socket.gethostname()
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    raw = f"{hostname}-{timestamp}-{secrets.token_hex(4)}"
    return f"node-{hashlib.md5(raw.encode()).hexdigest()[:12]}"

def detect_hardware():
    """检测硬件并推荐节点级别"""
    tier = "L1"  # 默认
    gpu_info = "Unknown"
    
    try:
        if platform.system() == "Windows":
            # 尝试用 wmic 检测
            result = subprocess.run(
                ["wmic", "computerSystem", "get", "TotalPhysicalMemory"],
                capture_output=True, text=True
            )
            memory_bytes = int(result.stdout.split("\n")[1].strip())
            memory_gb = memory_bytes / (1024**3)
            
            # 检测 NVIDIA GPU
            gpu_result = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                capture_output=True, text=True
            )
            gpu_info = gpu_result.stdout.split("\n")[1].strip() if len(gpu_result.stdout.split("\n")) > 1 else "No GPU"
            
            if "3060" in gpu_info or "3070" in gpu_info or "4060" in gpu_info:
                tier = "L2"
            elif "3090" in gpu_info or "4090" in gpu_info or "A100" in gpu_info:
                tier = "L3"
                
        elif platform.system() == "Linux":
            # Linux 检测
            with open("/proc/meminfo") as f:
                memory_gb = int(f.readline().split()[1]) / (1024**2)
            
            gpu_result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True
            )
            gpu_info = gpu_result.stdout.strip() if gpu_result.returncode == 0 else "No GPU"
            
            if "3060" in gpu_info:
                tier = "L2"
            elif "3090" in gpu_info or "4090" in gpu_info:
                tier = "L3"
    except Exception as e:
        print(f"[WARN] 硬件检测失败，使用默认值: {e}")
    
    return {
        "tier": tier,
        "gpu": gpu_info,
        "memory_gb": memory_gb if 'memory_gb' in dir() else "Unknown"
    }

def generate_config(node_id, tier, bootstrap_nodes):
    """生成配置文件"""
    config = {
        "node": {
            "id": node_id,
            "role": tier,
            "alias": socket.gethostname()
        },
        "network": {
            "api_port": 8080,
            "p2p_port": 9090,
            "bootstrap_nodes": bootstrap_nodes or [
                "bootstrap.decentralai.network:9090"
            ]
        },
        "blockchain": {
            "rpc_url": "",
            "wallet_private_key": "",
            "contract_address": ""
        },
        "inference": {
            "model_path": "",
            "max_batch_size": 4,
            "gpu_enabled": tier in ["L2", "L3", "L4"]
        }
    }
    return config

def main():
    print("=" * 50)
    print("  DecentralAI 节点加入向导")
    print("=" * 50)
    print()
    
    # Step 1: 检测硬件
    print("[1/4] 检测硬件配置...")
    hardware = detect_hardware()
    print(f"  GPU: {hardware['gpu']}")
    print(f"  推荐节点级别: {hardware['tier']}")
    print()
    
    # Step 2: 生成节点ID
    print("[2/4] 生成节点ID...")
    node_id = get_node_id()
    print(f"  节点ID: {node_id}")
    print()
    
    # Step 3: 配置引导节点
    print("[3/4] 配置网络...")
    bootstrap_input = input("  引导节点地址（直接回车使用默认）: ").strip()
    bootstrap_nodes = [bootstrap_input] if bootstrap_input else None
    print()
    
    # Step 4: 生成配置文件
    print("[4/4] 生成配置文件...")
    config = generate_config(node_id, hardware['tier'], bootstrap_nodes)
    
    config_path = "config.yaml"
    import yaml
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    
    print(f"  配置文件已生成: {config_path}")
    print()
    
    # 启动节点
    print("=" * 50)
    print("  硬件检测完成！")
    print(f"  节点ID: {node_id}")
    print(f"  节点级别: {hardware['tier']}")
    print()
    print("  启动命令:")
    print(f"    ./decentral-ai-core --config {config_path}")
    print()
    print("  API 端点:")
    print("    GET  http://localhost:8080/health")
    print("    POST http://localhost:8080/v1/chat/completions")
    print("=" * 50)

if __name__ == "__main__":
    main()
