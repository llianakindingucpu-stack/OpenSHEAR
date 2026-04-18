"""
DecentralAI Decentralized Node Launcher (v2)
===============================================
Uses config file, WebSocket transport, and API server.

Usage:
    python run.py                          # Use config.yaml
    python run.py --config prod.yaml       # Use custom config
    python run.py --level L2               # Override level
    python run.py --demo                   # Run demo mode
"""

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, r'D:\pylib')

from core import (
    Node, NodeIdentity, NodeLevel, NodeCapabilities,
    ExpertModel, ModelArchitecture,
)
from config_loader import ConfigLoader, DecentralAIConfig
from ws_transport import P2PNode


LEVEL_MAP = {
    'L0': NodeLevel.L0_COLLECTOR,
    'L1': NodeLevel.L1_LIGHT_INFERENCE,
    'L2': NodeLevel.L2_STANDARD_INFERENCE,
    'L3': NodeLevel.L3_HEAVY_INFERENCE,
    'L4': NodeLevel.L4_DATA_CENTER,
}

ARCH_MAP = {
    'rwkv': ModelArchitecture.RWKV,
    'transformer': ModelArchitecture.TRANSFORMER,
    'mamba': ModelArchitecture.MAMBA,
    'xlstm': ModelArchitecture.XLSTM,
}


def create_node_from_config(config: DecentralAIConfig) -> tuple:
    """Create a Node and P2PNode from configuration"""
    
    level = LEVEL_MAP.get(config.node.level, NodeLevel.L1_LIGHT_INFERENCE)
    
    # Determine architectures from experts
    architectures = set()
    for e in config.experts:
        arch = ARCH_MAP.get(e.architecture, ModelArchitecture.RWKV)
        architectures.add(arch)
    if not architectures:
        architectures.add(ModelArchitecture.RWKV)
    
    # Estimate hardware from level
    level_specs = {
        NodeLevel.L0_COLLECTOR: (2048, 10e9, 50, 50),
        NodeLevel.L1_LIGHT_INFERENCE: (8192, 50e9, 100, 100),
        NodeLevel.L2_STANDARD_INFERENCE: (8192, 12000e9, 1000, 500),
        NodeLevel.L3_HEAVY_INFERENCE: (24576, 82000e9, 1000, 2000),
        NodeLevel.L4_DATA_CENTER: (163840, 312000e9, 10000, 10000),
    }
    mem, flops, bw, storage = level_specs[level]
    
    caps = NodeCapabilities(
        level=level,
        architectures=list(architectures),
        max_memory_mb=config.limits.max_memory_mb or mem,
        compute_flops=flops,
        bandwidth_mbps=bw,
        storage_gb=storage,
    )
    
    identity = NodeIdentity(
        node_id=config.node.node_id or None,
        capabilities=caps,
    )
    
    node = Node(identity)
    
    # Add experts from config
    for e in config.experts:
        if e.model_path:
            arch = ARCH_MAP.get(e.architecture, ModelArchitecture.RWKV)
            param_mb = 0
            if os.path.exists(e.model_path):
                param_mb = os.path.getsize(e.model_path) // 1024 // 1024
            
            expert = ExpertModel(
                base_model=os.path.basename(e.model_path),
                architecture=arch,
                domain=e.domain,
                param_count_mb=param_mb,
            )
            node.add_expert(expert)
    
    # Create P2P node
    p2p = P2PNode(
        node_id=identity.node_id,
        listen_addr=config.network.listen,
    )
    
    return node, p2p


async def run_node(config: DecentralAIConfig):
    """Run the full node with P2P + API"""
    
    node, p2p = create_node_from_config(config)
    
    print("=" * 60)
    print("DecentralAI Node")
    print("=" * 60)
    print(f"  ID: {node.identity.node_id}")
    print(f"  Level: {config.node.level}")
    print(f"  Experts: {len(node.experts)}")
    for e in node.experts:
        print(f"    - {e.base_model} ({e.architecture.value})")
    print(f"  P2P: {config.network.listen}")
    print(f"  API: {config.network.api_listen}")
    
    # Connect to bootstrap peers
    for peer_addr in config.network.bootstrap_peers:
        print(f"  Connecting to bootstrap: {peer_addr}")
        await p2p.connect(peer_addr)
    
    # Start P2P server
    await p2p.start()


def main():
    parser = argparse.ArgumentParser(description="DecentralAI Node")
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--level', choices=['L0', 'L1', 'L2', 'L3', 'L4'],
                       help='Override node level')
    parser.add_argument('--demo', action='store_true', help='Run demo')
    args = parser.parse_args()
    
    if args.demo:
        from ws_transport import demo as ws_demo
        asyncio.run(ws_demo())
        return
    
    # Load config
    config = ConfigLoader.load(args.config)
    
    # Override level if specified
    if args.level:
        config.node.level = args.level
    
    # Validate
    warnings = ConfigLoader.validate(config)
    for w in warnings:
        print(f"WARNING: {w}")
    
    asyncio.run(run_node(config))


if __name__ == "__main__":
    main()
