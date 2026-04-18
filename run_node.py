#!/usr/bin/env python3
"""
DecentralAI Node Launcher
==========================
Start a DecentralAI node with the specified level.

Usage:
    python run_node.py --level L1
    python run_node.py --level L2 --model rwkv-4-world-430m
    python run_node.py --level L0 --bootstrap 192.168.1.100:8080
"""

import argparse
import json
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, r'D:\pylib')

from core import (
    Node, NodeIdentity, NodeLevel, NodeCapabilities,
    ExpertModel, ModelArchitecture, RequestType,
)
from evolution import EvolutionCycle
from network import NetworkNode, PeerDiscovery, Message, MessageType


MODEL_MAP = {
    NodeLevel.L0_COLLECTOR: None,  # No model needed
    NodeLevel.L1_LIGHT_INFERENCE: "rwkv-4-169m-pile",
    NodeLevel.L2_STANDARD_INFERENCE: "rwkv-4-world-430m",
    NodeLevel.L3_HEAVY_INFERENCE: "rwkv-5-world-3b",
    NodeLevel.L4_DATA_CENTER: "rwkv-6-world-7b",
}

ARCH_MAP = {
    NodeLevel.L0_COLLECTOR: [],
    NodeLevel.L1_LIGHT_INFERENCE: [ModelArchitecture.RWKV],
    NodeLevel.L2_STANDARD_INFERENCE: [ModelArchitecture.RWKV, ModelArchitecture.TRANSFORMER],
    NodeLevel.L3_HEAVY_INFERENCE: [ModelArchitecture.TRANSFORMER, ModelArchitecture.RWKV],
    NodeLevel.L4_DATA_CENTER: [ModelArchitecture.TRANSFORMER],
}


def create_node(level: NodeLevel, model_path: str = None) -> NetworkNode:
    """Create a node with the given level"""
    
    # Determine capabilities based on level
    level_specs = {
        NodeLevel.L0_COLLECTOR: {
            'max_memory_mb': 2048,
            'compute_flops': 10e9,
            'bandwidth_mbps': 50,
            'storage_gb': 50,
        },
        NodeLevel.L1_LIGHT_INFERENCE: {
            'max_memory_mb': 8192,
            'compute_flops': 50e9,
            'bandwidth_mbps': 100,
            'storage_gb': 100,
        },
        NodeLevel.L2_STANDARD_INFERENCE: {
            'max_memory_mb': 8192,
            'compute_flops': 12000e9,
            'bandwidth_mbps': 1000,
            'storage_gb': 500,
        },
        NodeLevel.L3_HEAVY_INFERENCE: {
            'max_memory_mb': 24576,
            'compute_flops': 82000e9,
            'bandwidth_mbps': 1000,
            'storage_gb': 2000,
        },
        NodeLevel.L4_DATA_CENTER: {
            'max_memory_mb': 163840,
            'compute_flops': 312000e9,
            'bandwidth_mbps': 10000,
            'storage_gb': 10000,
        },
    }
    
    specs = level_specs[level]
    caps = NodeCapabilities(
        level=level,
        architectures=ARCH_MAP[level],
        **specs
    )
    
    identity = NodeIdentity(capabilities=caps)
    node = NetworkNode(identity)
    
    # Add expert model if applicable
    if level != NodeLevel.L0_COLLECTOR and model_path:
        model_name = os.path.basename(model_path)
        arch = ModelArchitecture.RWKV if 'rwkv' in model_name.lower() else ModelArchitecture.TRANSFORMER
        
        expert = ExpertModel(
            base_model=model_name,
            architecture=arch,
            domain="general",
            param_count_mb=os.path.getsize(model_path) // 1024 // 1024 if os.path.exists(model_path) else 0,
        )
        node.add_expert(expert)
    
    return node


def run_interactive(node: NetworkNode):
    """Run node in interactive mode"""
    print(f"\nDecentralAI Node v0.1.0")
    print(f"Level: {node.identity.capabilities.level.name}")
    print(f"ID: {node.identity.node_id}")
    print(f"Experts: {len(node.experts)}")
    print(f"Type 'status' for info, 'quit' to exit\n")
    
    cycle = EvolutionCycle(node) if node.experts else None
    
    while True:
        try:
            cmd = input("decentral-ai> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if cmd in ('quit', 'exit', 'q'):
            break
        elif cmd == 'status':
            print(json.dumps(node.get_network_status(), indent=2))
            if cycle:
                print(f"Evolution: {json.dumps(cycle.get_status(), indent=2)}")
        elif cmd == 'announce':
            msg = node.announce_self()
            print(f"Announced: {msg.msg_type.value}")
            print(json.dumps(msg.payload, indent=2))
        elif cmd == 'evolve':
            if cycle:
                plan = cycle.run_cycle()
                if plan:
                    print(f"Evolution plan: {plan.action.value}")
                    accepted, imp = cycle.execute_and_verify(plan)
                    print(f"Result: {'accepted' if accepted else 'rejected'}")
                else:
                    print("No evolution needed yet")
        elif cmd.startswith('ask '):
            prompt = cmd[4:]
            request = InferenceRequest(
                request_type=RequestType.GENERAL_CHAT,
                prompt=prompt,
            )
            routes = node.router.route(request)
            print(f"Routed to {len(routes)} expert(s)")
            for eid, score in routes:
                e = node.router.experts[eid]
                print(f"  {e.base_model} (score={score:.3f})")
        elif cmd == 'help':
            print("Commands: status, announce, evolve, ask <prompt>, quit")
        else:
            print(f"Unknown command: {cmd}")
    
    print("Node shutting down...")


def main():
    parser = argparse.ArgumentParser(description="DecentralAI Node")
    parser.add_argument('--level', choices=['L0', 'L1', 'L2', 'L3', 'L4'],
                       default='L1', help='Node level')
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--bootstrap', type=str, help='Bootstrap node address')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    level_map = {'L0': NodeLevel.L0_COLLECTOR, 'L1': NodeLevel.L1_LIGHT_INFERENCE,
                 'L2': NodeLevel.L2_STANDARD_INFERENCE, 'L3': NodeLevel.L3_HEAVY_INFERENCE,
                 'L4': NodeLevel.L4_DATA_CENTER}
    level = level_map[args.level]
    model_path = args.model or MODEL_MAP.get(level)
    
    node = create_node(level, model_path)
    
    if args.bootstrap:
        node.discovery.add_bootstrap(args.bootstrap)
    
    if args.interactive or True:  # Default to interactive
        run_interactive(node)


if __name__ == "__main__":
    main()
