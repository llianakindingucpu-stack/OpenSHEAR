"""
DecentralAI Network Layer
==========================
Peer-to-peer communication between nodes.

No central server. No single point of failure.
Nodes discover each other, exchange requests, and verify results.

Protocol:
1. Discovery: Find peers via DHT or bootstrap nodes
2. Gossip: Share node status and expert info
3. Request: Forward inference requests to best nodes
4. Verify: Cross-validate results between nodes
"""

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from core import (
    Node, NodeIdentity, NodeLevel, NodeCapabilities,
    ExpertModel, ModelArchitecture,
    InferenceRequest, InferenceResponse, RequestType,
    Router, Verifier, CreditLedger
)


# ============================================================
# 1. Message Types
# ============================================================

class MessageType(Enum):
    """P2P message types"""
    PING = "ping"
    PONG = "pong"
    NODE_ANNOUNCE = "node_announce"     # I exist, here's what I can do
    EXPERT_ANNOUNCE = "expert_announce" # I have this expert
    REQUEST_FORWARD = "request_forward" # Please handle this request
    RESPONSE_RETURN = "response_return" # Here's the response
    VERIFICATION = "verification"       # I verified this result
    GOSSIP = "gossip"                   # Status update
    CREDIT_TRANSFER = "credit_transfer" # Payment
    EVOLUTION_SYNC = "evolution_sync"   # Share evolution progress


@dataclass
class Message:
    """A P2P message"""
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    msg_type: MessageType = MessageType.PING
    sender_id: str = ""
    recipient_id: str = ""  # Empty = broadcast
    payload: Dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    signature: str = ""     # Cryptographic signature
    
    def to_json(self) -> str:
        return json.dumps({
            'msg_id': self.msg_id,
            'msg_type': self.msg_type.value,
            'sender_id': self.sender_id,
            'recipient_id': self.recipient_id,
            'payload': self.payload,
            'timestamp': self.timestamp,
        })
    
    @classmethod
    def from_json(cls, data: str) -> 'Message':
        d = json.loads(data)
        return cls(
            msg_id=d.get('msg_id', ''),
            msg_type=MessageType(d.get('msg_type', 'ping')),
            sender_id=d.get('sender_id', ''),
            recipient_id=d.get('recipient_id', ''),
            payload=d.get('payload', {}),
            timestamp=d.get('timestamp', 0),
        )


# ============================================================
# 2. Peer Discovery
# ============================================================

class PeerDiscovery:
    """
    Find other nodes in the network.
    
    Strategy:
    1. Bootstrap nodes (well-known entry points)
    2. Gossip protocol (nodes tell each other about other nodes)
    3. DHT (distributed hash table for expert lookup)
    
    Like how people find each other:
    - You know someone, who knows someone, who knows someone...
    """
    
    def __init__(self, local_node_id: str):
        self.local_id = local_node_id
        self.known_peers: Dict[str, Dict] = {}  # node_id -> info
        self.bootstrap_nodes: List[str] = []
        self.last_gossip: float = 0
        self.gossip_interval: float = 60  # seconds
    
    def add_bootstrap(self, address: str):
        """Add a bootstrap node address"""
        self.bootstrap_nodes.append(address)
    
    def learn_about_peer(self, node_id: str, info: Dict):
        """Learn about a peer node"""
        if node_id == self.local_id:
            return  # Don't learn about self
        self.known_peers[node_id] = {
            **info,
            'last_seen': time.time()
        }
    
    def forget_peer(self, node_id: str, timeout: float = 300):
        """Remove peers we haven't heard from"""
        if node_id in self.known_peers:
            if time.time() - self.known_peers[node_id].get('last_seen', 0) > timeout:
                del self.known_peers[node_id]
    
    def find_experts(self, domain: str = "", architecture: ModelArchitecture = None) -> List[Dict]:
        """Find peers that have matching experts"""
        results = []
        for node_id, info in self.known_peers.items():
            for expert in info.get('experts', []):
                if domain and expert.get('domain', '') != domain and expert.get('domain', '') != 'general':
                    continue
                if architecture and expert.get('architecture', '') != architecture.value:
                    continue
                results.append({
                    'node_id': node_id,
                    'expert': expert,
                    'latency_ms': info.get('latency_ms', 9999)
                })
        return sorted(results, key=lambda x: x['latency_ms'])
    
    def get_status(self) -> Dict:
        return {
            'known_peers': len(self.known_peers),
            'bootstrap_nodes': len(self.bootstrap_nodes),
        }


# ============================================================
# 3. Network Node (extends core Node with networking)
# ============================================================

class NetworkNode(Node):
    """
    A DecentralAI node with P2P networking capabilities.
    
    This is a complete node that can:
    - Discover peers
    - Route requests across the network
    - Verify results from other nodes
    - Share evolution progress
    - Earn and spend credits
    """
    
    def __init__(self, identity: NodeIdentity):
        super().__init__(identity)
        self.discovery = PeerDiscovery(identity.node_id)
        self.message_queue: List[Message] = []
        self.sent_messages: int = 0
        self.received_messages: int = 0
    
    def announce_self(self) -> Message:
        """Create a node announcement message"""
        expert_info = [{
            'expert_id': e.expert_id,
            'base_model': e.base_model,
            'architecture': e.architecture.value,
            'domain': e.domain,
            'param_count_mb': e.param_count_mb,
            'avg_tokens_per_sec': e.avg_tokens_per_sec,
            'success_rate': e.success_rate,
        } for e in self.experts]
        
        return Message(
            msg_type=MessageType.NODE_ANNOUNCE,
            sender_id=self.identity.node_id,
            payload={
                'node_id': self.identity.node_id,
                'level': self.identity.capabilities.level.name,
                'experts': expert_info,
                'uptime': time.time() - self.identity.registered_at,
            }
        )
    
    def handle_message(self, msg: Message) -> Optional[Message]:
        """Process an incoming message"""
        self.received_messages += 1
        
        if msg.msg_type == MessageType.PING:
            return Message(
                msg_type=MessageType.PONG,
                sender_id=self.identity.node_id,
                recipient_id=msg.sender_id,
                payload={'status': 'ok'}
            )
        
        elif msg.msg_type == MessageType.NODE_ANNOUNCE:
            # Learn about this peer
            info = msg.payload
            self.discovery.learn_about_peer(info['node_id'], info)
            
            # Register their experts in our router
            for expert_data in info.get('experts', []):
                # Create a remote expert entry
                remote_expert = ExpertModel(
                    expert_id=expert_data['expert_id'],
                    base_model=expert_data['base_model'],
                    architecture=ModelArchitecture(expert_data.get('architecture', 'rwkv')),
                    domain=expert_data.get('domain', 'general'),
                    param_count_mb=expert_data.get('param_count_mb', 0),
                    avg_tokens_per_sec=expert_data.get('avg_tokens_per_sec', 0),
                    success_rate=expert_data.get('success_rate', 0),
                )
                self.router.register_expert(info['node_id'], remote_expert)
            
            return None  # No response needed for announcements
        
        elif msg.msg_type == MessageType.REQUEST_FORWARD:
            # Another node is asking us to handle a request
            request_data = msg.payload
            request = InferenceRequest(
                request_type=RequestType(request_data.get('request_type', 'chat')),
                prompt=request_data.get('prompt', ''),
                max_tokens=request_data.get('max_tokens', 512),
                complexity=request_data.get('complexity', 1),
                requester_id=msg.sender_id,
            )
            
            response = self.handle_request(request)
            if response:
                return Message(
                    msg_type=MessageType.RESPONSE_RETURN,
                    sender_id=self.identity.node_id,
                    recipient_id=msg.sender_id,
                    payload={
                        'response_id': response.response_id,
                        'request_id': request.request_id,
                        'text': response.text,
                        'tokens_generated': response.tokens_generated,
                        'latency_ms': response.latency_ms,
                        'expert_id': response.expert_id,
                    }
                )
            return None
        
        elif msg.msg_type == MessageType.RESPONSE_RETURN:
            # A node returned a response to our request
            # Store for verification
            return None
        
        elif msg.msg_type ==MessageType.VERIFICATION:
            # Another node verified a result
            return None
        
        return None
    
    def forward_request(self, request: InferenceRequest) -> List[Message]:
        """
        Forward a request to appropriate remote nodes.
        Called when local experts can't handle the request.
        """
        # Find remote experts
        domain = request.request_type.value
        remote_experts = self.discovery.find_experts(domain=domain)
        
        # Select best candidates
        candidates = self.router.route(request)
        
        # If no local experts, use remote
        messages = []
        if not candidates:
            for re in remote_experts[:request.redundancy_count]:
                msg = Message(
                    msg_type=MessageType.REQUEST_FORWARD,
                    sender_id=self.identity.node_id,
                    recipient_id=re['node_id'],
                    payload={
                        'request_type': request.request_type.value,
                        'prompt': request.prompt,
                        'max_tokens': request.max_tokens,
                        'temperature': request.temperature,
                        'complexity': request.complexity,
                    }
                )
                messages.append(msg)
                self.sent_messages += 1
        
        return messages
    
    def get_network_status(self) -> Dict:
        """Get full network status"""
        return {
            **self.get_status(),
            'peers': self.discovery.get_status(),
            'messages_sent': self.sent_messages,
            'messages_received': self.received_messages,
        }


# ============================================================
# 4. Demo: Two nodes talking
# ============================================================

def demo():
    """Demonstrate P2P networking"""
    print("=" * 60)
    print("DecentralAI Network Layer - Demo")
    print("=" * 60)
    
    # Node A: L1 (like current machine)
    caps_a = NodeCapabilities(
        level=NodeLevel.L1_LIGHT_INFERENCE,
        architectures=[ModelArchitecture.RWKV],
        max_memory_mb=8192,
        compute_flops=50e9,
        bandwidth_mbps=100,
        storage_gb=1400
    )
    node_a = NetworkNode(NodeIdentity(capabilities=caps_a))
    node_a.add_expert(ExpertModel(
        base_model="rwkv-4-169m-pile",
        architecture=ModelArchitecture.RWKV,
        domain="general",
        param_count_mb=646,
        avg_tokens_per_sec=5.8,
        success_rate=0.0,
    ))
    
    # Node B: L2 (RTX 3060)
    caps_b = NodeCapabilities(
        level=NodeLevel.L2_STANDARD_INFERENCE,
        architectures=[ModelArchitecture.TRANSFORMER],
        max_memory_mb=8192,
        compute_flops=12000e9,  # 12 TFLOPS
        bandwidth_mbps=1000,
        storage_gb=50000
    )
    node_b = NetworkNode(NodeIdentity(capabilities=caps_b))
    node_b.add_expert(ExpertModel(
        base_model="qwen2.5-coder-7b",
        architecture=ModelArchitecture.TRANSFORMER,
        domain="code",
        param_count_mb=4000,
        avg_tokens_per_sec=25,
        success_rate=0.45,  # After some fine-tuning
    ))
    
    # Node A announces itself to Node B
    print("\n[1] Node A announces to Node B...")
    announce = node_a.announce_self()
    print(f"  Announce: {announce.msg_type.value} from {announce.sender_id}")
    print(f"  Experts: {len(announce.payload['experts'])}")
    
    # Node B processes the announcement
    response = node_b.handle_message(announce)
    print(f"  Node B now knows {len(node_b.discovery.known_peers)} peer(s)")
    
    # Node B announces back
    announce_b = node_b.announce_self()
    node_a.handle_message(announce_b)
    print(f"  Node A now knows {len(node_a.discovery.known_peers)} peer(s)")
    
    # Node A (L1) gets a code request it can't handle well
    print("\n[2] Node A receives complex code request...")
    request = InferenceRequest(
        request_type=RequestType.CODE_GENERATION,
        prompt="def quicksort(arr):",
        complexity=4,
        redundancy_count=2,
        require_verification=True,
    )
    
    # Local routing
    routes = node_a.router.route(request)
    print(f"  Local experts: {len(routes)}")
    for eid, score in routes:
        e = node_a.router.experts[eid]
        print(f"    {e.base_model} ({e.architecture.value}) score={score:.3f}")
    
    # Forward to remote (Node B has better code expert)
    forwards = node_a.forward_request(request)
    print(f"  Forwarding to {len(forwards)} remote node(s)")
    
    # Simulate: Node B handles the request
    if forwards:
        print(f"\n[3] Node B handles forwarded request...")
        response_msg = node_b.handle_message(forwards[0])
        if response_msg:
            print(f"  Response from Node B's code expert")
            print(f"  Tokens: {response_msg.payload.get('tokens_generated', 0)}")
    
    # Network status
    print(f"\n[4] Network status:")
    print(f"  Node A: {json.dumps(node_a.get_network_status(), indent=2)}")
    print(f"  Node B: {json.dumps(node_b.get_network_status(), indent=2)}")
    
    print("\n--- Heterogeneous Network ---")
    print("Node A (L1/RWKV/169M):  General chat, light tasks")
    print("Node B (L2/Transformer/7B): Code generation, complex tasks")
    print("")
    print("The Router sends each request to the right expert,")
    print("regardless of where it lives in the network.")
    print("Like a market: tasks flow to who does them best.")


if __name__ == "__main__":
    demo()
