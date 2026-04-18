"""
DecentralAI Node Framework - Core Module
=========================================
The foundation of the decentralized AI network.

Each node is an independent agent that can:
1. Receive inference requests
2. Route to appropriate experts (local or remote)
3. Verify results from other nodes
4. Evolve through LoRA fine-tuning
5. Earn credits for contributions

Architecture:
    Node -> Router -> Expert Pool -> Verification -> Credit System
"""

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# 1. Identity & Levels
# ============================================================

class NodeLevel(Enum):
    """Five-tier node hierarchy - each role matters"""
    L0_COLLECTOR = 0      # CPU only - data, validation, relay
    L1_LIGHT_INFERENCE = 1 # CPU+4GB - 0.5B-1.5B models
    L2_STANDARD_INFERENCE = 2  # 3060 8GB - 7B models
    L3_HEAVY_INFERENCE = 3     # 3090/4090 - 14B+ models
    L4_DATA_CENTER = 4         # A100/H100 - 70B backbone


class ModelArchitecture(Enum):
    """Supported model architectures for heterogeneous network"""
    TRANSFORMER = "transformer"
    RWKV = "rwkv"
    MAMBA = "mamba"
    XLSTM = "xlstm"


@dataclass
class NodeCapabilities:
    """What a node can do"""
    level: NodeLevel
    architectures: List[ModelArchitecture]
    max_memory_mb: int          # Available RAM/VRAM
    compute_flops: float        # Theoretical FLOPS
    bandwidth_mbps: float       # Network bandwidth
    storage_gb: float           # Available disk
    uptime_hours: float = 0     # Historical uptime


@dataclass
class NodeIdentity:
    """Who a node is"""
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    public_key: str = ""        # For message signing
    capabilities: NodeCapabilities = None
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    
    def fingerprint(self) -> str:
        """Unique fingerprint for this node"""
        data = f"{self.node_id}:{self.capabilities.level.name}:{self.capabilities.max_memory_mb}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


# ============================================================
# 2. Expert Model
# ============================================================

@dataclass
class ExpertModel:
    """
    A single expert in the network.
    Each node hosts one or more experts.
    """
    expert_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    base_model: str = ""                # e.g., "rwkv-4-world-430m"
    architecture: ModelArchitecture = ModelArchitecture.RWKV
    domain: str = "general"             # code, math, reasoning, chat, etc.
    lora_path: Optional[str] = None     # Path to LoRA adapter
    lora_version: int = 0               # Evolution generation
    
    # Performance metrics
    avg_latency_ms: float = 0
    avg_tokens_per_sec: float = 0
    success_rate: float = 0
    total_requests: int = 0
    
    # Size info
    param_count_mb: float = 0
    
    def can_handle(self, request_type: str, complexity: int) -> bool:
        """Check if this expert can handle a request"""
        domain_match = self.domain == request_type or self.domain == "general"
        load_ok = self.total_requests < 100  # Simple load check
        return domain_match and load_ok
    
    def score_for(self, request_type: str) -> float:
        """Score this expert for a given request type (higher = better match)"""
        domain_score = 1.0 if self.domain == request_type else 0.5 if self.domain == "general" else 0.1
        performance_score = self.success_rate * 0.5 + min(self.avg_tokens_per_sec / 20, 1.0) * 0.3 + (1 - self.avg_latency_ms / 5000) * 0.2
        return domain_score * 0.6 + performance_score * 0.4


# ============================================================
# 3. Request & Response
# ============================================================

class RequestType(Enum):
    CODE_GENERATION = "code"
    CODE_REVIEW = "code_review"
    MATH_REASONING = "math"
    GENERAL_CHAT = "chat"
    DATA_ANALYSIS = "data"
    TRANSLATION = "translation"


@dataclass
class InferenceRequest:
    """A request entering the network"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_type: RequestType = RequestType.GENERAL_CHAT
    prompt: str = ""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    complexity: int = 1              # 1-5 scale
    requester_id: str = ""
    timestamp: float = field(default_factory=time.time)
    deadline_ms: float = 30000       # Max wait time
    
    # Routing hints
    preferred_architecture: Optional[ModelArchitecture] = None
    require_verification: bool = True
    redundancy_count: int = 1        # How many experts to query


@dataclass
class InferenceResponse:
    """A response from an expert"""
    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = ""
    expert_id: str = ""
    node_id: str = ""
    text: str = ""
    tokens_generated: int = 0
    latency_ms: float = 0
    tokens_per_sec: float = 0
    architecture: ModelArchitecture = ModelArchitecture.RWKV
    timestamp: float = field(default_factory=time.time)
    
    # Verification
    verified: bool = False
    verification_score: float = 0     # 0-1, how confident
    verifier_ids: List[str] = field(default_factory=list)
    
    def hash_content(self) -> str:
        """Hash of response content for verification"""
        return hashlib.sha256(f"{self.request_id}:{self.text}".encode()).hexdigest()[:16]


# ============================================================
# 4. Router - The Brain of the Network
# ============================================================

class Router:
    """
    Routes requests to the best available expert.
    This is the "invisible hand" that makes the network self-organize.
    
    Like market prices in human society, the router doesn't command -
    it discovers. It matches supply (experts) with demand (requests)
    based on capability, reputation, and availability.
    """
    
    def __init__(self):
        self.experts: Dict[str, ExpertModel] = {}
        self.node_experts: Dict[str, List[str]] = {}  # node_id -> [expert_ids]
        self.routing_history: List[Dict] = []
    
    def register_expert(self, node_id: str, expert: ExpertModel):
        """Register an expert with the router"""
        self.experts[expert.expert_id] = expert
        if node_id not in self.node_experts:
            self.node_experts[node_id] = []
        self.node_experts[node_id].append(expert.expert_id)
    
    def unregister_expert(self, expert_id: str):
        """Remove an expert"""
        if expert_id in self.experts:
            del self.experts[expert_id]
            for node_id, expert_ids in self.node_experts.items():
                if expert_id in expert_ids:
                    expert_ids.remove(expert_id)
    
    def route(self, request: InferenceRequest) -> List[Tuple[str, float]]:
        """
        Find the best experts for a request.
        Returns [(expert_id, score), ...] sorted by score.
        
        The routing logic mirrors how society allocates tasks:
        - Capability match (can you do the job?)
        - Reputation (are you good at it?)
        - Availability (are you free?)
        - Cost efficiency (are you worth it?)
        """
        candidates = []
        
        for expert_id, expert in self.experts.items():
            if not expert.can_handle(request.request_type.value, request.complexity):
                continue
            
            # Architecture preference
            if request.preferred_architecture and expert.architecture != request.preferred_architecture:
                if expert.architecture != ModelArchitecture.TRANSFORMER:
                    continue  # Non-preferred non-transformer experts skip
            
            score = expert.score_for(request.request_type.value)
            
            # Complexity penalty: small models penalized for complex tasks
            if request.complexity >= 4 and expert.param_count_mb < 3000:
                score *= 0.3
            elif request.complexity >= 3 and expert.param_count_mb < 1000:
                score *= 0.5
            
            candidates.append((expert_id, score))
        
        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N based on redundancy
        n = min(request.redundancy_count, len(candidates))
        selected = candidates[:n]
        
        # Log routing decision
        self.routing_history.append({
            'request_id': request.request_id,
            'type': request.request_type.value,
            'candidates': len(candidates),
            'selected': [(eid, round(s, 3)) for eid, s in selected],
            'timestamp': time.time()
        })
        
        return selected


# ============================================================
# 5. Verification System
# ============================================================

class VerificationMethod(Enum):
    """How to verify a response"""
    SYNTAX_CHECK = "syntax"
    REDUNDANCY_CONSENSUS = "consensus"
    L0_HUMAN_ANNOTATION = "annotation"
    EXECUTION_TEST = "execution"
    CROSS_ARCHITECTURE = "cross_arch"


class Verifier:
    """
    Verify inference results. The "quality control" layer.
    
    Three-tier verification (from DESIGN.md):
    1. Requester annotation (L0 nodes can do this)
    2. Redundancy consensus (multiple experts agree)
    3. Reputation-weighted scoring
    """
    
    def __init__(self):
        self.verification_results: Dict[str, Dict] = {}
    
    def verify_syntax(self, response: InferenceResponse, language: str = "python") -> Tuple[bool, float]:
        """Tier 1: Quick syntax check - any node can do this"""
        code = response.text
        if language == "python":
            try:
                compile(code, '<string>', 'exec')
                return True, 0.7  # Syntax OK but may be logically wrong
            except SyntaxError:
                return False, 0.1
        return False, 0.0
    
    def verify_consensus(self, responses: List[InferenceResponse]) -> Dict[str, float]:
        """Tier 2: Redundancy consensus - do multiple experts agree?"""
        if len(responses) < 2:
            return {r.response_id: 0.5 for r in responses}
        
        results = {}
        for i, resp_a in enumerate(responses):
            agreement_count = 0
            for j, resp_b in enumerate(responses):
                if i == j:
                    continue
                # Simple text similarity
                similarity = self._text_similarity(resp_a.text, resp_b.text)
                if similarity > 0.6:
                    agreement_count += 1
            
            # Score based on how many others agree
            agreement_rate = agreement_count / max(len(responses) - 1, 1)
            results[resp_a.response_id] = 0.5 + 0.5 * agreement_rate
        
        return results
    
    def _text_similarity(self, a: str, b: str) -> float:
        """Simple Jaccard similarity on words"""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)
    
    def verify_execution(self, response: InferenceResponse, test_cases: List[Dict]) -> Tuple[bool, float]:
        """Tier 3: Execute code against test cases (requires sandbox)"""
        # This would run in a sandboxed environment
        # For now, just check structure
        passed = 0
        total = len(test_cases)
        if total == 0:
            return False, 0.0
        return passed == total, passed / total


# ============================================================
# 6. Credit System - The Economic Layer
# ============================================================

@dataclass
class CreditTransaction:
    """A credit transaction"""
    tx_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    from_node: str = ""
    to_node: str = ""
    amount: float = 0
    reason: str = ""
    timestamp: float = field(default_factory=time.time)
    block_hash: str = ""  # On-chain anchor


class CreditLedger:
    """
    Track credits earned by nodes.
    Every contribution has value - this is "not everyone is equal, but everyone has a path."
    
    Credit rates by level:
    - L0: 1 credit/task (data, validation)
    - L1: 5 credits/task (light inference)
    - L2: 20 credits/task (standard inference)
    - L3: 50 credits/task (heavy inference)
    - L4: 200 credits/task (backbone inference)
    """
    
    CREDIT_RATES = {
        NodeLevel.L0_COLLECTOR: 1,
        NodeLevel.L1_LIGHT_INFERENCE: 5,
        NodeLevel.L2_STANDARD_INFERENCE: 20,
        NodeLevel.L3_HEAVY_INFERENCE: 50,
        NodeLevel.L4_DATA_CENTER: 200,
    }
    
    def __init__(self):
        self.balances: Dict[str, float] = {}
        self.transactions: List[CreditTransaction] = []
        self.reputation: Dict[str, float] = {}  # node_id -> 0-100
    
    def credit(self, node_id: str, amount: float, reason: str, from_node: str = "network"):
        """Add credits to a node"""
        if node_id not in self.balances:
            self.balances[node_id] = 0
        
        self.balances[node_id] += amount
        self.transactions.append(CreditTransaction(
            from_node=from_node,
            to_node=node_id,
            amount=amount,
            reason=reason
        ))
        
        # Update reputation
        if node_id not in self.reputation:
            self.reputation[node_id] = 50  # Start neutral
        self.reputation[node_id] = min(100, self.reputation[node_id] + amount * 0.01)
    
    def debit(self, node_id: str, amount: float, reason: str, to_node: str = "network"):
        """Deduct credits from a node"""
        if node_id not in self.balances:
            self.balances[node_id] = 0
        
        self.balances[node_id] -= amount
        self.transactions.append(CreditTransaction(
            from_node=node_id,
            to_node=to_node,
            amount=amount,
            reason=reason
        ))
    
    def reward_inference(self, node_id: str, level: NodeLevel, quality: float):
        """Reward a node for inference work"""
        base = self.CREDIT_RATES[level]
        reward = base * quality  # Quality-adjusted
        self.credit(node_id, reward, f"inference_{level.name}")
    
    def reward_validation(self, node_id: str, correct: bool):
        """Reward an L0/L1 node for validation"""
        if correct:
            self.credit(node_id, 1, "validation_correct")
            # 5% rebate to requester + quality bonus
        else:
            self.credit(node_id, 0.5, "validation_attempt")


# ============================================================
# 7. Node - The Complete Agent
# ============================================================

class Node:
    """
    A DecentralAI node. The atomic unit of the network.
    
    Each node is like a person in society:
    - Has capabilities (some are strong, some are weak)
    - Has a role (collector, worker, validator)
    - Earns credits for contributions
    - Can evolve (learn new skills via LoRA)
    - Has reputation (built over time through work quality)
    """
    
    def __init__(self, identity: NodeIdentity):
        self.identity = identity
        self.experts: List[ExpertModel] = []
        self.router = Router()
        self.verifier = Verifier()
        self.ledger = CreditLedger()
        self.peers: Dict[str, NodeIdentity] = {}
        self.is_online = True
    
    def add_expert(self, expert: ExpertModel):
        """Add an expert model to this node"""
        self.experts.append(expert)
        self.router.register_expert(self.identity.node_id, expert)
    
    def handle_request(self, request: InferenceRequest) -> Optional[InferenceResponse]:
        """Handle an incoming inference request"""
        if not self.is_online:
            return None
        
        # Route to best local expert
        candidates = self.router.route(request)
        if not candidates:
            return None
        
        expert_id, score = candidates[0]
        expert = self.router.experts.get(expert_id)
        if not expert:
            return None
        
        # This is where actual inference happens
        # For now, return a placeholder
        response = InferenceResponse(
            request_id=request.request_id,
            expert_id=expert_id,
            node_id=self.identity.node_id,
            architecture=expert.architecture,
        )
        
        return response
    
    def verify_response(self, response: InferenceResponse) -> Tuple[bool, float]:
        """Verify another node's response (L0/L1 can do this)"""
        return self.verifier.verify_syntax(response)
    
    def get_status(self) -> Dict:
        """Get node status"""
        return {
            'node_id': self.identity.node_id,
            'fingerprint': self.identity.fingerprint(),
            'level': self.identity.capabilities.level.name,
            'experts': len(self.experts),
            'peers': len(self.peers),
            'credits': self.ledger.balances.get(self.identity.node_id, 0),
            'reputation': self.ledger.reputation.get(self.identity.node_id, 0),
            'online': self.is_online,
        }


# ============================================================
# 8. Quick Demo
# ============================================================

def demo():
    """Demonstrate the DecentralAI node framework"""
    print("=" * 60)
    print("DecentralAI Node Framework - Demo")
    print("=" * 60)
    
    # Create an L1 node (like the current machine)
    caps = NodeCapabilities(
        level=NodeLevel.L1_LIGHT_INFERENCE,
        architectures=[ModelArchitecture.RWKV],
        max_memory_mb=8192,
        compute_flops=50e9,  # ~50 GFLOPS for Pentium G4560
        bandwidth_mbps=100,
        storage_gb=1400
    )
    identity = NodeIdentity(capabilities=caps)
    node = Node(identity)
    
    # Add an expert (RWKV-4-169M)
    expert = ExpertModel(
        base_model="rwkv-4-169m-pile",
        architecture=ModelArchitecture.RWKV,
        domain="general",
        param_count_mb=646,
        avg_tokens_per_sec=5.8,
        success_rate=0.0,  # Baseline: 0% HumanEval
    )
    node.add_expert(expert)
    
    # Simulate a request
    request = InferenceRequest(
        request_type=RequestType.CODE_GENERATION,
        prompt="def add(a, b):",
        max_tokens=100,
        complexity=2,
        require_verification=True,
        redundancy_count=3
    )
    
    # Route
    routes = node.router.route(request)
    print(f"\nRequest: {request.request_type.value}")
    print(f"Routing: {len(routes)} experts selected")
    for eid, score in routes:
        e = node.router.experts[eid]
        print(f"  Expert {eid}: {e.base_model} ({e.architecture.value}) score={score:.3f}")
    
    # Verify
    response = node.handle_request(request)
    if response:
        ok, score = node.verify_response(response)
        print(f"\nVerification: syntax={'PASS' if ok else 'FAIL'}, score={score:.2f}")
    
    # Credit
    node.ledger.reward_inference(node.identity.node_id, caps.level, quality=0.3)
    print(f"Credits earned: {node.ledger.balances.get(node.identity.node_id, 0):.1f}")
    
    # Status
    print(f"\nNode status: {json.dumps(node.get_status(), indent=2)}")
    
    print("\n--- Network Vision ---")
    print("L0 (Raspberry Pi): 1 credit/task  -> ~2 CNY/month")
    print("L1 (This machine): 5 credits/task -> ~10 CNY/month")
    print("L2 (RTX 3060):     20 credits/task -> ~65 CNY/month")
    print("L3 (RTX 3090):     50 credits/task -> ~200 CNY/month")
    print("L4 (A100):         200 credits/task -> ~2000 CNY/month")
    print("\nEvery node has a path. Not equal, but fair.")


if __name__ == "__main__":
    demo()
