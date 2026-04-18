"""
DecentralAI Evolution Engine
=============================
The self-improvement layer. This is what makes DecentralAI alive.

Cycle: Observe -> Reflect -> Evolve -> Verify
Each node independently runs this cycle, improving its experts over time.
The network improves not by central command, but by distributed evolution.

Inspired by:
- EvoAgent's Neuroplasticity Engine
- FedMoECap's Rolling strategy
- Natural selection in biological systems
"""

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Import from core
from core import (
    Node, NodeLevel, ExpertModel, ModelArchitecture,
    InferenceRequest, InferenceResponse, RequestType,
    Router, Verifier, CreditLedger
)


# ============================================================
# 1. Observation - What happened?
# ============================================================

class ObservationType(Enum):
    """Types of observations a node can make"""
    REQUEST_RECEIVED = "request"
    RESPONSE_GENERATED = "response"
    VERIFICATION_RESULT = "verification"
    PEER_BEHAVIOR = "peer"
    PERFORMANCE_METRIC = "metric"
    ERROR_OCCURRED = "error"


@dataclass
class Observation:
    """A single observation event"""
    obs_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    obs_type: ObservationType = ObservationType.REQUEST_RECEIVED
    timestamp: float = field(default_factory=time.time)
    expert_id: str = ""
    data: Dict = field(default_factory=dict)
    # e.g., {"success": True, "latency_ms": 150, "tokens": 50}


class ObservationBuffer:
    """
    Collects observations. The node's memory of recent events.
    Like short-term memory - keeps recent, forgets old.
    """
    
    def __init__(self, max_size: int = 1000):
        self.buffer: List[Observation] = []
        self.max_size = max_size
    
    def observe(self, obs_type: ObservationType, expert_id: str = "", **data):
        """Record an observation"""
        obs = Observation(
            obs_type=obs_type,
            expert_id=expert_id,
            data=data
        )
        self.buffer.append(obs)
        
        # Trim old observations
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]
    
    def recent(self, n: int = 100) -> List[Observation]:
        """Get recent observations"""
        return self.buffer[-n:]
    
    def for_expert(self, expert_id: str) -> List[Observation]:
        """Get observations for a specific expert"""
        return [o for o in self.buffer if o.expert_id == expert_id]
    
    def success_rate(self, expert_id: str, window: int = 50) -> float:
        """Calculate recent success rate for an expert"""
        obs = [o for o in self.for_expert(expert_id) 
               if o.obs_type == ObservationType.VERIFICATION_RESULT]
        if not obs:
            return 0.5
        recent = obs[-window:]
        successes = sum(1 for o in recent if o.data.get('passed', False))
        return successes / len(recent)
    
    def error_patterns(self, expert_id: str) -> Dict[str, int]:
        """Find common error patterns"""
        obs = [o for o in self.for_expert(expert_id)
               if o.obs_type == ObservationType.VERIFICATION_RESULT 
               and not o.data.get('passed', False)]
        
        patterns = {}
        for o in obs:
            error_type = o.data.get('error_type', 'unknown')
            patterns[error_type] = patterns.get(error_type, 0) + 1
        return patterns


# ============================================================
# 2. Reflection - What does it mean?
# ============================================================

@dataclass
class Reflection:
    """An insight derived from observations"""
    reflection_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    expert_id: str = ""
    insight: str = ""
    confidence: float = 0.0  # 0-1
    action: str = ""         # What to do about it
    priority: float = 0.0    # How important
    timestamp: float = field(default_factory=time.time)


class Reflector:
    """
    Analyzes observations and produces insights.
    The node's ability to learn from experience.
    
    Like a person reflecting on their work:
    - "I keep making syntax errors in Python"
    - "My math answers are usually wrong"
    - "I'm good at simple tasks but fail on complex ones"
    """
    
    def __init__(self, observation_buffer: ObservationBuffer):
        self.buffer = observation_buffer
        self.reflections: List[Reflection] = []
    
    def reflect(self, expert_id: str) -> List[Reflection]:
        """Generate reflections for an expert"""
        reflections = []
        
        # Pattern 1: Low success rate
        success_rate = self.buffer.success_rate(expert_id)
        if success_rate < 0.3:
            reflections.append(Reflection(
                expert_id=expert_id,
                insight=f"Success rate critically low: {success_rate:.1%}",
                confidence=0.9,
                action="fine_tune",
                priority=1.0 - success_rate
            ))
        elif success_rate < 0.6:
            reflections.append(Reflection(
                expert_id=expert_id,
                insight=f"Success rate below target: {success_rate:.1%}",
                confidence=0.7,
                action="collect_more_data",
                priority=0.6 - success_rate
            ))
        
        # Pattern 2: Common error types
        errors = self.buffer.error_patterns(expert_id)
        for error_type, count in sorted(errors.items(), key=lambda x: -x[1]):
            if count >= 3:
                reflections.append(Reflection(
                    expert_id=expert_id,
                    insight=f"Recurring error: {error_type} ({count} times)",
                    confidence=min(count / 10, 0.9),
                    action="targeted_training",
                    priority=min(count * 0.1, 0.8)
                ))
        
        # Pattern 3: Domain-specific weakness
        domain_obs = [o for o in self.buffer.for_expert(expert_id)
                      if o.obs_type == ObservationType.REQUEST_RECEIVED]
        domain_failures = {}
        for o in domain_obs:
            dtype = o.data.get('request_type', 'unknown')
            if dtype not in domain_failures:
                domain_failures[dtype] = {'total': 0, 'failed': 0}
            domain_failures[dtype]['total'] += 1
        
        # Count failures
        for o in self.buffer.for_expert(expert_id):
            if o.obs_type == ObservationType.VERIFICATION_RESULT:
                dtype = o.data.get('request_type', 'unknown')
                if not o.data.get('passed', False) and dtype in domain_failures:
                    domain_failures[dtype]['failed'] += 1
        
        for dtype, stats in domain_failures.items():
            if stats['total'] > 5:
                fail_rate = stats['failed'] / stats['total']
                if fail_rate > 0.5:
                    reflections.append(Reflection(
                        expert_id=expert_id,
                        insight=f"Weak domain: {dtype} ({fail_rate:.0%} failure rate)",
                        confidence=0.6,
                        action="domain_specialization",
                        priority=fail_rate * 0.7
                    ))
        
        self.reflections.extend(reflections)
        return reflections
    
    def top_priority(self, n: int = 3) -> List[Reflection]:
        """Get top priority reflections"""
        return sorted(self.reflections, key=lambda r: r.priority, reverse=True)[:n]


# ============================================================
# 3. Evolution - What to change?
# ============================================================

class EvolutionAction(Enum):
    """Possible evolution actions"""
    LORA_FINE_TUNE = "lora_finetune"
    TARGETED_TRAINING = "targeted_training"
    DOMAIN_SPECIALIZATION = "domain_specialization"
    ROLLBACK = "rollback"
    NO_ACTION = "no_action"


@dataclass
class EvolutionPlan:
    """A plan to evolve an expert"""
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    expert_id: str = ""
    action: EvolutionAction = EvolutionAction.NO_ACTION
    data_source: str = ""       # Where to get training data
    lora_r: int = 8             # LoRA rank
    lora_alpha: int = 16
    epochs: int = 3
    learning_rate: float = 1e-4
    target_domain: str = ""
    expected_improvement: float = 0.0  # Estimated success rate improvement
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)


class Evolver:
    """
    Decides how to evolve based on reflections.
    
    Like a person deciding to study:
    - "I'll practice Python problems tonight"
    - "I need to review math fundamentals"
    - "Maybe I should specialize in web development"
    
    Evolution is constrained by:
    - Available training data
    - Compute budget (credits)
    - Risk tolerance (don't break what works)
    """
    
    def __init__(self, reflector: Reflector, credit_ledger: CreditLedger):
        self.reflector = reflector
        self.ledger = credit_ledger
        self.evolution_history: List[Dict] = []
        self.generation: int = 0
    
    def plan_evolution(self, expert: ExpertModel) -> Optional[EvolutionPlan]:
        """Create an evolution plan for an expert"""
        reflections = self.reflector.reflect(expert.expert_id)
        if not reflections:
            return None
        
        top = reflections[0]  # Highest priority
        
        # Map reflection action to evolution action
        action_map = {
            'fine_tune': EvolutionAction.LORA_FINE_TUNE,
            'collect_more_data': EvolutionAction.NO_ACTION,  # Need data first
            'targeted_training': EvolutionAction.TARGETED_TRAINING,
            'domain_specialization': EvolutionAction.DOMAIN_SPECIALIZATION,
        }
        
        action = action_map.get(top.action, EvolutionAction.NO_ACTION)
        if action == EvolutionAction.NO_ACTION:
            return None
        
        # Check compute budget
        estimated_cost = self._estimate_cost(action, expert)
        # For MVP, always allow evolution
        
        plan = EvolutionPlan(
            expert_id=expert.expert_id,
            action=action,
            lora_r=8,
            lora_alpha=16,
            epochs=3,
            learning_rate=1e-4,
            expected_improvement=top.priority * 0.3,  # Conservative estimate
            confidence=top.confidence
        )
        
        if action == EvolutionAction.DOMAIN_SPECIALIZATION:
            plan.target_domain = top.insight.split(": ")[1].split(" ")[0] if ": " in top.insight else ""
        
        self.evolution_history.append({
            'plan_id': plan.plan_id,
            'expert_id': expert.expert_id,
            'action': action.value,
            'generation': self.generation,
            'based_on': top.insight,
            'timestamp': time.time()
        })
        
        return plan
    
    def execute_evolution(self, plan: EvolutionPlan, expert: ExpertModel) -> ExpertModel:
        """
        Execute an evolution plan.
        In production, this would:
        1. Gather training data
        2. Fine-tune LoRA adapter
        3. Verify improvement
        4. Hot-swap to new version
        
        For now, update metadata.
        """
        self.generation += 1
        
        # Create evolved expert
        evolved = ExpertModel(
            expert_id=expert.expert_id,
            base_model=expert.base_model,
            architecture=expert.architecture,
            domain=plan.target_domain if plan.target_domain else expert.domain,
            lora_path=f"lora_gen{self.generation}",
            lora_version=expert.lora_version + 1,
            avg_latency_ms=expert.avg_latency_ms,
            avg_tokens_per_sec=expert.avg_tokens_per_sec,
            success_rate=expert.success_rate,  # Will be updated after verification
            total_requests=0,  # Reset for new generation
            param_count_mb=expert.param_count_mb,
        )
        
        return evolved
    
    def _estimate_cost(self, action: EvolutionAction, expert: ExpertModel) -> float:
        """Estimate credit cost of evolution"""
        base_costs = {
            EvolutionAction.LORA_FINE_TUNE: 50,
            EvolutionAction.TARGETED_TRAINING: 30,
            EvolutionAction.DOMAIN_SPECIALIZATION: 80,
            EvolutionAction.ROLLBACK: 5,
        }
        return base_costs.get(action, 0)


# ============================================================
# 4. Verification - Did it work?
# ============================================================

class EvolutionVerifier:
    """
    Verify that evolution actually improved things.
    Never deploy without verification - that's how you get regression.
    
    Process:
    1. Run benchmark on old version
    2. Run benchmark on new version
    3. Compare results
    4. Accept/Reject/Rollback
    """
    
    def __init__(self):
        self.benchmark_results: Dict[str, List[Dict]] = {}
    
    def verify_evolution(self, 
                         old_expert: ExpertModel,
                         new_expert: ExpertModel,
                         benchmark_scores: Dict[str, Tuple[float, float]]  # {test: (old_score, new_score)}
                         ) -> Tuple[bool, float]:
        """
        Verify that the evolution improved performance.
        Returns (accepted, improvement_ratio)
        """
        improvements = []
        regressions = []
        
        for test_name, (old_score, new_score) in benchmark_scores.items():
            delta = new_score - old_score
            if delta > 0:
                improvements.append(delta)
            elif delta < -0.05:  # Tolerate small regression
                regressions.append(delta)
        
        if not improvements and not regressions:
            return False, 0.0
        
        # Accept if net improvement and no major regression
        total_improvement = sum(improvements)
        total_regression = sum(regressions)
        
        if total_regression < -0.2:  # Major regression
            return False, 0.0
        
        if total_improvement > 0:
            improvement_ratio = total_improvement / max(len(improvements), 1)
            return True, improvement_ratio
        
        return False, 0.0


# ============================================================
# 5. Evolution Cycle - The Complete Loop
# ============================================================

class EvolutionCycle:
    """
    The complete evolution cycle: Observe -> Reflect -> Evolve -> Verify
    
    This runs continuously on each node. The network improves
    not by central direction, but by each node independently
    getting better at what it does.
    
    Like how a market improves: not by someone planning it,
    but by millions of individuals each optimizing their corner.
    """
    
    def __init__(self, node: Node):
        self.node = node
        self.buffer = ObservationBuffer()
        self.reflector = Reflector(self.buffer)
        self.evolver = Evolver(self.reflector, node.ledger)
        self.evolution_verifier = EvolutionVerifier()
        self.cycle_count = 0
        self.auto_evolve = True
    
    def observe_request(self, request: InferenceRequest):
        """Step 1: Observe incoming request"""
        self.buffer.observe(
            ObservationType.REQUEST_RECEIVED,
            data={'request_type': request.request_type.value,
                  'complexity': request.complexity,
                  'prompt_length': len(request.prompt)}
        )
    
    def observe_response(self, response: InferenceResponse):
        """Step 1b: Observe generated response"""
        self.buffer.observe(
            ObservationType.RESPONSE_GENERATED,
            expert_id=response.expert_id,
            data={'latency_ms': response.latency_ms,
                  'tokens': response.tokens_generated,
                  'tok_per_sec': response.tokens_per_sec}
        )
    
    def observe_verification(self, expert_id: str, passed: bool, error_type: str = ""):
        """Step 1c: Observe verification result"""
        self.buffer.observe(
            ObservationType.VERIFICATION_RESULT,
            expert_id=expert_id,
            data={'passed': passed, 'error_type': error_type}
        )
    
    def run_cycle(self) -> Optional[EvolutionPlan]:
        """
        Run one complete evolution cycle.
        Returns an evolution plan if one is warranted.
        """
        self.cycle_count += 1
        
        # Step 2: Reflect
        for expert in self.node.experts:
            reflections = self.reflector.reflect(expert.expert_id)
        
        # Step 3: Evolve
        plans = []
        for expert in self.node.experts:
            plan = self.evolver.plan_evolution(expert)
            if plan:
                plans.append(plan)
        
        if not plans:
            return None
        
        # Return highest priority plan
        plans.sort(key=lambda p: p.confidence, reverse=True)
        return plans[0]
    
    def execute_and_verify(self, plan: EvolutionPlan) -> Tuple[bool, float]:
        """Execute evolution plan and verify results"""
        # Find expert
        expert = None
        for e in self.node.experts:
            if e.expert_id == plan.expert_id:
                expert = e
                break
        
        if not expert:
            return False, 0.0
        
        # Evolve
        new_expert = self.evolver.execute_evolution(plan, expert)
        
        # Verify (would run actual benchmarks here)
        # For now, assume improvement based on plan confidence
        accepted = plan.confidence > 0.5
        improvement = plan.expected_improvement if accepted else 0
        
        if accepted:
            # Update expert in-place
            idx = self.node.experts.index(expert)
            self.node.experts[idx] = new_expert
            print(f"  Evolution accepted! Gen {new_expert.lora_version}, "
                  f"domain={new_expert.domain}, expected +{improvement:.1%}")
        else:
            print(f"  Evolution rejected. Keeping Gen {expert.lora_version}")
        
        return accepted, improvement
    
    def get_status(self) -> Dict:
        """Get evolution cycle status"""
        return {
            'cycle_count': self.cycle_count,
            'observations': len(self.buffer.buffer),
            'reflections': len(self.reflector.reflections),
            'evolutions': len(self.evolver.evolution_history),
            'generation': self.evolver.generation,
            'auto_evolve': self.auto_evolve,
        }


# ============================================================
# 6. Demo
# ============================================================

def demo():
    """Demonstrate the evolution cycle"""
    print("=" * 60)
    print("DecentralAI Evolution Engine - Demo")
    print("=" * 60)
    
    # Create node with L1 capabilities
    from core import NodeCapabilities, NodeIdentity
    caps = NodeCapabilities(
        level=NodeLevel.L1_LIGHT_INFERENCE,
        architectures=[ModelArchitecture.RWKV],
        max_memory_mb=8192,
        compute_flops=50e9,
        bandwidth_mbps=100,
        storage_gb=1400
    )
    identity = NodeIdentity(capabilities=caps)
    node = Node(identity)
    
    # Add expert
    expert = ExpertModel(
        base_model="rwkv-4-169m-pile",
        architecture=ModelArchitecture.RWKV,
        domain="general",
        param_count_mb=646,
        avg_tokens_per_sec=5.8,
        success_rate=0.0,
    )
    node.add_expert(expert)
    
    # Create evolution cycle
    cycle = EvolutionCycle(node)
    
    # Simulate observations
    print("\n[1] Simulating observations...")
    for i in range(20):
        # Simulate requests
        cycle.observe_request(InferenceRequest(
            request_type=RequestType.CODE_GENERATION,
            prompt="test", complexity=2
        ))
        # Simulate mostly failures (0% baseline)
        cycle.observe_response(InferenceResponse(
            expert_id=expert.expert_id,
            tokens_generated=50,
            latency_ms=1000,
            tokens_per_sec=5.0
        ))
        # Most verifications fail
        passed = i % 10 == 0  # 10% pass rate
        error_type = "" if passed else "syntax_error"
        cycle.observe_verification(expert.expert_id, passed, error_type)
    
    # Run evolution cycle
    print("\n[2] Running evolution cycle...")
    plan = cycle.run_cycle()
    
    if plan:
        print(f"  Plan: {plan.action.value}")
        print(f"  Expert: {plan.expert_id}")
        print(f"  Confidence: {plan.confidence:.2f}")
        print(f"  Expected improvement: +{plan.expected_improvement:.1%}")
        
        # Execute
        print("\n[3] Executing evolution...")
        accepted, improvement = cycle.execute_and_verify(plan)
        print(f"  Result: {'ACCEPTED' if accepted else 'REJECTED'}")
    
    # Show status
    print(f"\n[4] Evolution status:")
    print(f"  {json.dumps(cycle.get_status(), indent=2)}")
    
    print("\n--- The Cycle Never Stops ---")
    print("Each node continuously:")
    print("  1. OBSERVE  - Watch what happens")
    print("  2. REFLECT  - Find patterns in failures")
    print("  3. EVOLVE   - Train to fix weaknesses")
    print("  4. VERIFY   - Prove improvement before deploying")
    print("")
    print("The network evolves like life evolves:")
    print("  Not by design, but by selection.")
    print("  Not uniformly, but in niches.")
    print("  Not perfectly, but persistently.")


if __name__ == "__main__":
    demo()
