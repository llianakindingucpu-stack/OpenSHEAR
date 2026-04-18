"""
DecentralAI Unit Tests
=======================
Core framework, evolution engine, and network layer tests.
"""

import json
import sys
import os
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ws_transport import P2PNode, ConnectionManager, PeerConnection

from core import (
    Node, NodeIdentity, NodeLevel, NodeCapabilities,
    ExpertModel, ModelArchitecture,
    InferenceRequest, InferenceResponse, RequestType,
    Router, Verifier, CreditLedger,
)
from evolution import (
    EvolutionCycle, ObservationBuffer, Reflector, Evolver,
    ObservationType, EvolutionAction,
)
from network import (
    NetworkNode, PeerDiscovery, Message, MessageType,
)


class TestNodeIdentity(unittest.TestCase):
    def test_create_identity(self):
        caps = NodeCapabilities(
            level=NodeLevel.L1_LIGHT_INFERENCE,
            architectures=[ModelArchitecture.RWKV],
            max_memory_mb=8192,
            compute_flops=50e9,
            bandwidth_mbps=100,
            storage_gb=100,
        )
        identity = NodeIdentity(capabilities=caps)
        self.assertTrue(identity.node_id)
        self.assertEqual(identity.capabilities.level, NodeLevel.L1_LIGHT_INFERENCE)
    
    def test_fingerprint(self):
        caps = NodeCapabilities(
            level=NodeLevel.L2_STANDARD_INFERENCE,
            architectures=[ModelArchitecture.TRANSFORMER],
            max_memory_mb=8192,
            compute_flops=12000e9,
            bandwidth_mbps=1000,
            storage_gb=500,
        )
        identity = NodeIdentity(capabilities=caps)
        fp = identity.fingerprint()
        self.assertTrue(fp)
        self.assertEqual(len(fp), 16)


class TestRouter(unittest.TestCase):
    def setUp(self):
        self.router = Router()
        self.expert_code = ExpertModel(
            expert_id="code_expert",
            base_model="qwen2.5-coder-7b",
            architecture=ModelArchitecture.TRANSFORMER,
            domain="code",
            param_count_mb=4000,
            avg_tokens_per_sec=25,
            success_rate=0.45,
        )
        self.expert_general = ExpertModel(
            expert_id="general_expert",
            base_model="rwkv-4-169m",
            architecture=ModelArchitecture.RWKV,
            domain="general",
            param_count_mb=646,
            avg_tokens_per_sec=5.8,
            success_rate=0.0,
        )
    
    def test_register_expert(self):
        self.router.register_expert("node_a", self.expert_code)
        self.assertIn("code_expert", self.router.experts)
        self.assertIn("node_a", self.router.node_experts)
    
    def test_route_code_request(self):
        self.router.register_expert("node_a", self.expert_code)
        self.router.register_expert("node_b", self.expert_general)
        
        request = InferenceRequest(
            request_type=RequestType.CODE_GENERATION,
            complexity=3,
        )
        routes = self.router.route(request)
        self.assertTrue(len(routes) >= 1)
        # Code expert should rank higher for code requests
        top_id, top_score = routes[0]
        self.assertEqual(top_id, "code_expert")
    
    def test_route_complexity_penalty(self):
        self.router.register_expert("node_b", self.expert_general)
        
        request = InferenceRequest(
            request_type=RequestType.CODE_GENERATION,
            complexity=4,  # High complexity
        )
        routes = self.router.route(request)
        if routes:
            _, score = routes[0]
            # Small model should be penalized for high complexity
            self.assertLess(score, 0.5)


class TestVerifier(unittest.TestCase):
    def test_syntax_valid_python(self):
        verifier = Verifier()
        response = InferenceResponse(text="def add(a, b):\n    return a + b")
        ok, score = verifier.verify_syntax(response, "python")
        self.assertTrue(ok)
        self.assertGreater(score, 0.5)
    
    def test_syntax_invalid_python(self):
        verifier = Verifier()
        response = InferenceResponse(text="def add(a, b)\n    return a +")
        ok, score = verifier.verify_syntax(response, "python")
        self.assertFalse(ok)
        self.assertLess(score, 0.5)
    
    def test_consensus_agreement(self):
        verifier = Verifier()
        responses = [
            InferenceResponse(text="The answer is 42"),
            InferenceResponse(text="The answer is 42"),
            InferenceResponse(text="The answer is 42"),
        ]
        scores = verifier.verify_consensus(responses)
        # Identical responses should get high scores
        self.assertTrue(all(s >= 0.8 for s in scores.values()))


class TestCreditLedger(unittest.TestCase):
    def test_credit_inference(self):
        ledger = CreditLedger()
        ledger.reward_inference("node_1", NodeLevel.L1_LIGHT_INFERENCE, quality=0.8)
        self.assertEqual(ledger.balances["node_1"], 5 * 0.8)
    
    def test_credit_levels(self):
        ledger = CreditLedger()
        for level in NodeLevel:
            ledger.reward_inference(f"node_{level.name}", level, quality=1.0)
        
        # L4 should earn much more than L0
        l4_balance = ledger.balances.get("node_L4_DATA_CENTER", 0)
        l0_balance = ledger.balances.get("node_L0_COLLECTOR", 0)
        self.assertGreater(l4_balance, l0_balance)
        self.assertEqual(l4_balance, 200)  # L4 rate
        self.assertEqual(l0_balance, 1)    # L0 rate
    
    def test_debit(self):
        ledger = CreditLedger()
        ledger.credit("node_1", 100, "test")
        ledger.debit("node_1", 30, "payment")
        self.assertEqual(ledger.balances["node_1"], 70)
    
    def test_validation_reward(self):
        ledger = CreditLedger()
        ledger.reward_validation("node_l0", correct=True)
        self.assertEqual(ledger.balances["node_l0"], 1)
        
        ledger.reward_validation("node_l0_fail", correct=False)
        self.assertEqual(ledger.balances["node_l0_fail"], 0.5)


class TestObservationBuffer(unittest.TestCase):
    def test_observe_and_retrieve(self):
        buf = ObservationBuffer()
        buf.observe(ObservationType.REQUEST_RECEIVED, data={'type': 'code'})
        buf.observe(ObservationType.VERIFICATION_RESULT, expert_id="e1", data={'passed': True})
        
        self.assertEqual(len(buf.buffer), 2)
        self.assertEqual(len(buf.for_expert("e1")), 1)
    
    def test_success_rate(self):
        buf = ObservationBuffer()
        for i in range(10):
            buf.observe(ObservationType.VERIFICATION_RESULT, "e1", 
                       passed=(i % 2 == 0))
        
        rate = buf.success_rate("e1")
        self.assertAlmostEqual(rate, 0.5, places=1)
    
    def test_buffer_trim(self):
        buf = ObservationBuffer(max_size=5)
        for i in range(10):
            buf.observe(ObservationType.REQUEST_RECEIVED)
        self.assertEqual(len(buf.buffer), 5)


class TestEvolutionCycle(unittest.TestCase):
    def setUp(self):
        caps = NodeCapabilities(
            level=NodeLevel.L1_LIGHT_INFERENCE,
            architectures=[ModelArchitecture.RWKV],
            max_memory_mb=8192,
            compute_flops=50e9,
            bandwidth_mbps=100,
            storage_gb=100,
        )
        self.node = Node(NodeIdentity(capabilities=caps))
        self.expert = ExpertModel(
            expert_id="test_expert",
            base_model="rwkv-4-169m",
            architecture=ModelArchitecture.RWKV,
            domain="general",
            param_count_mb=646,
        )
        self.node.add_expert(self.expert)
        self.cycle = EvolutionCycle(self.node)
    
    def test_observe_request(self):
        self.cycle.observe_request(InferenceRequest(
            request_type=RequestType.CODE_GENERATION,
            prompt="test"
        ))
        self.assertEqual(len(self.cycle.buffer.buffer), 1)
    
    def test_reflect_on_failures(self):
        # Simulate failures
        for _ in range(10):
            self.cycle.observe_verification("test_expert", passed=False, error_type="syntax_error")
        
        reflections = self.cycle.reflector.reflect("test_expert")
        self.assertTrue(len(reflections) > 0)
        # Should detect low success rate
        self.assertTrue(any(r.action == "fine_tune" for r in reflections))
    
    def test_full_cycle(self):
        # Simulate observations
        for _ in range(20):
            self.cycle.observe_request(InferenceRequest(request_type=RequestType.CODE_GENERATION))
            self.cycle.observe_verification("test_expert", passed=False)
        
        plan = self.cycle.run_cycle()
        if plan:
            self.assertIn(plan.action, [EvolutionAction.LORA_FINE_TUNE, 
                                        EvolutionAction.TARGETED_TRAINING])


class TestNetworkNode(unittest.TestCase):
    def test_announce_self(self):
        caps = NodeCapabilities(
            level=NodeLevel.L1_LIGHT_INFERENCE,
            architectures=[ModelArchitecture.RWKV],
            max_memory_mb=8192,
            compute_flops=50e9,
            bandwidth_mbps=100,
            storage_gb=100,
        )
        node = NetworkNode(NodeIdentity(capabilities=caps))
        node.add_expert(ExpertModel(
            base_model="rwkv-4-169m",
            architecture=ModelArchitecture.RWKV,
            domain="general",
        ))
        
        msg = node.announce_self()
        self.assertEqual(msg.msg_type, MessageType.NODE_ANNOUNCE)
        self.assertIn('experts', msg.payload)
    
    def test_handle_ping(self):
        node = NetworkNode(NodeIdentity(capabilities=NodeCapabilities(
            level=NodeLevel.L0_COLLECTOR,
            architectures=[],
            max_memory_mb=2048,
            compute_flops=10e9,
            bandwidth_mbps=50,
            storage_gb=50,
        )))
        
        ping = Message(msg_type=MessageType.PING, sender_id="other_node")
        pong = node.handle_message(ping)
        self.assertIsNotNone(pong)
        self.assertEqual(pong.msg_type, MessageType.PONG)
    
    def test_peer_discovery(self):
        discovery = PeerDiscovery("local_node")
        discovery.learn_about_peer("peer_1", {
            'level': 'L2_STANDARD_INFERENCE',
            'experts': [{'domain': 'code', 'architecture': 'transformer'}],
            'latency_ms': 50,
        })
        
        self.assertEqual(len(discovery.known_peers), 1)
        experts = discovery.find_experts(domain="code")
        self.assertEqual(len(experts), 1)
    
    def test_message_serialization(self):
        msg = Message(
            msg_type=MessageType.NODE_ANNOUNCE,
            sender_id="node_1",
            payload={'level': 'L1'},
        )
        json_str = msg.to_json()
        restored = Message.from_json(json_str)
        self.assertEqual(restored.msg_type, MessageType.NODE_ANNOUNCE)
        self.assertEqual(restored.sender_id, "node_1")


class TestWebSocketTransport(unittest.TestCase):
    """Test WebSocket P2P transport"""
    
    def test_message_round_trip(self):
        """Test message serialization for WebSocket transport"""
        msg = Message(
            msg_type=MessageType.NODE_ANNOUNCE,
            sender_id="node_a",
            payload={'level': 'L1', 'experts': []},
        )
        json_str = msg.to_json()
        restored = Message.from_json(json_str)
        self.assertEqual(restored.msg_type, MessageType.NODE_ANNOUNCE)
        self.assertEqual(restored.sender_id, "node_a")
        self.assertEqual(restored.payload['level'], 'L1')
    
    def test_connection_manager(self):
        from ws_transport import ConnectionManager, PeerConnection
        
        cm = ConnectionManager("local")
        self.assertEqual(cm.count(), 0)
    
    def test_peer_connection_stats(self):
        from ws_transport import ConnectionManager, PeerConnection
        
        cm = ConnectionManager("local")
        conn = PeerConnection(ws=None, peer_id="peer_1", address="ws://localhost:9001")
        conn.messages_sent = 5
        conn.messages_received = 3
        cm.add(conn)
        
        status = cm.get_status()
        self.assertEqual(status['connected_peers'], 1)
        self.assertEqual(status['peers'][0]['sent'], 5)


class TestConfigLoader(unittest.TestCase):
    """Test configuration system"""
    
    def test_default_config(self):
        from config_loader import ConfigLoader
        config = ConfigLoader.load("nonexistent.yaml")
        self.assertEqual(config.node.level, "L1")
        self.assertEqual(config.network.listen, "0.0.0.0:8001")
    
    def test_load_example_config(self):
        from config_loader import ConfigLoader
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.example.yaml')
        if os.path.exists(config_path):
            config = ConfigLoader.load(config_path)
            self.assertEqual(config.node.level, "L1")
            self.assertTrue(len(config.experts) >= 1)
            self.assertEqual(config.evolution.lora.r, 8)
    
    def test_env_override(self):
        from config_loader import ConfigLoader
        os.environ['DECENTRAL_AI_LEVEL'] = 'L3'
        data = ConfigLoader._merge_env({'node': {'level': 'L1'}})
        self.assertEqual(data['node']['level'], 'L3')
        del os.environ['DECENTRAL_AI_LEVEL']
    
    def test_validate_config(self):
        from config_loader import ConfigLoader, DecentralAIConfig, NodeConfig
        config = DecentralAIConfig(node=NodeConfig(level="L9"))
        warnings = ConfigLoader.validate(config)
        self.assertTrue(any("Invalid" in w for w in warnings))
    
    def test_validate_good_config(self):
        from config_loader import ConfigLoader, DecentralAIConfig
        config = DecentralAIConfig()
        warnings = ConfigLoader.validate(config)
        # Default config should have no critical warnings
        self.assertFalse(any("Invalid" in w for w in warnings))


class TestFiveLevelSystem(unittest.TestCase):
    """Test the five-level node hierarchy"""
    
    def test_all_levels_exist(self):
        self.assertEqual(len(NodeLevel), 5)
    
    def test_credit_rates(self):
        ledger = CreditLedger()
        self.assertEqual(ledger.CREDIT_RATES[NodeLevel.L0_COLLECTOR], 1)
        self.assertEqual(ledger.CREDIT_RATES[NodeLevel.L4_DATA_CENTER], 200)
    
    def test_level_progression(self):
        """Verify level values are ordered"""
        for i in range(4):
            self.assertLess(NodeLevel(i).value, NodeLevel(i+1).value)


if __name__ == "__main__":
    print("=" * 60)
    print("DecentralAI Unit Tests")
    print("=" * 60)
    unittest.main(verbosity=2)
