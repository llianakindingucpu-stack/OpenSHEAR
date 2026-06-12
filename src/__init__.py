"""
DecentralAI - Born Distributed AI Network
==========================================

A decentralized Mixture-of-Experts network where every node
contributes according to its capability, earns according to
its contribution, and evolves through continuous learning.

"Not everyone is equal, but everyone has a path."

Core Modules:
- core: Node, Expert, Router, Verifier, CreditLedger
- evolution: Observation, Reflection, Evolution, Verification cycle
- network: P2P messaging, Peer discovery, Request forwarding
"""

__version__ = "0.1.0"
__author__ = "DecentralAI Team"

from core import (
    Node, NodeIdentity, NodeLevel, NodeCapabilities,
    ExpertModel, ModelArchitecture,
    InferenceRequest, InferenceResponse, RequestType,
    Router, Verifier, CreditLedger,
)

from evolution import (
    EvolutionCycle, ObservationBuffer, Reflector, Evolver, EvolutionVerifier,
)

from network import (
    NetworkNode, PeerDiscovery, Message, MessageType,
)
