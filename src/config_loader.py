"""
DecentralAI Configuration System
===================================
Load and validate node configuration from YAML files.

Usage:
    from config import NodeConfig
    config = NodeConfig.load("config.yaml")
    print(config.node.level)
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

sys.path.insert(0, r'D:\pylib')

try:
    import yaml
except ImportError:
    yaml = None


# ============================================================
# 1. Config Data Classes
# ============================================================

@dataclass
class NodeConfig:
    level: str = "L1"
    node_id: str = ""
    name: str = ""


@dataclass
class NetworkConfig:
    listen: str = "0.0.0.0:8001"
    api_listen: str = "0.0.0.0:8000"
    bootstrap_peers: List[str] = field(default_factory=list)
    max_peers: int = 50
    gossip_interval: int = 60


@dataclass
class ExpertConfig:
    model_path: str = ""
    architecture: str = "rwkv"
    domain: str = "general"
    strategy: str = "cpu fp32"


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    learning_rate: float = 0.0001
    epochs: int = 3
    batch_size: int = 4


@dataclass
class VerificationConfig:
    min_improvement: float = 0.05
    max_regression: float = 0.1
    benchmark_samples: int = 20


@dataclass
class EvolutionConfig:
    auto_evolve: bool = True
    min_observations: int = 20
    check_interval: int = 300
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)


@dataclass
class CreditsConfig:
    initial_credits: float = 100
    min_balance: float = 0
    quality_multiplier: float = 1.0


@dataclass
class LimitsConfig:
    max_inference_time: int = 30
    max_tokens: int = 2048
    max_concurrent: int = 5
    max_memory_mb: int = 8192


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = ""
    max_size: int = 10
    backup_count: int = 3


@dataclass
class AdvancedConfig:
    seed: int = 0
    cache_dir: str = "./cache"
    data_dir: str = "./data"
    results_dir: str = "./results"


@dataclass
class DecentralAIConfig:
    """Top-level configuration"""
    node: NodeConfig = field(default_factory=NodeConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    experts: List[ExpertConfig] = field(default_factory=list)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    credits: CreditsConfig = field(default_factory=CreditsConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)


# ============================================================
# 2. Config Loader
# ============================================================

class ConfigLoader:
    """
    Load configuration from YAML file with validation.
    
    Priority:
    1. YAML file
    2. Environment variables (DECENTRAL_AI_ prefix)
    3. Defaults
    """
    
    @staticmethod
    def load(path: str = "config.yaml") -> DecentralAIConfig:
        """Load config from YAML file"""
        data = {}
        
        if os.path.exists(path):
            if yaml is None:
                print(f"Warning: pyyaml not installed, using defaults")
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
        
        # Merge with environment variables
        data = ConfigLoader._merge_env(data)
        
        return ConfigLoader._parse(data)
    
    @staticmethod
    def _merge_env(data: Dict) -> Dict:
        """Merge environment variables into config"""
        env_map = {
            'DECENTRAL_AI_LEVEL': ('node', 'level'),
            'DECENTRAL_AI_LISTEN': ('network', 'listen'),
            'DECENTRAL_AI_API_LISTEN': ('network', 'api_listen'),
            'DECENTRAL_AI_BOOTSTRAP': ('network', 'bootstrap_peers'),
        }
        
        for env_key, (section, key) in env_map.items():
            val = os.environ.get(env_key)
            if val:
                if section not in data:
                    data[section] = {}
                if key == 'bootstrap_peers':
                    data[section][key] = val.split(',')
                else:
                    data[section][key] = val
        
        return data
    
    @staticmethod
    def _parse(data: Dict) -> DecentralAIConfig:
        """Parse raw dict into typed config"""
        config = DecentralAIConfig()
        
        # Node
        if 'node' in data:
            n = data['node']
            config.node = NodeConfig(
                level=n.get('level', 'L1'),
                node_id=n.get('node_id', ''),
                name=n.get('name', ''),
            )
        
        # Network
        if 'network' in data:
            n = data['network']
            config.network = NetworkConfig(
                listen=n.get('listen', '0.0.0.0:8001'),
                api_listen=n.get('api_listen', '0.0.0.0:8000'),
                bootstrap_peers=n.get('bootstrap_peers', []),
                max_peers=n.get('max_peers', 50),
                gossip_interval=n.get('gossip_interval', 60),
            )
        
        # Experts
        if 'experts' in data:
            config.experts = []
            for e in data['experts']:
                config.experts.append(ExpertConfig(
                    model_path=e.get('model_path', ''),
                    architecture=e.get('architecture', 'rwkv'),
                    domain=e.get('domain', 'general'),
                    strategy=e.get('strategy', 'cpu fp32'),
                ))
        
        # Evolution
        if 'evolution' in data:
            ev = data['evolution']
            lora_data = ev.get('lora', {})
            ver_data = ev.get('verification', {})
            config.evolution = EvolutionConfig(
                auto_evolve=ev.get('auto_evolve', True),
                min_observations=ev.get('min_observations', 20),
                check_interval=ev.get('check_interval', 300),
                lora=LoRAConfig(
                    r=lora_data.get('r', 8),
                    alpha=lora_data.get('alpha', 16),
                    dropout=lora_data.get('dropout', 0.05),
                    learning_rate=lora_data.get('learning_rate', 0.0001),
                    epochs=lora_data.get('epochs', 3),
                    batch_size=lora_data.get('batch_size', 4),
                ),
                verification=VerificationConfig(
                    min_improvement=ver_data.get('min_improvement', 0.05),
                    max_regression=ver_data.get('max_regression', 0.1),
                    benchmark_samples=ver_data.get('benchmark_samples', 20),
                ),
            )
        
        # Credits
        if 'credits' in data:
            c = data['credits']
            config.credits = CreditsConfig(
                initial_credits=c.get('initial_credits', 100),
                min_balance=c.get('min_balance', 0),
                quality_multiplier=c.get('quality_multiplier', 1.0),
            )
        
        # Limits
        if 'limits' in data:
            l = data['limits']
            config.limits = LimitsConfig(
                max_inference_time=l.get('max_inference_time', 30),
                max_tokens=l.get('max_tokens', 2048),
                max_concurrent=l.get('max_concurrent', 5),
                max_memory_mb=l.get('max_memory_mb', 8192),
            )
        
        # Logging
        if 'logging' in data:
            lg = data['logging']
            config.logging = LoggingConfig(
                level=lg.get('level', 'INFO'),
                file=lg.get('file', ''),
                max_size=lg.get('max_size', 10),
                backup_count=lg.get('backup_count', 3),
            )
        
        # Advanced
        if 'advanced' in data:
            a = data['advanced']
            config.advanced = AdvancedConfig(
                seed=a.get('seed', 0),
                cache_dir=a.get('cache_dir', './cache'),
                data_dir=a.get('data_dir', './data'),
                results_dir=a.get('results_dir', './results'),
            )
        
        return config
    
    @staticmethod
    def validate(config: DecentralAIConfig) -> List[str]:
        """Validate config, return list of warnings"""
        warnings = []
        
        # Check level
        valid_levels = ['L0', 'L1', 'L2', 'L3', 'L4']
        if config.node.level not in valid_levels:
            warnings.append(f"Invalid node level: {config.node.level}")
        
        # Check experts exist for L1+
        if config.node.level != 'L0' and not config.experts:
            warnings.append("L1+ node requires at least one expert model")
        
        # Check model paths
        for expert in config.experts:
            if expert.model_path and not os.path.exists(expert.model_path):
                warnings.append(f"Model not found: {expert.model_path}")
        
        # Check LoRA params
        if config.evolution.lora.r <= 0:
            warnings.append("LoRA r must be positive")
        if config.evolution.lora.learning_rate <= 0:
            warnings.append("Learning rate must be positive")
        
        return warnings


# ============================================================
# 3. Demo
# ============================================================

def demo():
    """Demonstrate configuration loading"""
    print("=" * 60)
    print("DecentralAI Configuration System - Demo")
    print("=" * 60)
    
    # Load from example config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.example.yaml')
    config = ConfigLoader.load(config_path)
    
    print(f"\n[1] Loaded config from: {config_path}")
    print(f"    Node level: {config.node.level}")
    print(f"    Network listen: {config.network.listen}")
    print(f"    Experts: {len(config.experts)}")
    for e in config.experts:
        print(f"      - {e.architecture}/{e.domain}: {e.model_path}")
    print(f"    Evolution auto: {config.evolution.auto_evolve}")
    print(f"    LoRA r={config.evolution.lora.r}, alpha={config.evolution.lora.alpha}")
    print(f"    Credits initial: {config.credits.initial_credits}")
    
    # Validate
    warnings = ConfigLoader.validate(config)
    if warnings:
        print(f"\n[2] Warnings:")
        for w in warnings:
            print(f"    - {w}")
    else:
        print(f"\n[2] Validation: OK")
    
    # Load with defaults (no file)
    default_config = ConfigLoader.load("nonexistent.yaml")
    print(f"\n[3] Default config:")
    print(f"    Level: {default_config.node.level}")
    print(f"    Listen: {default_config.network.listen}")
    print(f"    Experts: {len(default_config.experts)}")
    
    # Environment variable override
    print(f"\n[4] Env var override test:")
    os.environ['DECENTRAL_AI_LEVEL'] = 'L3'
    env_config = ConfigLoader._merge_env({'node': {'level': 'L1'}})
    print(f"    DECENTRAL_AI_LEVEL=L3 -> node.level = {env_config['node']['level']}")
    del os.environ['DECENTRAL_AI_LEVEL']
    
    print("\n--- Configuration is Flexible ---")
    print("YAML file → Environment variables → Defaults")
    print("Override anything, break nothing.")


if __name__ == "__main__":
    demo()
