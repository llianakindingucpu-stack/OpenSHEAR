# Node Hierarchy

SHEAR's network supports **heterogeneous hardware** — from a $50 used PC to a $30,000 GPU server. Every node contributes, and every contribution is rewarded.

## Philosophy

> "Not everyone equal, but everyone has a path."

Traditional distributed AI requires uniform, expensive hardware. SHEAR embraces diversity: a student with a laptop can run an L0 node alongside a research lab's A100 cluster running L4.

## Five Levels

```
┌─────────────────────────────────────────────────────┐
│ L4 — Datacenter (A100/H100, 70B+)                    │
│ Full training · Dataset curation · Large-scale inf.   │
├─────────────────────────────────────────────────────┤
│ L3 — Heavy (3090/4090, 14B+)                         │
│ Heavy inference · LoRA fine-tuning · Verification     │
├─────────────────────────────────────────────────────┤
│ L2 — Standard (3060, 7B)                             │
│ Standard inference · Draft verification · Evaluation  │
├─────────────────────────────────────────────────────┤
│ L1 — Lightweight (CPU + 4GB RAM, 0.5B-1.5B)          │
│ Draft generation · Lightweight inference · Routing     │
├─────────────────────────────────────────────────────┤
│ L0 — Collector (CPU only, no model)                  │
│ Data collection · Task routing · Network relay        │
└─────────────────────────────────────────────────────┘
```

## Level Details

### L0 — Collector

| Property | Value |
|----------|-------|
| Hardware | Any CPU, 1GB+ RAM |
| Model | None |
| Role | Data collection, task routing, network relay |
| Credits earned | Low (per-task relay) |
| Example | Old laptop, Raspberry Pi, VPS |

L0 nodes don't run any model. They contribute by:
- Collecting and preprocessing data from the web/APIs
- Routing inference requests to appropriate nodes
- Acting as network relays for P2P communication
- Tracking node availability and health

### L1 — Lightweight

| Property | Value |
|----------|-------|
| Hardware | CPU + 4GB RAM |
| Model | 0.5B–1.5B (RWKV-4, quantized) |
| Role | Draft generation, lightweight inference |
| Credits earned | Medium (per-token draft) |
| Example | Desktop PC, cloud VM |

L1 nodes are the **drafters** in speculative decoding. They run small models fast on CPU:
- Generate draft tokens for L2/L3 verification
- Handle simple queries directly (chat, basic Q&A)
- Provide "quick and cheap" inference

### L2 — Standard

| Property | Value |
|----------|-------|
| Hardware | GPU with 8GB+ VRAM (RTX 3060) |
| Model | 7B (RWKV-6/7, quantized) |
| Role | Standard inference, draft verification |
| Credits earned | High (per-token verified) |
| Example | Gaming PC with mid-range GPU |

L2 nodes are the **workhorses** of the network:
- Verify draft tokens from L1 nodes
- Handle standard-quality inference requests
- Run evaluation benchmarks

### L3 — Heavy

| Property | Value |
|----------|-------|
| Hardware | GPU with 24GB+ VRAM (RTX 3090/4090) |
| Model | 14B+ (RWKV-6/7) |
| Role | Heavy inference, LoRA fine-tuning, verification |
| Credits earned | Very high (per-token + training) |
| Example | Workstation with high-end GPU |

L3 nodes provide **premium quality**:
- High-quality inference for complex tasks
- LoRA fine-tuning on domain data
- Final verification for critical requests

### L4 — Datacenter

| Property | Value |
|----------|-------|
| Hardware | GPU cluster (A100/H100) |
| Model | 70B+ |
| Role | Full training, dataset curation, research |
| Credits earned | Highest (per-epoch + dataset) |
| Example | Cloud GPU rental, research lab |

L4 nodes are the **trainers**:
- Pre-train new model versions
- Curate and validate training datasets
- Push trained weights to the network

## Upgrade Path

Nodes can upgrade at any time:

```
L0 ──(add model)──► L1 ──(add GPU)──► L2 ──(better GPU)──► L3 ──(cluster)──► L4
```

Each upgrade increases earning potential and network contribution.

## Speculative Decoding Flow

The node hierarchy naturally maps to speculative decoding:

```
User Request
     │
     ▼
┌─ L0 Router ──────────────────────────────────────┐
│  Analyze request complexity, route to nodes        │
└────────────────────┬──────────────────────────────┘
                     │
         ┌───────────▼────────────┐
         │  L1 Drafters (3 cells) │
         │  T=0.5  T=0.8  T=1.1  │
         │  ~10 tok/s each        │
         └───────────┬────────────┘
                     │ draft tokens
         ┌───────────▼────────────┐
         │  L2/L3 Verifier        │
         │  7B+ model             │
         │  ~5 tok/s but accurate │
         └───────────┬────────────┘
                     │ verified output
                     ▼
               User Response
```

## Credit Flow

```
Requester pays Credits
         │
         ├──► L1 Drafters: 30% (per draft token)
         ├──► L2/L3 Verifier: 50% (per verified token)
         ├──► L0 Router: 10% (per request)
         └──► Network treasury: 10% (for infrastructure)
```

Higher-level nodes earn more per contribution, but their hardware costs more. The system self-balances: if there are too many L1 drafters, their per-token rate drops, incentivizing some to upgrade to L2.
