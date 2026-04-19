# SHEAR Tokenomics

**Version**: 1.0 | **Date**: 2026-04-19 | **Status**: Design — Not yet deployed

---

## TL;DR

SHEAR runs a dual-token system:

| Token | Symbol | Role | Price |
|-------|--------|------|-------|
| Stablecoin | SHEAR-USD | Payment for inference | 1:1 USD |
| Utility Token | $SHEAR | Staking + Governance + Rewards | Floating |

**SHEAR-USD**: Users pay for inference. Nodes receive stablecoin income.
**$SHEAR**: Nodes stake to participate. Holders vote on protocol governance.
**Credits**: Internal accounting unit. 1 Credit = $0.0001 USD. Users buy Credits with SHEAR-USD.

---

## 1. Dual-Token Architecture

### Why Two Tokens?

A single token creates a conflict: if used for payments, volatile price makes it unusable for commerce; if stable, it cannot appreciate and has no investment incentive.

**Solution: separate the mediums.**

```
                    SHEAR Protocol
                         |
     +-------------------+-------------------+
     |                   |                   |
SHEAR-USD           Credits             $SHEAR
(Stablecoin)      (Accounting)          (Token)
     |                   |                   |
Payment for        Deducted from        Stake to run
inference          user balance         a node
     |                   |                   |
     v                   v                   v
Node income        Settlement            Governance
in real money     happens here         + inflation rewards
```

### SHEAR-USD (Stablecoin)

- **Peg**: 1 SHEAR-USD = 1 USD
- **Backing**: 100% treasury reserves (USDC strategy: USD + short-term T-bills)
- **Mint/Burn**: Users deposit USD to mint SHEAR-USD. Withdraw USD to burn SHEAR-USD.
- **Custodian**: Multi-sig treasury controlled by L4 governance nodes
- **No algorithmic peg**: No depeg risk. Direct redemption always available.

### $SHEAR (Utility Token)

- **Total Supply**: 1,000,000,000 (1 billion), fixed at genesis
- **Price**: Floating market price on DEX (after launch)
- **Utility**:
  1. **Staking** - stake to become a node, earn income + inflation rewards
  2. **Governance** - 1 $SHEAR = 1 vote on protocol decisions
  3. **Fee Discount** - stake $SHEAR to get up to 50% discount on inference fees
  4. **Security Bond** - slashable stake for misbehavior

---

## 2. Credits — Internal Accounting Unit

Credits abstract away token volatility and fiat conversion for users.

```
1 Credit = $0.0001 USD (fixed)
```

### Why This Rate?

- Inference for 1M tokens at market rate = ~$0.10-1.00
- 1 Credit = $0.0001 keeps numbers human-readable
- Rate adjustable by governance if inference economics change dramatically

### Credit Pricing (Reference)

| Action | Cost in Credits | Cost in USD |
|--------|-----------------|-------------|
| 1K token inference (L1) | 1,000 | $0.10 |
| 1K token inference (L2) | 3,000 | $0.30 |
| 1K token inference (L3) | 10,000 | $1.00 |
| 1M token inference (L2) | 3,000,000 | $300 |

---

## 3. Node Staking Requirements

Nodes must stake $SHEAR to participate. Higher tiers require more stake but earn more.

| Tier | Role | Hardware | $SHEAR Stake | Income Multiplier |
|------|------|----------|-------------|-------------------|
| L0 | Collector | CPU | 0 (none) | N/A |
| L1 | Light Inference | CPU+4GB | 10,000 | 1x |
| L2 | Standard | 3060 | 100,000 | 3x |
| L3 | Heavy | 3090/4090 | 1,000,000 | 10x |
| L4 | Datacenter | A100/H100 | 10,000,000 | 50x |

### Staking Mechanics

```
Stake Amount x Current $SHEAR Price = USD Value Locked

Example (L2, $SHEAR = $0.10):
100,000 x $0.10 = $10,000 USD equivalent locked

Annual node income (L2): ~$3,000-$10,000 (varies with network usage)
APY from staking: 30-100% (subsidized by inflation early on)
```

### Stake Rules

- **Cooldown**: 7 days after unstaking before funds are released
- **Slashing**: -1% of stake per infraction (false results, downtime)
- **Jail**: Repeated violations - node temporarily barred from earning

---

## 4. Revenue Model

### User Payment Flow

```
User -> Pays SHEAR-USD -> Converted to Credits -> Deducted per request
                              |
                     Treasury Pool (smart contract)
                              |
         +--------------------+--------------------+
         |                    |                    |
    95% to Nodes        3% to Foundation      2% to Insurance Pool
    (proportional       (development,         (slashing reserve,
    to work done)       operations, grants)    emergency fund)
```

### Node Income

```
Node Income = Work Payment + Inflation Reward

Work Payment:
  SHEAR-USD from treasury
  Amount = (node verified work / total network work) x 95% of treasury

Inflation Reward:
  $SHEAR minted per block
  Distributed proportionally to staked nodes
  Rate: see Section 6 (Inflation Schedule)
```

### Settlement Cycle

- **Work Payment**: Settled per request
- **Inflation Rewards**: Distributed weekly
- **Foundation Cut**: Auto-deducted at treasury level

---

## 5. $SHEAR Initial Allocation

### Genesis Allocation

| Category | Amount | % | Vesting |
|----------|--------|---|---------|
| Community & Miners | 600,000,000 | 60% | No lock. Staking rewards over 4 years |
| Foundation Reserve | 150,000,000 | 15% | 4-year linear vesting. Grants, partnerships, liquidity |
| Team | 120,000,000 | 12% | 1-year cliff, then 3-year linear vesting |
| Early Backers | 80,000,000 | 8% | 1-year cliff, then 2-year linear vesting |
| Airdrop & Community | 50,000,000 | 5% | 10% at TGE, 90% over 1 year |

**Total**: 1,000,000,000 $SHEAR - no minting beyond genesis except inflation

### Allocation Rationale

- **60% to community/miners**: Ensures decentralization. No single entity dominates.
- **15% to foundation**: Sufficient for 3-5 years of operations without selling tokens
- **12% to team**: Aligned incentives (vesting = long-term commitment)
- **No ICO/IEO**: Protocol grows organically through staking rewards, not token sales

---

## 6. Inflation Schedule

### Emission Model

```
Year 1:  5% (= 50,000,000 $SHEAR)
Year 2:  4% (= 40,000,000)
Year 3:  3% (= 30,000,000)
Year 4:  2% (= 20,000,000)
Year 5+: 1% (= 10,000,000) - floor rate
```

**Total over 5 years**: ~140M $SHEAR emitted

### Distribution of Inflation

| Recipient | Share | Purpose |
|-----------|-------|---------|
| Staked Nodes (L1-L4) | 80% | Security + participation reward |
| Foundation | 15% | Operations, grants, development |
| Security Council | 5% | Emergency response, bug bounties |

### Deflationary Mechanisms

| Burn Sink | Amount | Trigger |
|-----------|--------|---------|
| Governance participation | 1 $SHEAR per vote | Cast a governance vote |
| Slashed stake | 50% of slashed amount | Node misbehavior |
| Protocol fee burn | 10% of inference fees | Paid in $SHEAR (optional) |

---

## 7. Governance

### Voting Rights

```
1 $SHEAR staked = 1 vote
```

### Governance Scope

**Can vote on:**
- Adjusting Credit-to-USD rate (within +/-20% band)
- Changing tier staking thresholds
- Distributing foundation treasury funds
- Upgrading protocol logic
- Adding/removing node tiers
- Emergency parameter changes

**Cannot vote on:**
- Total supply changes (fixed at genesis)
- Individual node rewards (smart contract determined)
- Censorship of specific users or content

### Voting Mechanism

- **Proposal threshold**: 1% of circulating supply must vote
- **Pass threshold**: >50% of votes in favor
- **Quorum**: >10% of circulating supply participating
- **Execution**: Approved proposals execute automatically via smart contract

### Security Council

- 5 members elected by $SHEAR holders (top 5 by stake weight)
- Handles: emergency pauses, critical bug fixes, fork decisions
- Cannot unilaterally change tokenomics or steal funds
- All actions are public and reversible by governance vote

---

## 8. Anti-Sybil & Anti-Fraud

### Sybil Attack Prevention

**Problem**: Attacker creates thousands of fake nodes to dominate the network.

**Solution**:
1. **Economic stake barrier**: Each node tier requires real $SHEAR stake (expensive to fake)
2. **Identity verification (optional)**: L3/L4 nodes may require KYC to prevent coordinated attacks
3. **Geographic distribution bonus**: Nodes in underrepresented regions get +10% rewards

### Fraud Prevention

**Problem**: Nodes submit false inference results to earn rewards without doing work.

**Solution - Three-Layer Verification**:
1. **Requestor annotation**: Users rate results (upvote/downvote - adjusts node reputation)
2. **Redundant consensus**: Same request sent to 3 nodes - results must match
3. **Reputation-weighted voting**: High-reputation nodes' answers count more

**Economic deterrent**: Slashed stake (1-10%) for detected fraud.

---

## 9. Roadmap to Launch

```
Phase 1 - Design        [Now]      Tokenomics finalized, docs published
Phase 2 - Testnet       [3 months] L0/L1 nodes only, fake traffic, stress test
Phase 3 - Staking       [6 months] $SHEAR staking enabled, L2/L3 nodes online
Phase 4 - Mainnet       [9 months] Full dual-token system, DEX listing
Phase 5 - Decentralized  [12 months] Full governance, no admin keys
```

---

*This is a design document. Token has not been deployed. Subject to governance approval.*
