"""
DecentralAI Solidity Contract Deployment Script
================================================
Deploys all contracts to a local Hardhat node.

Usage:
    1. Start Hardhat node: npx hardhat node
    2. Run this: python deploy_contracts.py
    3. Or use Foundry: forge create --rpc-url=http://localhost:8545 ...

Note: For actual deployment, use Hardhat or Foundry.
This script generates the constructor arguments and deployment order.
"""

import json
import hashlib

def generate_address():
    """Generate a deterministic address for demo"""
    return "0x" + hashlib.sha256(b"decentral-ai").hexdigest()[:40]

def main():
    print("=" * 60)
    print("DecentralAI Contract Deployment Guide")
    print("=" * 60)
    
    contracts = [
        ("1. DecentralAICredits",    "DCRED",     "ERC20 + Staking"),
        ("2. DecentralAIReputation", "DREP",      "Reputation + Slashing"),
        ("3. DecentralAISettlement", "DSET",      "Inference Settlement"),
        ("4. DecentralAIGovernance", "DGVN",      "On-chain Governance"),
    ]
    
    print("\nDeployment Order:")
    print("-" * 60)
    for name, symbol, desc in contracts:
        print(f"  {name:28s} ({symbol}) — {desc}")
    
    print("\nDeployment Commands (Hardhat):")
    print("-" * 60)
    print("npx hardhat compile")
    print()
    print("npx hardhat run --network localhost scripts/deploy.js")
    print()
    print("# Or use Foundry for faster deployment:")
    print("forge build")
    print("forge create src/DecentralAI.sol:DecentralAICredits")
    print("forge create src/DecentralAI.sol:DecentralAIReputation")
    
    print("\nExpected Gas (Hardhat estimate):")
    print("-" * 60)
    print(f"  DecentralAICredits:    ~1.2M gas  (ERC20 + Staking)")
    print(f"  DecentralAIReputation: ~0.8M gas  (Mappings + Events)")
    print(f"  DecentralAISettlement: ~1.5M gas  (Job lifecycle)")
    print(f"  DecentralAIGovernance: ~0.6M gas  (Proposals + Voting)")
    print(f"  TOTAL:                 ~4.1M gas")
    
    # Estimate cost at $10/gas (Scroll/ZKSync prices)
    eth_usd = 2000
    print(f"\nEstimated Cost at $10/gas × $2000/ETH:")
    print("-" * 60)
    total = 4.1e6 * 10 / 1e18 * eth_usd
    print(f"  ~${total:.2f} for all 4 contracts")
    print(f"  (Much cheaper than Ethereum mainnet: ~$500-1000)")
    
    print("\nContract Addresses (Hardhat localhost):")
    print("-" * 60)
    print(f"  DecentralAICredits:    0x... (deploy first, needed by others)")
    print(f"  DecentralAIReputation: 0x... (deploy second)")
    print(f"  DecentralAISettlement: 0x... (pass addresses of 1+2 as constructor args)")
    print(f"  DecentralAIGovernance: 0x... (pass DecentralAICredits address)")
    
    print("\nPost-Deployment Setup:")
    print("-" * 60)
    print("  1. Mint initial DCRED supply to treasury")
    print("  2. Register initial nodes via DecentralAIReputation.register()")
    print("  3. Configure Settlement contract with credit + reputation addresses")
    print("  4. Transfer DCRED to test accounts for demo")
    
    print("\nHardhat JS Deployment Script (scripts/deploy.js):")
    print("-" * 60)
    script = '''
// scripts/deploy.js
const hre = require("hardhat");

async function main() {
  // 1. Deploy Credits
  const Credits = await hre.ethers.getContractFactory("DecentralAICredits");
  const credits = await Credits.deploy();
  await credits.waitForDeployment();
  console.log("DecentralAICredits:", await credits.getAddress());

  // 2. Deploy Reputation
  const Reputation = await hre.ethers.getContractFactory("DecentralAIReputation");
  const reputation = await Reputation.deploy();
  await reputation.waitForDeployment();
  console.log("DecentralAIReputation:", await reputation.getAddress());

  // 3. Deploy Settlement
  const Settlement = await hre.ethers.getContractFactory("DecentralAISettlement");
  const settlement = await Settlement.deploy(await credits.getAddress(), await reputation.getAddress());
  await settlement.waitForDeployment();
  console.log("DecentralAISettlement:", await settlement.getAddress());

  // 4. Deploy Governance
  const Governance = await hre.ethers.getContractFactory("DecentralAIGovernance");
  const governance = await Governance.deploy(await credits.getAddress());
  await governance.waitForDeployment();
  console.log("DecentralAIGovernance:", await governance.getAddress());
}

main().catch(console.error);
'''
    print(script)
    
    print("\nAPI Integration:")
    print("-" * 60)
    print("  # After deployment, update config.yaml:")
    print('  contracts:')
    print('    credits: "0x..."')
    print('    reputation: "0x..."')
    print('    settlement: "0x..."')
    print('    governance: "0x..."')
    print('    rpc_url: "http://localhost:8545"')
    print('    private_key: "0x..."  # Node operator key')


if __name__ == "__main__":
    main()
