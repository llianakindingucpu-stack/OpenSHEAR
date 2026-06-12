// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * DecentralAI Credit System — On-Chain Settlement
 * ================================================
 * 
 * Tracks node credits, reputation, and inference settlements.
 * Designed for low gas: bulk operations, minimal storage.
 * 
 * Contracts:
 *   DecentralAICredits   — credit ledger + transfers
 *   DecentralAIReputation — reputation scores + slashing
 *   DecentralAISettlement — inference payment settlement
 *   DecentralAIGovernance — parameter updates via voting
 */

// ============================================================
// 1. Credit Ledger
// ============================================================

contract DecentralAICredits {
    // === State ===
    string public constant name = "DecentralAI Credit";
    string public constant symbol = "DCRED";
    uint8 public constant decimals = 18;
    uint256 public totalSupply;
    
    address public owner;
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;
    
    // Node level multipliers (credits per inference)
    // L0=1, L1=5, L2=20, L3=80, L4=200
    mapping(uint8 => uint256) public levelRates;
    
    // Staking: nodes stake credits to serve
    mapping(address => uint256) public staked;
    uint256 public minStake = 100 * 1e18; // 100 DCRED minimum
    
    // === Events ===
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Stake(address indexed node, uint256 amount);
    event Unstake(address indexed node, uint256 amount);
    event Mint(address indexed to, uint256 amount);
    event Burn(address indexed from, uint256 amount);
    
    // === Modifiers ===
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }
    
    // === Constructor ===
    constructor() {
        owner = msg.sender;
        levelRates[0] = 1e18;    // L0: 1 DCRED
        levelRates[1] = 5e18;    // L1: 5 DCRED
        levelRates[2] = 20e18;   // L2: 20 DCRED
        levelRates[3] = 80e18;   // L3: 80 DCRED
        levelRates[4] = 200e18;  // L4: 200 DCRED
    }
    
    // === ERC20 Core ===
    function transfer(address to, uint256 value) external returns (bool) {
        require(balanceOf[msg.sender] >= value, "Insufficient balance");
        balanceOf[msg.sender] -= value;
        balanceOf[to] += value;
        emit Transfer(msg.sender, to, value);
        return true;
    }
    
    function approve(address spender, uint256 value) external returns (bool) {
        allowance[msg.sender][spender] = value;
        return true;
    }
    
    function transferFrom(address from, address to, uint256 value) external returns (bool) {
        require(balanceOf[from] >= value, "Insufficient balance");
        require(allowance[from][msg.sender] >= value, "Insufficient allowance");
        balanceOf[from] -= value;
        allowance[from][msg.sender] -= value;
        balanceOf[to] += value;
        emit Transfer(from, to, value);
        return true;
    }
    
    // === Minting (owner only, for initial distribution) ===
    function mint(address to, uint256 amount) external onlyOwner {
        totalSupply += amount;
        balanceOf[to] += amount;
        emit Mint(to, amount);
    }
    
    function burn(uint256 amount) external {
        require(balanceOf[msg.sender] >= amount, "Insufficient balance");
        balanceOf[msg.sender] -= amount;
        totalSupply -= amount;
        emit Burn(msg.sender, amount);
    }
    
    // === Staking ===
    function stake(uint256 amount) external {
        require(balanceOf[msg.sender] >= amount, "Insufficient balance");
        balanceOf[msg.sender] -= amount;
        staked[msg.sender] += amount;
        emit Stake(msg.sender, amount);
    }
    
    function unstake(uint256 amount) external {
        require(staked[msg.sender] >= amount, "Insufficient stake");
        require(staked[msg.sender] - amount >= minStake || staked[msg.sender] == amount, 
                "Must maintain minimum stake");
        staked[msg.sender] -= amount;
        balanceOf[msg.sender] += amount;
        emit Unstake(msg.sender, amount);
    }
    
    // === Level rate updates ===
    function setLevelRate(uint8 level, uint256 rate) external onlyOwner {
        require(level <= 4, "Invalid level");
        levelRates[level] = rate;
    }
}


// ============================================================
// 2. Reputation System
// ============================================================

contract DecentralAIReputation {
    // === State ===
    address public owner;
    
    struct NodeReputation {
        uint96 score;           // 0-10000 (divide by 100 for 0-100.00)
        uint32 totalJobs;       // Total jobs completed
        uint32 successJobs;     // Successfully verified jobs
        uint32 failedJobs;      // Failed/slashed jobs
        uint64 lastUpdate;      // Timestamp of last update
        bool   isActive;        // Node is active
    }
    
    mapping(address => NodeReputation) public reputation;
    address[] public nodeList;
    mapping(address => bool) public isRegistered;
    
    // Slashing parameters
    uint96 public slashAmount = 500;     // -5.00 on failure
    uint96 public rewardAmount = 100;    // +1.00 on success
    uint96 public initialScore = 5000;   // 50.00 starting score
    uint96 public maxScore = 10000;      // 100.00 max
    uint96 public minActiveScore = 2000; // Below 20.00 = inactive
    
    // === Events ===
    event NodeRegistered(address indexed node, uint96 initialScore);
    event ReputationUpdated(address indexed node, uint96 newScore, bool success);
    event NodeSlashed(address indexed node, uint96 amount, string reason);
    event NodeDeactivated(address indexed node);
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }
    
    constructor() {
        owner = msg.sender;
    }
    
    // === Registration ===
    function register() external {
        require(!isRegistered[msg.sender], "Already registered");
        isRegistered[msg.sender] = true;
        nodeList.push(msg.sender);
        reputation[msg.sender] = NodeReputation({
            score: initialScore,
            totalJobs: 0,
            successJobs: 0,
            failedJobs: 0,
            lastUpdate: uint64(block.timestamp),
            isActive: true
        });
        emit NodeRegistered(msg.sender, initialScore);
    }
    
    // === Reputation Updates ===
    function reportSuccess(address node) external {
        NodeReputation storage rep = reputation[node];
        require(rep.isActive, "Node not active");
        
        rep.successJobs += 1;
        rep.totalJobs += 1;
        rep.lastUpdate = uint64(block.timestamp);
        
        // Increase score, cap at max
        uint96 newScore = rep.score + rewardAmount;
        if (newScore > maxScore) newScore = maxScore;
        rep.score = newScore;
        
        emit ReputationUpdated(node, newScore, true);
    }
    
    function reportFailure(address node, string calldata reason) external {
        NodeReputation storage rep = reputation[node];
        require(rep.isActive, "Node not active");
        
        rep.failedJobs += 1;
        rep.totalJobs += 1;
        rep.lastUpdate = uint64(block.timestamp);
        
        // Decrease score, floor at 0
        uint96 newScore;
        if (rep.score > slashAmount) {
            newScore = rep.score - slashAmount;
        } else {
            newScore = 0;
        }
        rep.score = newScore;
        
        // Auto-deactivate if below threshold
        if (newScore < minActiveScore) {
            rep.isActive = false;
            emit NodeDeactivated(node);
        }
        
        emit NodeSlashed(node, slashAmount, reason);
        emit ReputationUpdated(node, newScore, false);
    }
    
    // === Views ===
    function getReputation(address node) external view returns (
        uint96 score, uint32 totalJobs, uint32 successJobs, 
        uint32 failedJobs, bool isActive
    ) {
        NodeReputation storage rep = reputation[node];
        return (rep.score, rep.totalJobs, rep.successJobs, rep.failedJobs, rep.isActive);
    }
    
    function getNodeCount() external view returns (uint256) {
        return nodeList.length;
    }
    
    function getActiveNodes() external view returns (address[] memory) {
        uint256 count = 0;
        for (uint256 i = 0; i < nodeList.length; i++) {
            if (reputation[nodeList[i]].isActive) count++;
        }
        address[] memory active = new address[](count);
        uint256 idx = 0;
        for (uint256 i = 0; i < nodeList.length; i++) {
            if (reputation[nodeList[i]].isActive) {
                active[idx++] = nodeList[i];
            }
        }
        return active;
    }
}


// ============================================================
// 3. Settlement Contract
// ============================================================

contract DecentralAISettlement {
    // === State ===
    address public owner;
    DecentralAICredits public credits;
    DecentralAIReputation public reputation;
    
    struct InferenceJob {
        address requester;
        address provider;
        uint256 credits;
        uint8   providerLevel;
        uint64  timestamp;
        bool    settled;
        bool    disputed;
    }
    
    mapping(bytes32 => InferenceJob) public jobs;
    uint256 public totalJobs;
    uint256 public totalSettled;
    
    // Dispute resolution
    uint256 public disputeWindow = 1 hours;  // Time to dispute
    uint256 public disputeFee = 10e18;       // 10 DCRED to file dispute
    mapping(bytes32 => address[]) public disputeVoters;
    
    // === Events ===
    event JobCreated(bytes32 indexed jobId, address requester, address provider, uint256 credits);
    event JobSettled(bytes32 indexed jobId, uint256 credits);
    event JobDisputed(bytes32 indexed jobId, address disputer);
    event DisputeResolved(bytes32 indexed jobId, bool providerWins);
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }
    
    constructor(address _credits, address _reputation) {
        owner = msg.sender;
        credits = DecentralAICredits(_credits);
        reputation = DecentralAIReputation(_reputation);
    }
    
    // === Job Lifecycle ===
    
    /// Requester creates a job by escrowing credits
    function createJob(address provider, uint8 providerLevel) external returns (bytes32) {
        uint256 cost = credits.levelRates(providerLevel);
        require(credits.balanceOf(msg.sender) >= cost, "Insufficient credits");
        
        bytes32 jobId = keccak256(abi.encodePacked(
            msg.sender, provider, block.timestamp, totalJobs
        ));
        
        // Escrow credits
        credits.transferFrom(msg.sender, address(this), cost);
        
        jobs[jobId] = InferenceJob({
            requester: msg.sender,
            provider: provider,
            credits: cost,
            providerLevel: providerLevel,
            timestamp: uint64(block.timestamp),
            settled: false,
            disputed: false
        });
        
        totalJobs += 1;
        emit JobCreated(jobId, msg.sender, provider, cost);
        return jobId;
    }
    
    /// Provider completes job and claims payment
    function settleJob(bytes32 jobId) external {
        InferenceJob storage job = jobs[jobId];
        require(!job.settled, "Already settled");
        require(!job.disputed, "Under dispute");
        require(block.timestamp >= job.timestamp + disputeWindow, "Dispute window open");
        require(msg.sender == job.provider || msg.sender == owner, "Not authorized");
        
        job.settled = true;
        totalSettled += 1;
        
        // Pay provider (95%) + burn (5%) for deflationary pressure
        uint256 providerPay = (job.credits * 95) / 100;
        uint256 burnAmount = job.credits - providerPay;
        
        credits.transfer(job.provider, providerPay);
        credits.burn(burnAmount);
        
        // Update reputation
        reputation.reportSuccess(job.provider);
        
        emit JobSettled(jobId, job.credits);
    }
    
    /// Requester disputes a job result
    function disputeJob(bytes32 jobId) external {
        InferenceJob storage job = jobs[jobId];
        require(!job.settled, "Already settled");
        require(!job.disputed, "Already disputed");
        require(msg.sender == job.requester, "Not requester");
        require(block.timestamp < job.timestamp + disputeWindow, "Dispute window closed");
        
        job.disputed = true;
        
        // Lock dispute fee
        require(credits.balanceOf(msg.sender) >= disputeFee, "Insufficient dispute fee");
        credits.transferFrom(msg.sender, address(this), disputeFee);
        
        emit JobDisputed(jobId, msg.sender);
    }
    
    /// Owner resolves dispute
    function resolveDispute(bytes32 jobId, bool providerWins) external onlyOwner {
        InferenceJob storage job = jobs[jobId];
        require(job.disputed, "Not disputed");
        require(!job.settled, "Already settled");
        
        job.settled = true;
        
        if (providerWins) {
            // Provider gets payment + dispute fee
            uint256 providerPay = (job.credits * 95) / 100;
            credits.transfer(job.provider, providerPay);
            credits.burn(job.credits - providerPay);
            reputation.reportSuccess(job.provider);
        } else {
            // Requester gets refund, provider slashed
            credits.transfer(job.requester, job.credits);
            reputation.reportFailure(job.provider, "Dispute lost");
        }
        
        emit DisputeResolved(jobId, providerWins);
    }
}


// ============================================================
// 4. Governance (simple parameter updates)
// ============================================================

contract DecentralAIGovernance {
    address public owner;
    
    struct Proposal {
        string  description;
        address target;
        bytes   callData;
        uint256 voteCount;
        uint256 created;
        uint256 executeAfter;
        bool    executed;
        bool    passed;
    }
    
    mapping(uint256 => Proposal) public proposals;
    mapping(uint256 => mapping(address => bool)) public hasVoted;
    uint256 public proposalCount;
    
    uint256 public votingPeriod = 3 days;
    uint256 public quorum = 100e18; // 100 DCRED staked = 1 vote
    
    DecentralAICredits public credits;
    
    event ProposalCreated(uint256 indexed id, string description);
    event Voted(uint256 indexed id, address voter, uint256 weight);
    event ProposalExecuted(uint256 indexed id, bool passed);
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }
    
    constructor(address _credits) {
        owner = msg.sender;
        credits = DecentralAICredits(_credits);
    }
    
    function propose(string calldata description, address target, bytes calldata callData) 
        external returns (uint256) 
    {
        uint256 id = proposalCount++;
        proposals[id] = Proposal({
            description: description,
            target: target,
            callData: callData,
            voteCount: 0,
            created: block.timestamp,
            executeAfter: block.timestamp + votingPeriod,
            executed: false,
            passed: false
        });
        emit ProposalCreated(id, description);
        return id;
    }
    
    function vote(uint256 proposalId) external {
        require(!hasVoted[proposalId][msg.sender], "Already voted");
        require(block.timestamp < proposals[proposalId].executeAfter, "Voting ended");
        
        uint256 weight = credits.staked(msg.sender);
        require(weight >= quorum, "Need minimum stake to vote");
        
        hasVoted[proposalId][msg.sender] = true;
        proposals[proposalId].voteCount += weight / quorum;
        
        emit Voted(proposalId, msg.sender, weight / quorum);
    }
    
    function execute(uint256 proposalId) external {
        Proposal storage p = proposals[proposalId];
        require(!p.executed, "Already executed");
        require(block.timestamp >= p.executeAfter, "Voting period not ended");
        
        p.executed = true;
        p.passed = p.voteCount > 0;
        
        if (p.passed) {
            (bool ok,) = p.target.call(p.callData);
            require(ok, "Execution failed");
        }
        
        emit ProposalExecuted(proposalId, p.passed);
    }
}
