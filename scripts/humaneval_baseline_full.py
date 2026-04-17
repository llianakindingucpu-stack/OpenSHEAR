"""
DecentralAI MVP Step 1: HumanEval Full Baseline with RWKV-4-169M

Strategy: Use local RWKV for generation, use syntax check + pattern matching
for approximate correctness evaluation (no sandbox needed).
"""
import sys, os, json, time
sys.path.insert(0, r'D:\pylib')
os.environ['PYTHONIOENCODING'] = 'utf-8'

from rwkv.model import RWKV
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
import torch

# Config
MODEL_PATH = r'D:\pylib\rwkv-4-169m-native.pth'
VOCAB_PATH = r'D:\pylib\rwkv4-world-tok\rwkv_vocab_v20230424.txt'
MAX_TOKENS = 150
TEMPERATURE = 0.1  # very low for code
NUM_PROBLEMS = 164  # full HumanEval

# Full HumanEval (164 problems) - load from file if available, else use embedded subset
HUMAN_EVAL_PATH = r'D:\IdeaProjects\decentral-ai\data\HumanEval.jsonl'

def load_humaneval():
    """Load HumanEval problems"""
    if os.path.exists(HUMAN_EVAL_PATH):
        problems = []
        with open(HUMAN_EVAL_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    problems.append(json.loads(line))
        return problems
    
    # Fallback: embedded first 40 problems (abbreviated for disk space)
    return [
        {"task_id": "HumanEval/0", "prompt": "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than given threshold. \"\"\"\n"},
        {"task_id": "HumanEval/1", "prompt": "from typing import List\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to separate those group into separate strings and return the list of those. \"\"\"\n"},
        {"task_id": "HumanEval/2", "prompt": "def truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into and integer part (largest integer smaller than given number) and decimals (leftover part always smaller than 1). Return the decimal part of the number. \"\"\"\n"},
        {"task_id": "HumanEval/3", "prompt": "from typing import List\n\ndef below_zero(operations: List[int]) -> bool:\n    \"\"\" You're given a list of deposit and withdrawal operations on a bank account. Return True if the balance falls below zero at any point. \"\"\"\n"},
        {"task_id": "HumanEval/4", "prompt": "from typing import List\n\ndef mean_absolute_deviation(numbers: List[float]) -> float:\n    \"\"\" For a given list of input numbers, calculate Mean Absolute Deviation around the mean of this dataset. \"\"\"\n"},
        {"task_id": "HumanEval/5", "prompt": "from typing import List\n\ndef intersperse(numbers: List[int], delimeter: int) -> List[int]:\n    \"\"\" Insert a delimeter between every two consecutive items of the input list. \"\"\"\n"},
        {"task_id": "HumanEval/6", "prompt": "from typing import List\n\ndef parse_nested_parens(paren_string: str) -> List[int]:\n    \"\"\" Input to this function is a string represented multiple groups for nested parentheses separated by spaces. For each of the group, output the deepest level of nesting. \"\"\"\n"},
        {"task_id": "HumanEval/7", "prompt": "from typing import List\n\ndef filter_by_substring(strings: List[str], substring: str) -> List[str]:\n    \"\"\" Filter an input list of strings only for ones that contain given substring \"\"\"\n"},
        {"task_id": "HumanEval/8", "prompt": "from typing import List, Tuple\n\ndef sum_product(numbers: List[int]) -> Tuple[int, int]:\n    \"\"\" For a given list of integers, return a tuple consisting of a sum and a product of all the integers. \"\"\"\n"},
        {"task_id": "HumanEval/9", "prompt": "from typing import List, Tuple\n\ndef rolling_max(numbers: List[int]) -> List[int]:\n    \"\"\" From a given list of integers, generate a list of rolling maximum element found until given moment in the sequence. \"\"\"\n"},
    ]


def generate_completion(model, tokenizer, prompt, max_tokens=150, temperature=0.1):
    """Generate code completion"""
    tokens = tokenizer.encode(prompt)
    state = None
    out, state = model.forward(tokens, state)
    
    generated_ids = []
    for _ in range(max_tokens):
        logits = out.float()
        if temperature > 0:
            logits = logits / temperature
            # Clamp to avoid overflow
            logits = torch.clamp(logits, -100, 100)
            probs = torch.softmax(logits, dim=-1)
            # Top-k=50 sampling
            top_k = 50
            values, indices = torch.topk(probs, top_k)
            probs_top = values / values.sum()
            idx = torch.multinomial(probs_top, 1)
            token = indices[idx].item()
        else:
            token = out.float().argmax().item()
        
        generated_ids.append(token)
        
        # Stop conditions
        decoded = tokenizer.decode(generated_ids[-3:])
        if '\ndef ' in decoded or '\nclass ' in decoded or '\nif __name__' in decoded:
            break
        if token == 0:  # EOS
            break
        
        out, state = model.forward([token], state)
    
    return tokenizer.decode(generated_ids)


def check_syntax(code: str) -> bool:
    """Quick syntax check"""
    try:
        compile(code, '<string>', 'exec')
        return True
    except SyntaxError:
        return False


def has_return(code: str) -> bool:
    """Check if code has a return statement"""
    return 'return ' in code


def has_function_body(code: str) -> bool:
    """Check if code has indented lines after the prompt"""
    lines = code.split('\n')
    indented = [l for l in lines if l.startswith('    ') and l.strip()]
    return len(indented) >= 1


def main():
    print("=" * 60)
    print("DecentralAI MVP Step 1: HumanEval Full Baseline")
    print("Model: RWKV-4-169M-Pile (CPU, fp32)")
    print("Machine: Pentium G4560, 8GB RAM, No GPU")
    print("=" * 60)
    
    # Load model
    print("\n[1/3] Loading model...")
    t0 = time.time()
    model = RWKV(model=MODEL_PATH, strategy='cpu fp32')
    print(f"  Loaded in {time.time()-t0:.1f}s")
    
    print("\n[2/3] Loading tokenizer...")
    tokenizer = TRIE_TOKENIZER(VOCAB_PATH)
    print(f"  Ready")
    
    # Load problems
    problems = load_humaneval()
    actual_num = min(NUM_PROBLEMS, len(problems))
    print(f"\n[3/3] Running HumanEval ({actual_num} problems)...")
    
    results = []
    total_t0 = time.time()
    syntax_pass = 0
    has_return_count = 0
    has_body_count = 0
    
    for i, problem in enumerate(problems[:actual_num]):
        task_id = problem["task_id"]
        prompt = problem["prompt"]
        
        t0 = time.time()
        try:
            completion = generate_completion(
                model, tokenizer, prompt,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )
            elapsed = time.time() - t0
            full_code = prompt + completion
            
            syntax_ok = check_syntax(full_code)
            ret = has_return(completion)
            body = has_function_body(completion)
            
            if syntax_ok: syntax_pass += 1
            if ret: has_return_count += 1
            if body: has_body_count += 1
            
            result = {
                "task_id": task_id,
                "completion": completion[:300],
                "syntax_ok": syntax_ok,
                "has_return": ret,
                "has_body": body,
                "gen_time": round(elapsed, 2),
                "error": None
            }
        except Exception as e:
            result = {
                "task_id": task_id,
                "completion": "",
                "syntax_ok": False,
                "has_return": False,
                "has_body": False,
                "gen_time": 0,
                "error": str(e)[:100]
            }
        
        results.append(result)
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{actual_num}] syntax={syntax_pass}/{i+1} ret={has_return_count} body={has_body_count}")
    
    total_elapsed = time.time() - total_t0
    
    # Summary
    print("\n" + "=" * 60)
    print("BASELINE RESULTS - RWKV-4-169M-Pile")
    print("=" * 60)
    print(f"Problems tested: {actual_num}")
    print(f"Syntax pass: {syntax_pass}/{actual_num} ({syntax_pass/actual_num*100:.1f}%)")
    print(f"Has return: {has_return_count}/{actual_num} ({has_return_count/actual_num*100:.1f}%)")
    print(f"Has body: {has_body_count}/{actual_num} ({has_body_count/actual_num*100:.1f}%)")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    
    # Save
    out_path = r'D:\IdeaProjects\decentral-ai\results\humaneval_baseline_rwkv4_169m_full.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            "model": "RWKV-4-169M-Pile",
            "strategy": "cpu fp32",
            "hardware": "Pentium G4560, 8GB RAM, No GPU",
            "num_problems": actual_num,
            "syntax_pass_rate": syntax_pass / actual_num,
            "has_return_rate": has_return_count / actual_num,
            "has_body_rate": has_body_count / actual_num,
            "total_time": total_elapsed,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {out_path}")
    
    # Also note: 430M model doesn't fit in memory (OOM on this machine)
    print("\n--- Hardware Notes ---")
    print("RWKV-4-169M: OK, ~5.8 tok/s")
    print("RWKV-4-World-430M: OOM (needs >3GB free RAM after OS)")
    print("Need 16GB RAM machine or GPU for larger models")


if __name__ == "__main__":
    main()
