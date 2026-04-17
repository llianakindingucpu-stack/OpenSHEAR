"""
DecentralAI MVP Step 1: HumanEval Benchmark with RWKV-4-169M (CPU)

Uses RWKV + torch CPU to run HumanEval code generation benchmark.
This is the baseline before any fine-tuning.
"""
import sys, os, json, time
sys.path.insert(0, r'D:\pylib')
os.environ['PYTHONIOENCODING'] = 'utf-8'

from rwkv.model import RWKV
from tokenizers import Tokenizer

# Config
MODEL_PATH = r'D:\pylib\rwkv-4-169m-native.pth'
TOKENIZER_PATH = r'D:\pylib\tokenizer.json'
MAX_TOKENS = 200  # max generation per problem
TEMPERATURE = 0.2  # low temp for code generation
TOP_P = 0.9
NUM_PROBLEMS = 20  # test on first 20 problems for speed (full=164)

# HumanEval problems (first 20, abbreviated)
HUMAN_EVAL = [
    {"task_id": "HumanEval/0", "prompt": "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    \"\"\"\n"},
    {"task_id": "HumanEval/1", "prompt": "from typing import List\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    \"\"\"\n"},
    {"task_id": "HumanEval/2", "prompt": "def truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n    Return the decimal part of the number.\n    \"\"\"\n"},
    {"task_id": "HumanEval/3", "prompt": "from typing import List\n\ndef below_zero(operations: List[int]) -> bool:\n    \"\"\" You're given a list of deposit and withdrawal operations on a bank account.\n    Return True if the balance falls below zero at any point.\n    \"\"\"\n"},
    {"task_id": "HumanEval/4", "prompt": "from typing import List\n\ndef mean_absolute_deviation(numbers: List[float]) -> float:\n    \"\"\" For a given list of input numbers, calculate Mean Absolute Deviation\n    around the mean of this dataset.\n    \"\"\"\n"},
    {"task_id": "HumanEval/5", "prompt": "from typing import List\n\ndef intersperse(numbers: List[int], delimeter: int) -> List[int]:\n    \"\"\" Insert a delimeter between every two consecutive items of the input list.\n    \"\"\"\n"},
    {"task_id": "HumanEval/6", "prompt": "from typing import List\n\ndef parse_nested_parens(paren_string: str) -> List[int]:\n    \"\"\" Input to this function is a string represented multiple groups for nested parentheses separated by spaces.\n    For each of the group, output the deepest level of nesting.\n    \"\"\"\n"},
    {"task_id": "HumanEval/7", "prompt": "def filter_by_substring(strings: List[str], substring: str) -> List[str]:\n    \"\"\" Filter an input list of strings only for ones that contain given substring\n    \"\"\"\n"},
    {"task_id": "HumanEval/8", "prompt": "from typing import List, Tuple\n\ndef sum_product(numbers: List[int]) -> Tuple[int, int]:\n    \"\"\" For a given list of integers, return a tuple consisting of a sum and a product of all the integers.\n    \"\"\"\n"},
    {"task_id": "HumanEval/9", "prompt": "from typing import List, Tuple\n\ndef rolling_max(numbers: List[int]) -> List[int]:\n    \"\"\" From a given list of integers, generate a list of rolling maximum element found until given moment\n    in the sequence.\n    \"\"\"\n"},
    {"task_id": "HumanEval/10", "prompt": "def is_palindrome(string: str) -> bool:\n    \"\"\" Test if given string is a palindrome \"\"\"\n"},
    {"task_id": "HumanEval/11", "prompt": "from typing import List\n\ndef string_xor(a: str, b: str) -> str:\n    \"\"\" Input are two strings a and b consisting only of 1s and 0s.\n    Perform binary XOR on these inputs and return result.\n    \"\"\"\n"},
    {"task_id": "HumanEval/12", "prompt": "from typing import List, Optional\n\ndef longest(strings: List[str]) -> Optional[str]:\n    \"\"\" Out of list of strings, return the longest one. Return the first one in case of multiple\n    strings of the same length. Return None in case the input list is empty.\n    \"\"\"\n"},
    {"task_id": "HumanEval/13", "prompt": "def greatest_common_divisor(a: int, b: int) -> int:\n    \"\"\" Return a greatest common divisor of two integers a and b \"\"\"\n"},
    {"task_id": "HumanEval/14", "prompt": "from typing import List\n\ndef all_ints_exclusive(l: List[int]) -> List[int]:\n    \"\"\" Return list of all integers from l that are not at even indices \"\"\"\n"},
    {"task_id": "HumanEval/15", "prompt": "def string_sequence(n: int) -> str:\n    \"\"\" Return a string containing space-delimited numbers starting from 0 upto n inclusive.\n    \"\"\"\n"},
    {"task_id": "HumanEval/16", "prompt": "def count_distinct_characters(string: str) -> int:\n    \"\"\" Given a string, find out how many distinct characters it consists of \"\"\"\n"},
    {"task_id": "HumanEval/17", "prompt": "from typing import List\n\ndef parse_music(music_string: str) -> List[int]:\n    \"\"\" Input to this function is a string representing musical notes in a special ASCII format.\n    \"\"\"\n"},
    {"task_id": "HumanEval/18", "prompt": "def how_many_times(string: str, substring: str) -> int:\n    \"\"\" Find how many times a given substring can be found in the original string. Count overlaping cases.\n    \"\"\"\n"},
    {"task_id": "HumanEval/19", "prompt": "from typing import List\n\ndef sort_numbers(numbers: str) -> str:\n    \"\"\" Input is a space-delimited string of numberals from 'zero' to 'nine'.\n    Sort the numbers from smallest to largest and return them as space-delimited string.\n    \"\"\"\n"},
]


def generate_code(model, tokenizer, prompt, max_tokens=200, temperature=0.2, top_p=0.9):
    """Generate code completion for a given prompt"""
    import torch
    
    tokens = tokenizer.encode(prompt).ids
    state = None
    
    # Process prompt
    out, state = model.forward(tokens, state)
    
    generated_ids = []
    for _ in range(max_tokens):
        # Sample with temperature
        logits = out.float()
        if temperature > 0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            # Top-p sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs - sorted_probs > top_p
            sorted_probs[sorted_indices_to_remove] = 0
            sorted_probs = sorted_probs / sorted_probs.sum()
            # Sample
            index = torch.multinomial(sorted_probs, 1)
            token = sorted_indices[index].item()
        else:
            token = out.argmax().item()
        
        # Stop on newline after function body
        generated_ids.append(token)
        
        # Simple stop condition: if we see a top-level def or class, stop
        decoded = tokenizer.decode(generated_ids[-5:])  # check last few tokens
        if '\nclass ' in decoded or '\ndef ' in decoded or '\nif __name__' in decoded:
            break
        
        out, state = model.forward([token], state)
    
    completion = tokenizer.decode(generated_ids)
    return completion


def check_syntax(code: str) -> bool:
    """Quick syntax check"""
    try:
        compile(code, '<string>', 'exec')
        return True
    except SyntaxError:
        return False


def main():
    print("=" * 60)
    print("DecentralAI MVP Step 1: HumanEval Baseline")
    print("Model: RWKV-4-169M (CPU, float32)")
    print("Machine: Pentium G4560, 8GB RAM, No GPU")
    print("=" * 60)
    
    # Load model
    print("\n[1/3] Loading model...")
    t0 = time.time()
    model = RWKV(model=MODEL_PATH, strategy='cpu fp32')
    print(f"  Model loaded in {time.time()-t0:.1f}s")
    
    # Load tokenizer
    print("\n[2/3] Loading tokenizer...")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")
    
    # Run benchmark
    print(f"\n[3/3] Running HumanEval (first {NUM_PROBLEMS} problems)...")
    results = []
    total_t0 = time.time()
    
    for i, problem in enumerate(HUMAN_EVAL[:NUM_PROBLEMS]):
        task_id = problem["task_id"]
        prompt = problem["prompt"]
        
        print(f"\n  [{i+1}/{NUM_PROBLEMS}] {task_id}...", end="", flush=True)
        t0 = time.time()
        
        try:
            completion = generate_code(
                model, tokenizer, prompt,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P
            )
            elapsed = time.time() - t0
            syntax_ok = check_syntax(prompt + completion)
            
            result = {
                "task_id": task_id,
                "completion": completion[:200],  # truncate for storage
                "syntax_ok": syntax_ok,
                "gen_time": round(elapsed, 2),
                "gen_tokens": len(tokenizer.encode(completion).ids),
                "tok_per_sec": round(len(tokenizer.encode(completion).ids) / max(elapsed, 0.01), 1),
                "error": None
            }
            print(f" {result['tok_per_sec']} tok/s, syntax={'OK' if syntax_ok else 'FAIL'}")
        except Exception as e:
            result = {
                "task_id": task_id,
                "completion": "",
                "syntax_ok": False,
                "gen_time": 0,
                "gen_tokens": 0,
                "tok_per_sec": 0,
                "error": str(e)
            }
            print(f" ERROR: {e}")
        
        results.append(result)
    
    total_elapsed = time.time() - total_t0
    
    # Summary
    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    
    syntax_pass = sum(1 for r in results if r["syntax_ok"])
    errors = sum(1 for r in results if r["error"])
    avg_speed = sum(r["tok_per_sec"] for r in results if r["tok_per_sec"] > 0) / max(len(results) - errors, 1)
    
    print(f"Problems tested: {NUM_PROBLEMS}")
    print(f"Syntax pass: {syntax_pass}/{NUM_PROBLEMS} ({syntax_pass/NUM_PROBLEMS*100:.0f}%)")
    print(f"Errors: {errors}")
    print(f"Avg speed: {avg_speed:.1f} tok/s")
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"\nNote: Syntax pass != functional correctness.")
    print(f"Full HumanEval execution test requires sandboxed eval.")
    print(f"This baseline establishes: model CAN generate code on this hardware.")
    
    # Save results
    out_path = r'D:\IdeaProjects\decentral-ai\results\humaneval_baseline_rwkv4_169m.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            "model": "RWKV-4-169M",
            "strategy": "cpu fp32",
            "hardware": "Pentium G4560, 8GB RAM, No GPU",
            "num_problems": NUM_PROBLEMS,
            "syntax_pass_rate": syntax_pass / NUM_PROBLEMS,
            "avg_tok_per_sec": avg_speed,
            "total_time": total_elapsed,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")
    
    return syntax_pass / NUM_PROBLEMS


if __name__ == "__main__":
    import torch  # needed for softmax/multinomial
    main()
