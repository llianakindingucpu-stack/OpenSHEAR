"""
HumanEval Evaluation Harness

Loads results from JSONL, extracts the generated function,
executes it with the test inputs, and reports pass/fail.
"""
import json
import sys
import traceback
from typing import Any, Callable

def load_results(path: str):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def extract_function(full_code: str, entry_point: str) -> Callable | None:
    """
    Extract the user-defined function from full_code and compile it.
    We need to separate the PROMPT (given context) from the COMPLETION (generated).
    
    The prompt ends at the function signature for the target entry_point.
    The completion is everything after that.
    """
    # Find where the entry_point function starts
    target_sig = f"def {entry_point}("
    sig_idx = full_code.find(target_sig)
    if sig_idx == -1:
        return None
    
    # The completion part starts after the function signature line
    # Find the end of the signature line (next newline after sig_idx)
    line_end = full_code.find('\n', sig_idx)
    if line_end == -1:
        return None
    
    completion_only = full_code[line_end+1:]
    
    # Build exec context: import typing + the function
    # We extract just the function body
    import_line = "from typing import *"
    
    # Try to extract the function as a standalone
    exec_code = import_line + "\n" + full_code
    
    namespace = {}
    try:
        exec(compile(exec_code, '<string>', 'exec'), namespace)
        if entry_point in namespace:
            return namespace[entry_point]
    except Exception as e:
        pass
    
    return None

def run_check(candidate: Callable, test_code: str) -> tuple[bool, str]:
    """
    Execute the test_code with the candidate function bound as 'candidate'.
    Returns (passed, error_message)
    """
    namespace = {'candidate': candidate}
    try:
        exec(compile(test_code, '<string>', 'exec'), namespace)
        return True, ""
    except AssertionError as e:
        return False, f"AssertionError: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def evaluate(results_path: str) -> dict:
    results = load_results(results_path)
    
    passed = 0
    failed = 0
    errors = 0
    details = []
    
    for r in results:
        tid = r['task_id']
        entry = r['entry_point']
        test_code = r.get('test_code', '')
        canonical = r.get('canonical_solution', '')
        speed = r.get('speed_tok_per_s', 0)
        tokens = r.get('completion_tokens', 0)
        comp_text = r.get('completion_text', '')
        
        # Extract function
        candidate = extract_function(r.get('full_code', ''), entry)
        
        if candidate is None:
            status = "ERROR"
            detail = "Could not extract function from generated code"
            errors += 1
        else:
            # Run tests
            ok, err = run_check(candidate, test_code)
            if ok:
                status = "PASS"
                detail = ""
                passed += 1
            else:
                status = "FAIL"
                detail = err
                failed += 1
        
        details.append({
            'task_id': tid,
            'status': status,
            'detail': detail,
            'speed': speed,
            'tokens': tokens,
        })
        
        status_sym = "PASS" if status == "PASS" else ("ERR" if status == "ERROR" else "FAIL")
        print(f"  [{status_sym:4s}] {tid:20s} {detail[:60] if detail else ''}")
    
    total = passed + failed + errors
    print()
    print(f"=" * 50)
    print(f"  Results: {passed}/{total} passed, {failed} failed, {errors} errors")
    print(f"  Pass@1:  {passed/total*100:.1f}%")
    print(f"=" * 50)
    
    return {
        'passed': passed,
        'failed': failed,
        'errors': errors,
        'total': total,
        'pass_rate': passed / total if total > 0 else 0,
        'details': details,
    }

if __name__ == '__main__':
    result_path = sys.argv[1] if len(sys.argv) > 1 else 'humaneval_results.jsonl'
    evaluate(result_path)
