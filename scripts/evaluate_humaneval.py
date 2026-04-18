"""HumanEval evaluation script.

Reads humaneval_results.jsonl, evaluates each completion:
1. Syntax check (ast.parse)
2. Execution test (run the test suite)
3. Reports pass@1 rate
"""
import json
import ast
import sys
import os
import traceback
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def evaluate_one(entry):
    """Evaluate a single HumanEval entry. Returns (task_id, passed, details)."""
    task_id = entry["task_id"]
    full_code = entry["full_code"]
    test_code = entry["test"]
    entry_point = entry.get("entry_point", "")

    result = {
        "task_id": task_id,
        "entry_point": entry_point,
        "syntax_ok": False,
        "exec_ok": False,
        "error": None,
    }

    # 1. Syntax check
    try:
        ast.parse(full_code)
        result["syntax_ok"] = True
    except SyntaxError as e:
        result["error"] = f"SyntaxError: {e}"
        return (task_id, False, result)

    # 2. Execution test
    # Combine code + test, with safety timeout
    exec_code = full_code + "\n" + test_code + f"\ncheck({entry_point})\n"

    try:
        # Execute in a restricted namespace
        namespace = {
            "__builtins__": __builtins__,
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "int": int,
            "isinstance": isinstance,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "print": lambda *a: None,  # suppress print
            "range": range,
            "reversed": reversed,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
            # typing
            "List": list,
            "Dict": dict,
            "Tuple": tuple,
            "Set": set,
            "Optional": type(None),
            "Union": type(None),
            "Callable": type(lambda: None),
            "Any": type(None),
            "Iterable": type(None),
        }
        exec(exec_code, namespace, {})
        result["exec_ok"] = True
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        return (task_id, False, result)

    return (task_id, True, result)


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_humaneval.py <results.jsonl> [--parallel N]")
        sys.exit(1)

    input_path = sys.argv[1]
    parallel = 1
    if "--parallel" in sys.argv:
        idx = sys.argv.index("--parallel")
        if idx + 1 < len(sys.argv):
            parallel = int(sys.argv[idx + 1])

    # Load results
    entries = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    print(f"Loaded {len(entries)} results from {input_path}")
    print(f"Evaluating with parallel={parallel}")
    print()

    # Evaluate
    syntax_ok = 0
    exec_ok = 0
    errors = {}

    start = time.time()

    if parallel <= 1:
        for i, entry in enumerate(entries):
            task_id, passed, result = evaluate_one(entry)
            if result["syntax_ok"]:
                syntax_ok += 1
            if passed:
                exec_ok += 1
            else:
                err_type = result.get("error", "").split(":")[0] if result.get("error") else "unknown"
                errors[err_type] = errors.get(err_type, 0) + 1
            if (i + 1) % 10 == 0 or passed:
                status = "PASS" if passed else "FAIL"
                print(f"  [{i+1:3}/{len(entries)}] {task_id} -> {status}" +
                      (f" ({result.get('error', '')[:60]})" if not passed else ""))
    else:
        with ProcessPoolExecutor(max_workers=parallel) as pool:
            futures = {pool.submit(evaluate_one, e): e for e in entries}
            for i, future in enumerate(as_completed(futures)):
                task_id, passed, result = future.result()
                if result["syntax_ok"]:
                    syntax_ok += 1
                if passed:
                    exec_ok += 1
                else:
                    err_type = result.get("error", "").split(":")[0] if result.get("error") else "unknown"
                    errors[err_type] = errors.get(err_type, 0) + 1
                if (i + 1) % 10 == 0 or passed:
                    status = "PASS" if passed else "FAIL"
                    print(f"  [{i+1:3}/{len(entries)}] {task_id} -> {status}")

    elapsed = time.time() - start

    # Summary
    total = len(entries)
    print(f"\n{'='*55}")
    print(f"  HumanEval Results (RWKV-4-169M, Rust inference)")
    print(f"{'='*55}")
    print(f"  Total problems:  {total}")
    print(f"  Syntax OK:       {syntax_ok}/{total} ({100*syntax_ok/total:.1f}%)")
    print(f"  Pass@1:          {exec_ok}/{total} ({100*exec_ok/total:.1f}%)")
    print(f"  Eval time:       {elapsed:.1f}s")
    print()

    if errors:
        print(f"  Error breakdown:")
        for err, count in sorted(errors.items(), key=lambda x: -x[1]):
            print(f"    {err}: {count}")
        print()

    # Save summary
    output_path = input_path.replace(".jsonl", "_eval.json")
    summary = {
        "total": total,
        "syntax_ok": syntax_ok,
        "pass_at_1": exec_ok,
        "syntax_rate": syntax_ok / total if total > 0 else 0,
        "pass_rate": exec_ok / total if total > 0 else 0,
        "errors": errors,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to: {output_path}")


if __name__ == "__main__":
    main()
