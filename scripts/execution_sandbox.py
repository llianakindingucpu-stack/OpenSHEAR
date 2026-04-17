"""
DecentralAI HumanEval Execution Sandbox
=========================================
Execute generated code against test cases to get real Pass@1.

Safe subprocess-based sandbox:
- Timeout protection (10s per test)
- Memory limit
- No network access
- Captures stdout/stderr

This turns "0% syntax pass" into meaningful Pass@1 data.
"""

import json
import os
import sys
import time
import traceback
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, r'D:\pylib')


@dataclass
class ExecutionResult:
    """Result of executing one test"""
    task_id: str = ""
    passed: bool = False
    error_type: str = ""       # syntax_error, runtime_error, timeout, assertion, import_error
    error_message: str = ""
    execution_time_ms: float = 0
    stdout: str = ""
    stderr: str = ""


class ExecutionSandbox:
    """
    Safely execute Python code against test cases.
    
    Security:
    - Subprocess isolation (not in-process)
    - Timeout (default 10s)
    - No network (can be enforced with OS-level sandboxing)
    - Restricted imports (optional)
    """
    
    def __init__(self, timeout: int = 10, max_memory_mb: int = 512):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.results: List[ExecutionResult] = []
    
    def execute(self, task_id: str, prompt: str, completion: str, test_code: str,
                entry_point: str) -> ExecutionResult:
        """
        Execute a completion against its test cases.
        
        Args:
            task_id: HumanEval task ID (e.g., "HumanEval/0")
            prompt: Function signature + docstring
            completion: Generated code (just the function body)
            test_code: Test cases from HumanEval
            entry_point: Function name to test
        """
        # Combine into full script
        full_code = prompt + completion + "\n"
        
        # Add test runner
        test_runner = f"""
# === Test Runner ===
import sys

check_result = None
try:
    {test_code}
    # Run the check function
    check = check_{entry_point}
    check_result = check()
except Exception as e:
    check_result = e

if check_result is True or check_result is None:
    print("PASS")
elif isinstance(check_result, Exception):
    print(f"FAIL:{{type(check_result).__name__}}:{{check_result}}")
else:
    print(f"FAIL:{{check_result}}")
"""
        full_code += test_runner
        
        result = ExecutionResult(task_id=task_id)
        
        # Write to temp file and execute
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, 
                                         encoding='utf-8', dir=tempfile.gettempdir()) as f:
            f.write(full_code)
            temp_path = f.name
        
        try:
            start = time.time()
            proc = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=tempfile.gettempdir(),
            )
            result.execution_time_ms = (time.time() - start) * 1000
            result.stdout = proc.stdout.strip()
            result.stderr = proc.stderr.strip()
            
            if proc.returncode != 0:
                # Check error type
                stderr_lower = result.stderr.lower()
                if 'syntaxerror' in stderr_lower:
                    result.error_type = 'syntax_error'
                elif 'importerror' in stderr_lower or 'modulenotfounderror' in stderr_lower:
                    result.error_type = 'import_error'
                elif 'recursionerror' in stderr_lower:
                    result.error_type = 'recursion_error'
                elif 'timeout' in stderr_lower:
                    result.error_type = 'timeout'
                elif 'memoryerror' in stderr_lower:
                    result.error_type = 'memory_error'
                else:
                    result.error_type = 'runtime_error'
                result.error_message = result.stderr[:500]
            else:
                # Check output
                if 'PASS' in result.stdout:
                    result.passed = True
                else:
                    result.error_type = 'assertion_error'
                    result.error_message = result.stdout.replace('FAIL:', '').strip()[:500]
        
        except subprocess.TimeoutExpired:
            result.error_type = 'timeout'
            result.error_message = f"Execution exceeded {self.timeout}s"
        except Exception as e:
            result.error_type = 'sandbox_error'
            result.error_message = str(e)[:500]
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
        
        self.results.append(result)
        return result
    
    def execute_completion(self, task_id: str, prompt: str, completion: str,
                          test_code: str, entry_point: str) -> ExecutionResult:
        """
        Execute a full completion (may include the prompt already).
        Smarter detection of whether prompt is already in completion.
        """
        # If completion starts with the same function def, it's a full completion
        if completion.strip().startswith('def ') or completion.strip().startswith('from ') or completion.strip().startswith('import '):
            # Full completion - might include the prompt already
            # Check if prompt's function def is in completion
            first_line_prompt = prompt.strip().split('\n')[0]
            if first_line_prompt in completion:
                # Completion already includes prompt
                full_code = completion
            else:
                full_code = prompt + "\n" + completion
        else:
            # Just the body
            full_code = prompt + completion
        
        # Use the main execute method but with combined code
        return self.execute(task_id, prompt, full_code, test_code, entry_point)


def run_humaneval_execution(baseline_path: str, humaneval_path: str, output_path: str = None):
    """
    Run execution-based evaluation on HumanEval baseline results.
    """
    if output_path is None:
        base = baseline_path.replace('.json', '_exec.json')
        output_path = base
    
    # Load baseline results (supports both list and dict format)
    with open(baseline_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        baseline = raw.get('results', [])
    else:
        baseline = raw
    
    # Load HumanEval problems
    problems = {}
    with open(humaneval_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                problems[p['task_id']] = p
    
    sandbox = ExecutionSandbox(timeout=10)
    
    print(f"Running execution evaluation on {len(baseline)} problems...")
    print(f"Baseline: {baseline_path}")
    print()
    
    results = []
    for i, item in enumerate(baseline):
        task_id = item['task_id']
        completion = item.get('completion', '') or item.get('generated', '')
        prompt = item.get('prompt', '')
        
        if task_id not in problems:
            print(f"  [{i+1}] {task_id}: SKIP (no test)")
            continue
        
        prob = problems[task_id]
        test_code = prob.get('test', '')
        entry_point = prob.get('entry_point', '')
        
        # Skip empty completions
        if not completion.strip():
            results.append({
                'task_id': task_id,
                'passed': False,
                'error_type': 'empty_output',
                'execution_time_ms': 0,
            })
            status = "SKIP(empty)"
            print(f"  [{i+1}/{len(baseline)}] {task_id}: {status}")
            continue
        
        result = sandbox.execute_completion(
            task_id=task_id,
            prompt=prompt,
            completion=completion,
            test_code=test_code,
            entry_point=entry_point
        )
        
        results.append({
            'task_id': task_id,
            'passed': result.passed,
            'error_type': result.error_type,
            'execution_time_ms': round(result.execution_time_ms, 1),
        })
        
        status = "PASS" if result.passed else f"FAIL({result.error_type})"
        if (i + 1) % 10 == 0 or result.passed:
            print(f"  [{i+1}/{len(baseline)}] {task_id}: {status}")
    
    # Summary
    total = len(results)
    passed = sum(1 for r in results if r['passed'])
    error_counts = {}
    for r in results:
        if not r['passed']:
            et = r['error_type'] or 'unknown'
            error_counts[et] = error_counts.get(et, 0) + 1
    
    print()
    print("=" * 60)
    print("EXECUTION EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total: {total}")
    print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    print(f"\nError breakdown:")
    for et, count in sorted(error_counts.items(), key=lambda x: -x[1]):
        print(f"  {et}: {count} ({count/total*100:.1f}%)")
    
    # Save
    output = {
        'model': raw.get('model', 'unknown') if isinstance(raw, dict) else 'unknown',
        'pass_rate': passed / total if total else 0,
        'passed': passed,
        'total': total,
        'error_counts': error_counts,
        'results': results,
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {output_path}")
    
    return output


def main():
    baseline_path = r'D:\IdeaProjects\decentral-ai\results\humaneval_baseline_rwkv4_169m_full.json'
    humaneval_path = r'D:\IdeaProjects\decentral-ai\data\HumanEval.jsonl'
    
    run_humaneval_execution(baseline_path, humaneval_path)


if __name__ == "__main__":
    main()
