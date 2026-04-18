# HumanEval Baseline Progress — 2026-04-19

## Current Status
- **Running**: `humaneval.exe --model rwkv4_169m.bin --data HumanEval.jsonl --resume`
- **Progress**: 41/164 tasks completed (~25%)
- **Speed**: 3-6 tok/s (CPU)
- **Started**: ~00:05 GMT+8
- **Output**: `C:\Users\Administrator\.qclaw\workspace-agent-0b9a94a1\humaneval_results.jsonl`

## Completed Tasks (0-40)
All tasks completed without crashes. Speed stable at 3-6 tok/s.

## Key Fixes Applied
1. **NaN overflow fix** (rwkv_model.rs):
   - `decay.exp()` → `decay.exp().clamp(1e-30, 1e10)` in 2 places
   - Fixed attention WKV recurrence overflow → inf → NaN chain
2. **full_code duplication fix** (humaneval.rs):
   - `full_code = prompt + completion` → `full_code = completion_text`
   - completion already contains the full text

## Known Issues
- **Output quality**: Gibberish after function signature (169M raw model, expected)
- **Vocabulary mismatch**: id2token has 50254 tokens, model expects 50277 (23 tokens missing at end)
- **Crash at task 25**: Appeared once, then didn't recur on resume — likely OOM or specific prompt trigger

## Expected Pass@1
0% for 169M raw model. Baseline for fine-tuning comparison.

## Next Steps After Baseline
1. Parse results and compute exact baseline score
2. Build fine-tuning dataset from failed cases
3. Implement LoRA training pipeline (needs GPU)
