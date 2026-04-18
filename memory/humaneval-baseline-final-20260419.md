# HumanEval Baseline — Final Results (2026-04-19)

## Summary

| Metric | Value |
|--------|-------|
| **Pass@1** | **0/164 = 0.0%** |
| Completed | 164/164 (100%) |
| Crashes | 0 |
| Exec errors | 0 |
| Avg speed | 4.87 tok/s (range: 1.5-6.2) |
| Total time | ~3 hours (3 sessions with resume) |
| Model | RWKV-4-169M (raw, no fine-tuning) |
| Hardware | Pentium G4560, 8GB RAM, CPU only |

## Structural Analysis
- All 164 tasks have correct function signature (`def entry_point(...)`)
- 74/164 (45%) contain `return` statement somewhere in output
- Function bodies are random/gibberish tokens (expected for untrained model)
- No syntax errors in generated code structure

## Key Fixes That Made This Possible
1. **NaN overflow** (rwkv_model.rs): `decay.exp().clamp(1e-30, 1e10)` — prevents inf→NaN propagation
2. **full_code duplication** (humaneval.rs): `full_code = completion_text` instead of `prompt + completion`
3. **Resume capability**: `--resume` flag skips already-completed tasks
4. **Panic isolation**: Per-task panic handler prevents one crash from killing the whole run

## Files
- Results: `C:\Users\Administrator\.qclaw\workspace-agent-0b9a94a1\humaneval_results.jsonl` (512KB, 164 records)
- Binary: `D:\IdeaProjects\decentral-ai\src-rs\decentral-ai-core\target\release\humaneval.exe`
- Model: `D:\IdeaProjects\decentral-ai\data\rwkv4_169m.bin`
- Data: `D:\IdeaProjects\decentral-ai\data\HumanEval.jsonl`

## Next Steps
1. Use this 0% baseline to measure fine-tuning effectiveness
2. LoRA fine-tuning on unified_train.jsonl (5238 samples) — needs GPU
3. Re-run benchmark after fine-tuning to measure delta
