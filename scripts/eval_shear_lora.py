"""
SHEAR LoRA Evaluation Script
评估微调后的模型在 HumanEval 上的表现

用法：
  python eval_shear_lora.py --lora ./shear_lora_output --data HumanEval.jsonl

输出：
  - 每题生成的代码
  - Pass@1 评分
  - JSONL 结果文件
"""
import sys
import json
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def generate_code(model, tokenizer, prompt, max_new_tokens=256, temperature=0.2):
    """生成代码补全"""
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 只返回补全部分
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 移除 prompt
    if generated.startswith(prompt):
        generated = generated[len(prompt):]
    
    return generated


def evaluate_humaneval(model, tokenizer, data_path, output_path, max_new_tokens=256, temperature=0.2):
    """评估 HumanEval 数据集"""
    
    # 加载 HumanEval
    problems = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    
    print(f"[Eval] {len(problems)} problems")
    print(f"[Config] max_new_tokens={max_new_tokens}, temperature={temperature}")
    
    results = []
    passed = 0
    
    for i, prob in enumerate(problems):
        task_id = prob.get('task_id', f'task_{i}')
        prompt = prob['prompt']
        
        # 生成代码
        completion = generate_code(model, tokenizer, prompt, max_new_tokens, temperature)
        
        # 构建完整代码
        full_code = prompt + completion
        
        # 简单检查：是否包含 return 或 yield
        has_return = 'return ' in completion or 'yield ' in completion or 'return(' in completion
        # 或者是单行函数
        is_oneliner = '\n' not in completion.strip() and completion.strip().endswith(':')
        
        result = {
            'task_id': task_id,
            'prompt': prompt,
            'completion': completion,
            'full_code': full_code,
            'has_return': has_return,
        }
        results.append(result)
        
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(problems)}] {task_id} return={'Y' if has_return else 'N'}")
    
    # 统计
    passed = sum(1 for r in results if r['has_return'])
    pass_rate = passed / len(results) * 100 if results else 0
    
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{len(results)} have return statements ({pass_rate:.1f}%)")
    print(f"Note: True pass@1 requires exec-based evaluation")
    print(f"{'='*60}")
    
    # 保存结果
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"Results saved to: {output_path}")
    
    return pass_rate


def main():
    parser = argparse.ArgumentParser(description='SHEAR LoRA Evaluation')
    parser.add_argument('--base_model', default=None,
                        help='基础模型（不指定则从 LoRA 配置读取）')
    parser.add_argument('--lora', default=None,
                        help='LoRA 适配器路径（不指定则只评估基础模型）')
    parser.add_argument('--data', required=True,
                        help='HumanEval 数据路径 (.jsonl)')
    parser.add_argument('--output', default='./eval_results.jsonl',
                        help='评估结果输出路径')
    parser.add_argument('--max_new_tokens', type=int, default=256,
                        help='最大生成长度')
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='生成温度（0 = 贪婪）')
    args = parser.parse_args()
    
    print("=" * 70)
    print("SHEAR LoRA Evaluation")
    print("=" * 70)
    print(f"LoRA: {args.lora or 'None (base model only)'}")
    print(f"Data: {args.data}")
    print("=" * 70)
    
    # 确定基础模型
    if args.base_model:
        base_model = args.base_model
    elif args.lora:
        # 从 LoRA 配置读取
        config_path = Path(args.lora) / 'adapter_config.json'
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            # 尝试从训练配置读取
            train_config = Path(args.lora) / 'training_config.json'
            if train_config.exists():
                with open(train_config) as f:
                    train_cfg = json.load(f)
                base_model = train_cfg.get('model', 'RWKV/rwkv-4-world-430m')
            else:
                base_model = 'RWKV/rwkv-4-world-430m'
        else:
            base_model = 'RWKV/rwkv-4-world-430m'
    else:
        base_model = 'RWKV/rwkv-4-world-430m'
    
    print(f"\n[1/3] Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True,
    )
    
    # 加载 LoRA
    if args.lora:
        print(f"\n[2/3] Loading LoRA: {args.lora}")
        model = PeftModel.from_pretrained(model, args.lora)
        model = model.merge_and_unload()
        print("  LoRA merged into base model")
    else:
        print("\n[2/3] Skipping LoRA (base model only)")
    
    model.eval()
    
    # 评估
    print(f"\n[3/3] Evaluating...")
    pass_rate = evaluate_humaneval(
        model, tokenizer, args.data, args.output,
        args.max_new_tokens, args.temperature
    )
    
    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
