"""
SHEAR LoRA Fine-tuning Script
针对代码生成任务的 QLoRA 微调

用法：
  python train_shear_lora.py --model RWKV/rwkv-4-world-430m --data unified_train.jsonl
  python train_shear_lora.py --model deepseek-ai/deepseek-coder-1.3b-base --data unified_train.jsonl

支持的模型：
  - RWKV/rwkv-4-world-430m (小模型，快速验证)
  - RWKV/rwkv-5-world-1.5b (中等模型)
  - Qwen/Qwen2.5-1.5B-Instruct (通用模型)
  - deepseek-ai/deepseek-coder-1.3b-base (代码专用)
  - deepseek-ai/deepseek-coder-6.7b-base (需要更大显存)

环境要求：
  - PyTorch 2.8.0 + CUDA 12.8 (Blackwell 架构)
  - transformers peft bitsandbytes accelerate
"""
import sys
import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


class CodeDataset(Dataset):
    """代码训练数据集
    
    数据格式 (JSONL):
    {
        "id": "apps_001",
        "prompt": "def solve(n):",
        "solution": "    return n * 2\n",
        ...
    }
    """
    
    def __init__(self, data_path, tokenizer, max_length=512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载 JSONL 数据
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        
        print(f"[Dataset] Loaded {len(self.samples)} samples from {data_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # 构建训练文本：prompt + solution
        if 'solution' in item:
            text = item['prompt'] + item['solution']
        elif 'canonical_solution' in item:
            text = item['prompt'] + item['canonical_solution']
        elif 'text' in item:
            text = item['text']
        else:
            text = item['prompt']
        
        # Tokenize
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)
        
        # Labels = input_ids（因果语言模型）
        labels = input_ids.clone()
        # Pad token 的 label 设为 -100（忽略）
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def main():
    parser = argparse.ArgumentParser(description='SHEAR LoRA Fine-tuning')
    parser.add_argument('--model', default='RWKV/rwkv-4-world-430m', 
                        help='模型名称 (Hugging Face)')
    parser.add_argument('--data', required=True, 
                        help='训练数据路径 (.jsonl)')
    parser.add_argument('--output', default='./shear_lora_output',
                        help='LoRA 输出目录')
    parser.add_argument('--epochs', type=int, default=3,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批大小')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='学习率')
    parser.add_argument('--lora_r', type=int, default=8,
                        help='LoRA 秩')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA alpha')
    parser.add_argument('--max_length', type=int, default=512,
                        help='最大序列长度')
    parser.add_argument('--gradient_accumulation', type=int, default=4,
                        help='梯度累积步数')
    parser.add_argument('--fp16', action='store_true',
                        help='使用 FP16（不量化）')
    parser.add_argument('--no_quant', action='store_true',
                        help='不使用量化（显存充足时）')
    args = parser.parse_args()
    
    # 打印配置
    print("=" * 70)
    print("SHEAR LoRA Fine-tuning")
    print("=" * 70)
    print(f"Model:      {args.model}")
    print(f"Data:       {args.data}")
    print(f"Output:     {args.output}")
    print(f"Epochs:     {args.epochs}")
    print(f"Batch:      {args.batch_size} x {args.gradient_accumulation} = {args.batch_size * args.gradient_accumulation}")
    print(f"LR:         {args.lr}")
    print(f"LoRA:       r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"Max Length: {args.max_length}")
    print(f"Quantized:  {not args.no_quant}")
    print("=" * 70)
    
    # 检查 GPU
    if not torch.cuda.is_available():
        print("\n[ERROR] CUDA not available. This script requires GPU.")
        sys.exit(1)
    
    print(f"\n[GPU] {torch.cuda.get_device_name(0)}")
    print(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 1. 加载 Tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  Set pad_token = eos_token ({tokenizer.eos_token})")
    
    # 2. 加载模型
    print("\n[2/5] Loading model...")
    if args.no_quant:
        # 不量化（显存充足时，如 32GB+）
        print("  Loading in FP16 (no quantization)...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True,
        )
    else:
        # 4-bit 量化（省显存）
        print("  Loading in 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map='auto',
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    
    # 3. 应用 LoRA
    print("\n[3/5] Applying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        # 目标模块：适配大多数 Transformer 模型
        target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        bias='none',
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 4. 加载数据
    print("\n[4/5] Loading data...")
    dataset = CodeDataset(args.data, tokenizer, args.max_length)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        pin_memory=True
    )
    
    # 5. 训练
    print("\n[5/5] Training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs * len(dataloader)
    )
    
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        for i, batch in enumerate(dataloader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation
            loss.backward()
            
            if (i + 1) % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            total_loss += loss.item() * args.gradient_accumulation
            
            if i % 50 == 0:
                avg_loss = total_loss / (i + 1)
                print(f"  Epoch {epoch+1}/{args.epochs} [{i}/{len(dataloader)}] "
                      f"loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}")
        
        print(f"\n[Epoch {epoch+1}] avg_loss={total_loss/len(dataloader):.4f}")
    
    # 保存
    print(f"\n[Saving] {args.output}...")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    
    # 保存训练配置
    config = {
        'model': args.model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'max_length': args.max_length,
        'samples': len(dataset),
    }
    with open(f"{args.output}/training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"LoRA saved to: {args.output}")
    print("\nNext steps:")
    print(f"  1. Evaluate: python eval_shear_lora.py --lora {args.output}")
    print(f"  2. Merge:   python merge_lora.py --lora {args.output}")
    print("=" * 70)


if __name__ == '__main__':
    main()
