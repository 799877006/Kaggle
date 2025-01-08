from transformers import AutoTokenizer, AutoModelForCausalLM,TrainingArguments,Trainer, BitsAndBytesConfig, DataCollatorForLanguageModeling

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset, random_split
import pandas as pd
import os
from transformers.integrations import TensorBoardCallback

from torch.utils.tensorboard import SummaryWriter
import json

# 创建自定义的TensorBoard回调
class CustomTensorBoardCallback(TensorBoardCallback):
    def __init__(self):
        super().__init__()
        self.writer = None
        
    def on_init_end(self, args, state, control, model=None, **kwargs):
        super().on_init_end(args, state, control, **kwargs)
        self.writer = SummaryWriter(args.logging_dir)
        
        # 记录所有配置参数
        config_dict = {
            "training_args": {},
            "model_config": {},
            "peft_config": {},
            "quantization_config": {},
            "dataset_config": {}
        }
        
        # 1. 记录训练参数
        if hasattr(args, 'to_dict'):
            config_dict["training_args"] = args.to_dict()
            
        # 2. 记录模型配置
        if model and hasattr(model, 'config'):
            config_dict["model_config"] = model.config.to_dict()
            
        # 3. 记录 PEFT 配置
        if model and hasattr(model, 'peft_config'):
            peft_dict = {
                "r": peft_config.r,
                "lora_alpha": peft_config.lora_alpha,
                "lora_dropout": peft_config.lora_dropout,
                "inference_mode": peft_config.inference_mode,
                "bias": peft_config.bias,
                "task_type": str(peft_config.task_type),
                "target_modules": list(peft_config.target_modules),
            }
            config_dict["peft_config"] = peft_dict
            
        # 4. 记录量化配置
        if hasattr(model, 'config') and hasattr(model.config, 'quantization_config'):
            quant_config = model.config.quantization_config
            if isinstance(quant_config, dict):
                config_dict["quantization_config"] = quant_config
            else:
                config_dict["quantization_config"] = quant_config.to_dict()
            
        # 5. 记录数据集配置
        config_dict["dataset_config"] = {
            "max_length": 1024,
            "train_val_split": "90:10",
            "random_seed": 42
        }
        
        # 将所有配置写入 tensorboard
        self.writer.add_text('configurations', json.dumps(config_dict, indent=2, ensure_ascii=False))
        
        # 记录重要的标量参数
        # 训练相关
        self.writer.add_scalar('hyperparameters/learning_rate', args.learning_rate, 0)
        self.writer.add_scalar('hyperparameters/batch_size_train', args.per_device_train_batch_size, 0)
        self.writer.add_scalar('hyperparameters/batch_size_eval', args.per_device_eval_batch_size, 0)
        self.writer.add_scalar('hyperparameters/num_epochs', args.num_train_epochs, 0)
        self.writer.add_scalar('hyperparameters/warmup_ratio', args.warmup_ratio, 0)
        self.writer.add_scalar('hyperparameters/weight_decay', args.weight_decay, 0)
        self.writer.add_scalar('hyperparameters/gradient_accumulation_steps', args.gradient_accumulation_steps, 0)
        self.writer.add_scalar('hyperparameters/max_grad_norm', args.max_grad_norm, 0)
        
        # 序列相关
        if hasattr(model, 'config'):
            self.writer.add_scalar('sequence/max_length', model.config.max_position_embeddings, 0)
            self.writer.add_scalar('sequence/vocab_size', model.config.vocab_size, 0)
            self.writer.add_scalar('sequence/hidden_size', model.config.hidden_size, 0)
            if hasattr(model.config, 'max_sequence_length'):
                self.writer.add_scalar('sequence/max_sequence_length', model.config.max_sequence_length, 0)
        
        # 数据集序列长度
        self.writer.add_scalar('sequence/dataset_max_length', 1024, 0)  # 从 CustomDataset 的默认值
        
        # 记录 tokenizer 配置
        if hasattr(tokenizer, 'model_max_length'):
            self.writer.add_scalar('sequence/tokenizer_max_length', tokenizer.model_max_length, 0)
        
        # LoRA 相关
        if model and hasattr(model, 'peft_config'):
            self.writer.add_scalar('lora/rank', peft_config.r, 0)
            self.writer.add_scalar('lora/alpha', peft_config.lora_alpha, 0)
            self.writer.add_scalar('lora/dropout', peft_config.lora_dropout, 0)
            self.writer.add_text('lora/target_modules', str(list(peft_config.target_modules)), 0)
            
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.writer:
            return
        
        logs = logs or {}
        
        # 记录训练指标
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f'train/{k}', v, state.global_step)
                
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not self.writer:
            return
            
        # 记录评估指标
        metrics = metrics or {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'eval/{key}', value, state.global_step)

# 配置4位量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 设置环境变量以优化内存分配
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it",padding='max_length',truncation=True,max_length=1024)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b-it",
    device_map={"": 0},
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)

# 为k-bit训练准备模型
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_disable()

# 配置LoRA
peft_config = LoraConfig(
    inference_mode=False,
    r=64,
    lora_alpha=128,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj","o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


class CustomDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=1024):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = f"<start_of_turn>user\n{row['title']}<end_of_turn><eos>\n<start_of_turn>model\n{row['content']}<end_of_turn><eos>\n"
        return text

# 创建完整数据集
full_dataset = CustomDataset(csv_file="/home/yangrumei/Kaggle/dataset/train_dataset.csv", tokenizer=tokenizer)

# 计算训练集和验证集的大小
train_size = int(0.9 * len(full_dataset))  # 90% 用于训练
val_size = len(full_dataset) - train_size   # 10% 用于验证

# 随机分割数据集
train_dataset, val_dataset = random_split(
    full_dataset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # 设置随机种子以确保可重复性
)

def data_collator(features):
    # 使用 tokenizer 的 __call__ 方法一次性完成编码和填充
    batch = tokenizer(
        features,  # features 现在是文本列表
        padding=True,
        truncation=True,
        max_length=1024,
        return_tensors="pt",
    )
    
    # 添加 labels
    batch["labels"] = batch["input_ids"].clone()
    
    return batch

training_args = TrainingArguments(
    output_dir="models/train_dataset/gemma-2-9b-it-qlora-content1024",
    logging_dir="logs/tensorboard/train_dataset",
    logging_steps=1,
    learning_rate=1e-4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,   
    num_train_epochs=3,
    weight_decay=0.01,
    optim="adamw_apex_fused",
    bf16=True,
    bf16_full_eval=True,
    # group_by_length=True,
    gradient_accumulation_steps=4,
    remove_unused_columns=True,
    max_grad_norm=1,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    save_steps=100,
    eval_strategy="epoch",
    eval_steps=100,
    ddp_find_unused_parameters=False,
    report_to="tensorboard",
    dataloader_pin_memory=False,
)

# 创建自定义的tensorboard callback实例
custom_callback = CustomTensorBoardCallback()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[custom_callback],
)

trainer.train()