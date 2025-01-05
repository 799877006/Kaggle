from transformers import AutoTokenizer, AutoModelForCausalLM,TrainingArguments,Trainer, BitsAndBytesConfig
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset, random_split
import pandas as pd
import os

# 配置4位量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 设置环境变量以优化内存分配
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
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
        ## 需要加<eos>
        text = f"<start_of_turn>user\n{row['title']}<end_of_turn><eos>\n<start_of_turn>model\n{row['content']}<end_of_turn><eos>\n"
        
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings['input_ids'].squeeze(),
            "attention_mask": encodings['attention_mask'].squeeze(),
            "labels": encodings['input_ids'].squeeze()
        }

# 创建完整数据集
full_dataset = CustomDataset(csv_file="dataset/cleaned_data_no_nan.csv", tokenizer=tokenizer)

# 计算训练集和验证集的大小
train_size = int(0.9 * len(full_dataset))  # 90% 用于训练
val_size = len(full_dataset) - train_size   # 10% 用于验证

# 随机分割数据集
train_dataset, val_dataset = random_split(
    full_dataset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # 设置随机种子以确保可重复性
)

training_args = TrainingArguments(
    output_dir="models/cleaned_eos/gemma-2-9b-it-qlora-content1024",
    learning_rate=2e-4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,   
    num_train_epochs=3,
    weight_decay=0.01,
    optim="adamw_apex_fused",
    bf16=True,
    bf16_full_eval=True,
    group_by_length=True,
    gradient_accumulation_steps=4,
    remove_unused_columns=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    save_steps=100,
    eval_strategy="epoch",
    eval_steps=100,
    logging_steps=10,
    ddp_find_unused_parameters=False,
    report_to="none",
    dataloader_pin_memory=False,


)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)


trainer.train()