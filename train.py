from transformers import AutoTokenizer, AutoModelForCausalLM,TrainingArguments,Trainer, BitsAndBytesConfig
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset, random_split
import pandas as pd

peft_config = LoraConfig(inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,target_modules=["q_proj", "v_proj"])


tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b-it",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    use_cache=False
)
model = get_peft_model(model, LoraConfig(
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
))
model.print_trainable_parameters()
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

class CustomDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=4096):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = f"<start_of_turn>user\n{row['title']}<end_of_turn>\n<start_of_turn>model\n{row['content']}<end_of_turn>"
        
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
full_dataset = CustomDataset(csv_file="dataset/cleaned_data.csv", tokenizer=tokenizer)

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
    output_dir="models/cleaned/gemma-2-9b-it-lora-content",
    learning_rate=1e-3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,   
    num_train_epochs=5,
    weight_decay=0.01,
    optim="adamw_torch_fused",

    
    # 其他优化参数保持不变
    gradient_accumulation_steps=8,

)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

trainer.train()