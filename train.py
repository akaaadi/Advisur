import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Trainer
)
from peft import LoraConfig, get_peft_model
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
DATA_FILE = "dataset.jsonl"
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
print("Loading model (4-bit GPU + CPU Offload)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},         
    torch_dtype=torch.float16,
)
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
print("\nLoading dataset...")
dataset = load_dataset("json", data_files=DATA_FILE)

def preprocess(example):
    text = f"Instruction: {example['instruction']}\nResponse: {example['output']}"
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    tokens["labels"] = tokens["input_ids"]
    return tokens

tokenized = dataset.map(preprocess)

args = TrainingArguments(
    output_dir="mistral_lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=3,         # ← use epochs not max_steps
    warmup_ratio=0.05,
    learning_rate=2e-4,
    logging_steps=10,
    fp16=True,
    save_strategy="epoch",
    optim="paged_adamw_8bit"
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    data_collator=DataCollatorForSeq2Seq(tokenizer),
)

print("\n🚀 Training started...\n")
trainer.train()

print("\n💾 Saving model...")
trainer.save_model("mistral_lora")
tokenizer.save_pretrained("mistral_lora")

print("\n✅ Training complete! LoRA saved to mistral_lora/")