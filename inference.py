# inference.py
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig


BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"   
LORA_DIR = "mistral_lora"                          
USE_4BIT = True                                    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2
TOP_P = 0.95
REPETITION_PENALTY = 1.1
STOP_STRS = ["\nUser:", "\nUser :", "\nUser", "### End", "Response:"]  # generator will try to stop on these fragments

# --------- load tokenizer ----------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --------- prepare bitsandbytes config (if using 4-bit) ----------
bnb_config = None
device_map = None
if USE_4BIT and DEVICE == "cuda":
    print("Preparing 4-bit quantization config for inference...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )
    
    device_map = "auto"  
else:
    device_map = {"": DEVICE}

# ---------- load model ----------
print("Loading base model (this may take a while)...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map=device_map,
    quantization_config=bnb_config,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    low_cpu_mem_usage=True,
)

# ---------- load LoRA / PEFT adapters ----------
if os.path.isdir(LORA_DIR):
    print(f"Loading LoRA adapters from '{LORA_DIR}'...")
    # Use PeftModel to load adapters on top of base model
    model = PeftModel.from_pretrained(model, LORA_DIR, is_trainable=False)
else:
    print(f"No LoRA dir found at '{LORA_DIR}', running base model only.")

model.eval()
if DEVICE == "cuda":
    model.half()  

# ---------- utility helpers ----------
def build_prompt(user_text: str) -> str:
    """
    Build a small instruction/response template that the instruction-tuned model expects.
    Adjust if you used a different training prompt format.
    """
    prompt = (
        "Instruction: " + user_text.strip() + "\n"
        "Response:"
    )
    return prompt

def post_process(text: str) -> str:
    
    lowest_idx = None
    for s in STOP_STRS:
        idx = text.find(s)
        if idx != -1:
            if lowest_idx is None or idx < lowest_idx:
                lowest_idx = idx
    if lowest_idx is not None:
        text = text[:lowest_idx].strip()
    # trim weird repeated tokens at end
    text = text.rstrip(" \n")
    return text

# ---------- interactive loop ----------
print("\nReady. Type your prompt and press Enter. Type 'exit' to quit.\n")
try:
    while True:
        user_input = input("User: ")
        if not user_input:
            continue
        if user_input.lower().strip() in ("exit", "quit"):
            print("Exiting.")
            break

        prompt = build_prompt(user_input)

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        # Generation config
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        # generate
        with torch.no_grad():
            out = model.generate(**gen_kwargs)

        # decode
        full_output = tokenizer.decode(out[0], skip_special_tokens=True)
       
        generated_part = full_output[len(prompt):].strip() if full_output.startswith(prompt) else full_output.strip()
        cleaned = post_process(generated_part)

        
        print(cleaned)
        print("\n------------------\n")

except KeyboardInterrupt:
    print("\nInterrupted. Bye!")
