#!/usr/bin/env python3
"""
qwen_vl_sft_train.py
Supervised fine-tuning (SFT) of Qwen3-VL-8B-Instruct on a vision QA dataset using:
 - bitsandbytes 4-bit quantization
 - PEFT LoRA (QLoRA style)
 - TRL's SFTTrainer

Adapted from: Hugging Face Open-Source AI Cookbook (Fine-Tuning a Vision Language Model (Qwen3-VL-8B) with TRL).
(See cookbook for more background and optional tracking integration.)
"""

# pip installs (run once in your environment / notebook):
# pip install -U git+https://github.com/huggingface/trl.git bitsandbytes peft qwen-vl-utils trackio transformers datasets

import os
import gc
import time
import torch
from datasets import load_dataset
from transformers import BitsAndBytesConfig
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

# TRL and PEFT imports
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model

# utility for processing images as used in the cookbook
from qwen_vl_utils import process_vision_info

# -----------------------------
# Config - change to suit your run
# -----------------------------
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DATASET_ID = "HuggingFaceM4/ChartQA"   # example dataset from cookbook
OUTPUT_DIR = "qwen3-2b-instruct-trl-sft-ChartQA"
NUM_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Helper: clear GPU memory
# -----------------------------
def clear_memory():
    for name in ["inputs", "model", "processor", "trainer", "peft_model", "bnb_config"]:
        if name in globals():
            del globals()[name]
    time.sleep(1)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(1)

# -----------------------------
# 1) Load dataset & format as Chat-style + images
# -----------------------------
system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

def format_data(sample):
    # Each example is a "chat" with system -> user (image+query) -> assistant (answer)
    return {
        "images": [sample["image"]],
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {"role": "user", "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": sample["query"]}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": sample["label"][0]}]}
        ]
    }

print("Loading dataset (this may take a while)...")
train_raw, eval_raw, test_raw = load_dataset(DATASET_ID, split=['train[:10%]', 'val[:10%]', 'test[:10%]'])
train_dataset = [format_data(s) for s in train_raw]
eval_dataset = [format_data(s) for s in eval_raw]
test_dataset = [format_data(s) for s in test_raw]
print(f"train/eval/test sizes: {len(train_dataset)}/{len(eval_dataset)}/{len(test_dataset)}")

# -----------------------------
# 2) Load quantized model + processor (BitsAndBytes int4)
# -----------------------------
print("Preparing BitsAndBytes quantization config and loading the model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
)

processor = Qwen3VLProcessor.from_pretrained(MODEL_ID)

# -----------------------------
# 3) Apply LoRA (QLoRA-style) via PEFT
# -----------------------------
print("Applying LoRA PEFT adapters (QLoRA-style settings)...")
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj"],   # recommended target modules for Qwen VLM in cookbook
    task_type="CAUSAL_LM",
)

peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()

# -----------------------------
# 4) SFT training args (TRL)
# -----------------------------
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_length=None,
    optim="adamw_torch_fused",
    learning_rate=LEARNING_RATE,
    logging_steps=10,
    eval_steps=10,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=20,
    bf16=True,  # use bfloat16 if available (A100 / H100)
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    push_to_hub=True,         # optional: push adapter to Hub when saving
    report_to="trackio",      # optional: replace/remove if not using trackio
)

# Optional: initialize trackio / W&B here if desired (not required). See cookbook.

# -----------------------------
# 5) Build SFTTrainer and train
# -----------------------------
print("Creating SFTTrainer...")
trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    processing_class=processor,  # let SFTTrainer use DataCollatorForVisionLanguageModeling internally
)

print("Starting training...")
trainer.train()

print("Saving the trained adapter / model...")
trainer.save_model(training_args.output_dir)

# -----------------------------
# 6) Test: reload base model + attach adapter for inference
# -----------------------------
clear_memory()

print("Reloading base model and processor for inference...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
processor = Qwen3VLProcessor.from_pretrained(MODEL_ID)

# Load adapter (the saved PEFT adapter) and attach
adapter_path = OUTPUT_DIR  # if you pushed to hub, you can set "username/repo" path
try:
    model.load_adapter(adapter_path)
    print(f"Loaded adapter from {adapter_path}")
except Exception as e:
    print(f"Could not directly load adapter from {adapter_path}: {e}")
    print("If you pushed the adapter to the Hub, use model.load_adapter('username/repo').")

# helper generate function (same as cookbook)
def generate_text_from_sample(model, processor, sample, max_new_tokens=128, device="cuda"):
    # Preprocess text (apply chat template)
    text_input = processor.apply_chat_template(
        sample['messages'][1:2],  # the user part (image+query) usually
        tokenize=False,
        add_generation_prompt=True
    )
    # Process visual input
    image_inputs, _ = process_vision_info(sample['messages'])
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    trimmed_generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return output_text[0]

# quick inference test
print("Running a quick inference on a sample from the test set...")
sample = test_dataset[0]
out = generate_text_from_sample(model, processor, sample)
print("Model output:", out)
print("Reference answer:", sample['messages'][-1]['content'][0]['text'])

print("Done.")
