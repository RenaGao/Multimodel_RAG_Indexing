#!/usr/bin/env python3
"""
qwen_vl_sft_train_multimodal.py
Adapted from original Qwen3-VL SFT training script to work with custom multi-modal dataset:
- JSON fields: "doc_text", "images"
- Uses TRL SFTTrainer + PEFT LoRA + BitsAndBytes 4-bit quantization
"""

import os
import gc
import time
import torch
from datasets import load_dataset
from transformers import BitsAndBytesConfig
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model

from qwen_vl_utils import process_vision_info
from PIL import Image
import requests
from torchvision import transforms

# -----------------------------
# Config - change to suit your run
# -----------------------------
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DATASET_PATH = "fine_tuning_dataset.jsonl"
OUTPUT_DIR = "qwen3-2b-instruct-trl-sft-multimodal"
NUM_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Clear GPU memory
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
# Data preprocessing
# -----------------------------
system_message = """
You are a personalized conversational agent specialized in online shopping.
Your role is to simulate realistic customer interactions by generating and responding to natural, diverse shopper questions across various product categories.

Output responses in this format:
{
  "response": "...",
  "reasoning": "...",
  "follow_up": "..."
}
"""

processor = Qwen3VLProcessor.from_pretrained(MODEL_ID)


def merge_images(images):
    """Merge multiple PIL images into a single wide image"""
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_img = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width
    return new_img


def format_multimodal_sample(sample):
    images = []

    print(sample["images"])

    for url in sample["images"]:
        try:
            img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"Failed to load image {url}: {e}")

    if not images:
        raise ValueError("No valid images found for sample!")
    
    print(images)

    # # Pass images and text directly to processor for multimodal input
    # processed_inputs = processor(
    #     text=sample["doc_text"],   # pass text here
    #     images=images,
    #     return_tensors="pt"
    # )

    # Merge multiple images into one
    merged_img = merge_images(images)

    return {
        "images": [merged_img],
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {"role": "user", "content": [
                {"type": "text", "text": sample["doc_text"]},
                {"type": "image", "image": merged_img}  # keep PIL images here
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": sample["query"]}]}
        ]
    }


print("Loading dataset...")
dataset = load_dataset("json", data_files=DATASET_PATH)

# Split into train/val/test (example 80/10/10)
raw_dataset = dataset['train']
n_total = len(raw_dataset)
n_train = int(0.8 * n_total)
n_val = int(0.1 * n_total)

print("raw data =", raw_dataset)
print("n_total =", n_total)
print("n_train =", n_train)
print("n_val =", n_val)

raw_dataset_list = raw_dataset.to_dict()  # converts to dict of lists

# convert to list of samples
raw_dataset_samples = [
    {k: v[i] for k, v in raw_dataset_list.items()} 
    for i in range(len(raw_dataset))
]

train_dataset = [format_multimodal_sample(s) for s in raw_dataset_samples[:n_train]]
eval_dataset = [format_multimodal_sample(s) for s in raw_dataset_samples[n_train:n_train+n_val]]
test_dataset = [format_multimodal_sample(s) for s in raw_dataset_samples[n_train+n_val:]]
print(f"train/eval/test sizes: {len(train_dataset)}/{len(eval_dataset)}/{len(test_dataset)}")

# -----------------------------
# Load quantized model + processor
# -----------------------------
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
# Apply LoRA
# -----------------------------
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()

# -----------------------------
# SFT training args
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
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    push_to_hub=False,
    report_to=None,
)

# -----------------------------
# Build trainer and train
# -----------------------------
trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    processing_class=processor,
)

print("Starting training...")
trainer.train()
trainer.save_model(training_args.output_dir)

# -----------------------------
# Inference test
# -----------------------------
clear_memory()
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
processor = Qwen3VLProcessor.from_pretrained(MODEL_ID)
# Attach trained adapter
peft_model = get_peft_model(model, peft_config)
peft_model.load_adapter(model_id=OUTPUT_DIR, adapter_name="default")

def generate_from_multimodal_sample(model, processor, sample, max_new_tokens=128, device="cuda"):
    text_input = processor.apply_chat_template(
        sample['messages'][1:2],
        tokenize=False,
        add_generation_prompt=True
    )
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

# Quick inference
sample = test_dataset[0]
out = generate_from_multimodal_sample(peft_model, processor, sample)
print("Model output:", out)
print("Reference answer:", sample['messages'][-1]['content'][0]['text'])
print("Done.")
