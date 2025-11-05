# -*- coding: utf-8 -*-
"""
Full pipeline for Qwen3-VL-2B-Instruct with LoRA fine-tuning
- Handles missing images safely
- Ensures <image> token alignment
- Fixes 'cannot reshape tensor of 0 elements' errors
- Adds robust image feature handling
"""

import json
import time
import torch
import requests
from io import BytesIO
from PIL import Image
from datasets import Dataset, DatasetDict
from transformers import AutoProcessor, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from torch.nn.utils.rnn import pad_sequence
import numpy as np

# ------------------------------
# Utilities
# ------------------------------

def load_image_from_url_or_path(path_or_url: str):
    path_or_url = path_or_url.strip()
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        try:
            response = requests.get(path_or_url, timeout=5)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            print(f"Error loading image from URL {path_or_url}: {str(e)}")
            return None
    else:
        try:
            return Image.open(path_or_url).convert("RGB")
        except Exception as e:
            print(f"Error loading image from path {path_or_url}: {str(e)}")
            return None

def dummy_image(size=(224, 224)):
    """Returns a black RGB image for missing image cases."""
    return Image.new("RGB", size, color=(0, 0, 0))

# ------------------------------
# Tokenization Function (FIXED)
# ------------------------------

def tokenize_fn(example, tokenizer, processor, max_length=1024):
    text = example["prompt"]
    target = example["target"]
    image_path = example.get("image", None)

    # Always include image when <image> token exists in prompt
    image_obj = None
    has_image_token = "<image>" in text
    
    if has_image_token:
        if image_path and image_path.lower() != "dummy":
            image_obj = load_image_from_url_or_path(image_path.strip())
        if image_obj is None:
            image_obj = dummy_image()  # Fallback to dummy image

    # Process inputs - ALWAYS include image features when image token exists
    if has_image_token and image_obj is not None:
        model_inputs = processor(
            text=[text],
            images=[image_obj],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        # CRITICAL FIX: Don't remove image_grid_thw - model requires it when images exist
        image_grid_thw = model_inputs.get("image_grid_thw", None)
        if image_grid_thw is None:
            # Fallback grid_thw for dummy images (1 frame, 7x7 grid - common for ViT)
            image_grid_thw = torch.tensor([[1, 7, 7]], dtype=torch.long)
    else:
        model_inputs = processor(
            text=[text],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        image_grid_thw = None  # Only None when NO image token exists

    # Prepare labels
    input_ids = model_inputs["input_ids"].clone()
    prompt_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    prompt_len = len(prompt_ids)
    
    labels = torch.full_like(input_ids, -100)
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    target_len = min(len(target_ids), max_length - prompt_len)
    
    if target_len > 0:
        labels[0, prompt_len:prompt_len + target_len] = torch.tensor(target_ids[:target_len])

    model_inputs["labels"] = labels
    # Only include image_grid_thw when we have image features
    if has_image_token:
        model_inputs["image_grid_thw"] = image_grid_thw

    # Convert tensors to lists for Arrow dataset
    output = {}
    for k, v in model_inputs.items():
        if v is not None:
            if isinstance(v, torch.Tensor):
                output[k] = v.squeeze(0).tolist()
            else:
                output[k] = v
    return output

# ------------------------------
# Custom collator (FIXED)
# ------------------------------

def collate_fn(batch):
    # Separate examples with and without images
    with_image = [x for x in batch if "pixel_values" in x and x["pixel_values"] is not None]
    without_image = [x for x in batch if "pixel_values" not in x or x["pixel_values"] is None]
    
    # Process input_ids and labels for all examples
    input_ids = [torch.tensor(x["input_ids"]) for x in batch]
    labels = [torch.tensor(x["labels"]) for x in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    # Initialize outputs
    pixel_values = None
    image_grid_thw = None
    
    # Only process image features if we have examples with images
    if with_image:
        # Stack pixel values
        pixel_values_list = [torch.tensor(x["pixel_values"]) for x in with_image]
        pixel_values = torch.stack(pixel_values_list)
        
        # Stack image_grid_thw with fallback
        grid_thw_list = []
        for x in with_image:
            thw = x.get("image_grid_thw")
            if thw is None:
                grid_thw_list.append(torch.tensor([1, 7, 7], dtype=torch.long))
            else:
                grid_thw_list.append(torch.tensor(thw))
        image_grid_thw = torch.stack(grid_thw_list)
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw
    }

# ------------------------------
# Fine-tuning Function
# ------------------------------

def finetune_QwenVL_with_lora(data_file="./data/dataset.json",
                               model_name="Qwen/Qwen3-VL-2B-Instruct"):

    # Load dataset
    with open(data_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    prompts, targets, image_paths = [], [], []

    for item in raw_data:
        try:
            input_obj = json.loads(item["input"])
            product_title = input_obj.get("product title", "")
            product_id = input_obj.get("product id", "")
            document = input_obj.get("document", [])
            doc_text = " ".join(document) if isinstance(document, list) else str(document)
            query = input_obj.get("query") or input_obj.get("question", "")
            if not query:
                continue
            images = item.get("images", [[]])
            image_url = images[0][0].strip() if images and images[0] else "dummy"
            if not (product_title or doc_text):
                continue

            # Construct prompt with <image>
            prompt = (
                "You are an e-commerce assistant that generates realistic user queries based on product information and images.\n"
                "Product Title: {product_title}\n"
                "Product ID: {product_id}\n"
                "Description: {doc_text}\n"
                "Generate a realistic search query or question a customer might use to find this product."
            ).format(
                product_title=product_title,
                product_id=product_id,
                doc_text=doc_text[:500]
            )
            
            # Modern Qwen-VL uses <|vision_start|> and <|vision_end|> tokens
            prompt = (
                "<|im_start|>system\n"
                "You are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n"
                "<|vision_start|><|image_pad|><|vision_end|>\n"
                f"{prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

            prompts.append(prompt)
            targets.append(query)
            image_paths.append(image_url)

        except Exception as e:
            print(f"Error processing item: {str(e)}")
            continue

    # Train/validation split
    train_prompts, val_prompts, train_targets, val_targets, train_images, val_images = train_test_split(
        prompts, targets, image_paths, test_size=0.1, random_state=42
    )

    print(f"Training samples: {len(train_prompts)}, Validation samples: {len(val_prompts)}")

    # Load processor & tokenizer
    global processor
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer
    max_length = 1024

    # Prepare datasets
    train_dataset = Dataset.from_dict({
        "prompt": train_prompts,
        "target": train_targets,
        "image": train_images
    })
    val_dataset = Dataset.from_dict({
        "prompt": val_prompts,
        "target": val_targets,
        "image": val_images
    })

    # Tokenize with batched=False to handle images safely
    tokenized_train = train_dataset.map(
        lambda ex: tokenize_fn(ex, tokenizer, processor, max_length=max_length),
        batched=False,
        remove_columns=train_dataset.column_names
    )
    tokenized_val = val_dataset.map(
        lambda ex: tokenize_fn(ex, tokenizer, processor, max_length=max_length),
        batched=False,
        remove_columns=val_dataset.column_names
    )
    dataset = DatasetDict({"train": tokenized_train, "validation": tokenized_val})

    # ----------------------
    # Load Qwen3-VL model with LoRA
    # ----------------------
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./qwenvl_lora_finetune",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        num_train_epochs=3,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        logging_dir="./logs",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    # Custom trainer for multimodal inputs
    class VLTrainer(Trainer):
        def _prepare_inputs(self, inputs):
            if "pixel_values" in inputs and inputs["pixel_values"] is not None:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.args.device)
            if "image_grid_thw" in inputs and inputs["image_grid_thw"] is not None:
                inputs["image_grid_thw"] = inputs["image_grid_thw"].to(self.args.device)
            return super()._prepare_inputs(inputs)

    trainer = VLTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collate_fn,
        tokenizer=processor.tokenizer
    )

    # Train
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    print(f"Training completed in {training_time/60:.2f} minutes")

    # Save artifacts
    model.save_pretrained("./qwenvl_lora_finetune")
    processor.save_pretrained("./qwenvl_lora_finetune")
    with open("./training_metrics.json", "w") as f:
        json.dump({
            "training_time_minutes": training_time / 60,
            "total_samples": len(train_prompts) + len(val_prompts),
            "final_loss": trainer.state.log_history[-1].get("train_loss", None)
        }, f)

    return model, processor

# ------------------------------
# Query generation
# ------------------------------

def generate_query(model, processor, product_title, product_doc, image_url, max_new_tokens=50):
    """
    Generate a realistic search query using Qwen3-VL with proper vision token alignment.
    Handles missing images safely with dummy images.
    """
    image_url = image_url.strip()

    # Correct prompt with vision tokens
    prompt = (
        "<|im_start|>system\n"
        "You are an e-commerce assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>\n"
        f"Product Title: {product_title}\n"
        f"Description: {product_doc[:300]}...\n"
        "Generate a realistic search query.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    # Load image or fallback to dummy
    image_obj = load_image_from_url_or_path(image_url)
    if image_obj is None:
        image_obj = dummy_image()

    # Process inputs properly for multimodal model
    inputs = processor(
        text=[prompt],
        images=[image_obj],
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # Move to model device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate output
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=processor.tokenizer.pad_token_id
    )

    # Decode output
    generated_text = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract assistant response only
    if "<|im_start|>assistant\n" in generated_text:
        generated_text = generated_text.split("<|im_start|>assistant\n")[-1].strip()

    return generated_text

# ------------------------------
# Main
# ------------------------------

def main():
    model, processor = finetune_QwenVL_with_lora(
        data_file="./data/dataset_1-10000.json",
        model_name="Qwen/Qwen3-VL-2B-Instruct"
    )

    example_title = "Tresemme Shampoo Anti-Breakage 28 Ounce (828ml) (2 Pack)"
    example_doc = "Professional hair care for damaged hair. Strengthens strands and reduces breakage with keratin protein."
    example_image = "https://m.media-amazon.com/images/I/512k8O-UA5L.jpg"

    generated_query = generate_query(
        model,
        processor,
        example_title,
        example_doc,
        example_image
    )
    print("Generated query:", generated_query)

if __name__ == "__main__":
    main()
