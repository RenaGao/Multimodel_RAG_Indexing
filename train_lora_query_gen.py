# -*- coding: utf-8 -*-
"""
Full pipeline: 
1. Load and parse JSON dataset
2. Data processing and tokenization (concatenate prompt + target, robust)
3. LoRA fine-tuning (32-bit)
4. Save LoRA weights
5. Inference to generate user queries
6. Record training time & TensorBoard logging
"""

import json
import time
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split


def finetune_LLM_with_lora(data_file="./data/dataset_1-10.json", model_name="facebook/opt-1.3b"):
    # =====================
    # Load and parse dataset
    # =====================

    with open(data_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    train_texts, train_labels = [], []
    pairs = []

    for item in raw_data:
        try:
            input_obj = json.loads(item.get("input", "{}"))
            title = input_obj.get("product title", "")
            document = input_obj.get("document", "")
            doc_text = " ".join(document) if isinstance(document, list) else str(document)

            # Get the query/question
            query = input_obj.get("query") or input_obj.get("question")

            # Filter data that have no query or title/doc_text
            if not query or not (title or doc_text):
                continue

            # Construct the prompt
            input_text = (
                f"Product Title: {title}\n"
                f"Product Description: {doc_text}\n"
                f"Task: Generate a likely user search query or question.\nOutput:"
            )
            train_texts.append(input_text)
            train_labels.append(query)

            # Save pair for reference
            pairs.append({"input_text": input_text, "query": query})

        except Exception:
            continue


    # Save pairs to JSONL file
    with open("pairs.jsonl", "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")


    # Split into train/validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.05, random_state=42
    )

    # =====================
    # Tokenizer
    # =====================
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # crucial for padding

    max_input_length = 512
    max_output_length = 64
    total_max_length = max_input_length + max_output_length

    # =====================
    # Robust tokenization
    # =====================
    def tokenize_fn(examples):
        input_texts = examples["input_text"]
        target_texts = examples["target_text"]

        full_texts = [inp + " " + tgt for inp, tgt in zip(input_texts, target_texts)]
        tokenized = tokenizer(
            full_texts,
            max_length=total_max_length,
            truncation=True,
            padding="max_length"
        )

        # Mask prompt part in labels
        labels = torch.tensor(tokenized["input_ids"]).clone()
        for i, inp in enumerate(input_texts):
            input_ids = tokenizer(inp, add_special_tokens=False)["input_ids"]
            labels[i, :len(input_ids)] = -100
        tokenized["labels"] = labels.tolist()
        return tokenized

    train_dataset = Dataset.from_dict({"input_text": train_texts, "target_text": train_labels})
    val_dataset = Dataset.from_dict({"input_text": val_texts, "target_text": val_labels})

    tokenized_train = train_dataset.map(tokenize_fn, batched=True)
    tokenized_val = val_dataset.map(tokenize_fn, batched=True)

    dataset = DatasetDict({"train": tokenized_train, "validation": tokenized_val})

    # =====================
    # Load model and apply LoRA
    # =====================
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    # =====================
    # Training arguments with TensorBoard logging
    # =====================
    training_args = TrainingArguments(
        output_dir="./lora_query_gen",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=False,
        optim="adamw_torch",
        warmup_steps=50,
        report_to="tensorboard",  # create TensorBoard
        logging_dir="./logs"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"]
    )

    # =====================
    # Start training & record time
    # =====================
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Training completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

    # =====================
    # Save LoRA weights
    # =====================
    model.save_pretrained("./lora_query_gen")
    tokenizer.save_pretrained("./lora_query_gen")

    return model, tokenizer


def generate_query(model, tokenizer, product_title, product_doc, max_new_tokens=32):
    prompt = f"Product Title: {product_title}\nProduct Description: {product_doc}\nTask: Generate a likely user search query.\nOutput:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return query


def main():
    # Fine-tune model
    model, tokenizer = finetune_LLM_with_lora()

    # Example inference
    example_title = "Neutrogena Anti-Residue Clarifying Shampoo"
    example_doc = "Gentle clarifying shampoo to remove hair build-up."
    generated_query = generate_query(model, tokenizer, example_title, example_doc)
    print("Generated query:", generated_query)


if __name__ == "__main__":
    main()
