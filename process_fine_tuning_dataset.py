import json

data_file = "./data/dataset_1-10000.json"  # 替换成你的原始数据文件路径
output_file = "fine_tuning_dataset.jsonl"

with open(data_file, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

processed_data = []

for item in raw_data:
    try:
        input_obj = json.loads(item.get("input", "{}"))
        title = input_obj.get("product title", "")
        document = input_obj.get("document", "")
        doc_text = title
        if isinstance(document, list):
            doc_text += " " + " ".join(document)
        elif isinstance(document, str):
            doc_text += " " + document

        query = input_obj.get("query") or input_obj.get("question")
        images = item.get("images", [])

        # flatten nested lists and remove empty strings/lists
        flat_images = []
        for img in images:
            if isinstance(img, list):
                flat_images.extend([i for i in img if i])
            elif img:
                flat_images.append(img)
        images = flat_images

        # skip if no query or doc_text
        if not query or not doc_text.strip():
            continue

        processed_data.append({
            "doc_text": doc_text.strip(),
            "images": images,
            "query": query
        })

    except Exception as e:
        continue

# save to JSONL file
with open(output_file, "w", encoding="utf-8") as f:
    for sample in processed_data:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"Processed {len(processed_data)} samples.")
