"""
One-time script to split eval_results by rag_enabled field.
- baseline_eval/: rag_enabled = false
- textrag_eval/: rag_enabled = true
"""
import json
import os
import glob

# Hardcoded paths
INPUT_DIR = "eval_results"
BASELINE_DIR = os.path.join(INPUT_DIR, "baseline_eval")
TEXTRAG_DIR = os.path.join(INPUT_DIR, "textrag_eval")


def main():
    os.makedirs(BASELINE_DIR, exist_ok=True)
    os.makedirs(TEXTRAG_DIR, exist_ok=True)

    eval_files = glob.glob(os.path.join(INPUT_DIR, "*_eval.json"))

    total_baseline = 0
    total_textrag = 0

    for file_path in sorted(eval_files):
        filename = os.path.basename(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        baseline = [d for d in data if not d.get("rag_enabled", False)]
        textrag = [d for d in data if d.get("rag_enabled", False)]

        if baseline:
            with open(os.path.join(BASELINE_DIR, filename), "w", encoding="utf-8") as f:
                json.dump(baseline, f, ensure_ascii=False, indent=2)

        if textrag:
            with open(os.path.join(TEXTRAG_DIR, filename), "w", encoding="utf-8") as f:
                json.dump(textrag, f, ensure_ascii=False, indent=2)

        total_baseline += len(baseline)
        total_textrag += len(textrag)
        print(f"✓ {filename}: baseline={len(baseline)}, textrag={len(textrag)}")

    print(f"\n📊 Total: baseline={total_baseline}, textrag={total_textrag}")
    print(f"💾 Saved to: {BASELINE_DIR}/ and {TEXTRAG_DIR}/")


if __name__ == "__main__":
    main()
