#!/usr/bin/env python3
"""
Target Product Accuracy Evaluation Script

Evaluates whether assistant responses in multi-turn dialogues mention the target product.
Uses token overlap ratio to determine if the target product title appears in the response.
"""

import argparse
import csv
import json
import os
import re
import string
from collections import defaultdict
from pathlib import Path


# Stopwords for tokenization
STOPWORDS = {
    "the", "a", "an", "of", "for", "to", "and", "in", "on", "with",
    "by", "this", "that", "is", "are", "it", "you", "your", "my",
    "i", "we", "they", "their", "our"
}

# Default threshold for token overlap
DEFAULT_THRESHOLD = 0.3

# Method directories to evaluate
METHODS = [
    "baseline_dialogues_output",
    "multimdoal_no_user_review_output",
    "multimodal_dialogues_output",
    "textrag_dialogues_output"
]


def normalize_and_tokenize(text: str) -> list:
    """
    Normalize text and tokenize into words.
    - Convert to lowercase
    - Replace punctuation with spaces
    - Split on whitespace
    - Remove stopwords
    """
    if text is None:
        return []
    text = text.lower()
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    tokens = text.split()
    tokens = [tok for tok in tokens if tok and tok not in STOPWORDS]
    return tokens


def token_overlap_ratio(gold_title: str, pred_text: str) -> float:
    """
    Compute the ratio of gold title tokens that appear in the prediction text.
    """
    gold_tokens = set(normalize_and_tokenize(gold_title))
    pred_tokens = set(normalize_and_tokenize(pred_text))

    if not gold_tokens:
        return 0.0

    inter = gold_tokens & pred_tokens
    return len(inter) / len(gold_tokens)


def load_questions_mapping(questions_dir: str) -> dict:
    """
    Load all question files and build a mapping from user_id to target product title.

    Returns:
        dict: {user_id: target_product_title}
    """
    mapping = {}
    questions_path = Path(questions_dir)

    for json_file in questions_path.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for item in data:
                user_id = item.get("user_id")
                target_title = item.get("target_product", {}).get("product_meta", {}).get("title")

                if user_id and target_title:
                    mapping[user_id] = target_title
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Error reading {json_file}: {e}")
            continue

    return mapping


def extract_assistant_turns(dialogue_history: list) -> list:
    """
    Extract assistant responses from dialogue history.

    Returns:
        list: [(turn_id, content), ...] where turn_id is 1-indexed
    """
    assistant_turns = []
    turn_id = 0

    for turn in dialogue_history:
        if turn.get("role") == "assistant":
            turn_id += 1
            assistant_turns.append((turn_id, turn.get("content", "")))

    return assistant_turns


def evaluate_single_dialogue(dialogue: dict, target_title: str, threshold: float) -> list:
    """
    Evaluate a single dialogue against the target product.

    Returns:
        list: [{turn_id, overlap, hit}, ...]
    """
    results = []
    dialogue_history = dialogue.get("dialogue_history", [])
    assistant_turns = extract_assistant_turns(dialogue_history)

    for turn_id, content in assistant_turns:
        overlap = token_overlap_ratio(target_title, content)
        hit = 1 if overlap >= threshold else 0
        results.append({
            "turn_id": turn_id,
            "overlap": overlap,
            "hit": hit
        })

    return results


def evaluate_method(method_dir: Path, questions_mapping: dict, threshold: float) -> tuple:
    """
    Evaluate all dialogues for a single method.

    Returns:
        tuple: (detailed_results, per_turn_stats, overall_stats)
    """
    detailed_results = []
    per_turn_hits = defaultdict(int)
    per_turn_total = defaultdict(int)
    total_hits = 0
    total_turns = 0
    matched_dialogues = 0
    unmatched_dialogues = 0

    # Iterate through all category directories
    for category_dir in method_dir.iterdir():
        if not category_dir.is_dir():
            continue

        category_name = category_dir.name

        # Process only .json files (not .jsonl)
        for json_file in category_dir.glob("*.json"):
            if json_file.suffix != ".json":
                continue

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Handle both list and single object formats
                if isinstance(data, list):
                    dialogues = data
                else:
                    dialogues = [data]

                for dialogue in dialogues:
                    dialogue_id = dialogue.get("dialogue_id", "")

                    # Handle dialogue_id with suffix (e.g., "ABC123_1")
                    base_dialogue_id = dialogue_id.split("_")[0] if "_" in dialogue_id and dialogue_id.split("_")[-1].isdigit() else dialogue_id

                    # Try to match with questions mapping
                    target_title = questions_mapping.get(dialogue_id) or questions_mapping.get(base_dialogue_id)

                    if not target_title:
                        unmatched_dialogues += 1
                        continue

                    matched_dialogues += 1
                    turn_results = evaluate_single_dialogue(dialogue, target_title, threshold)

                    for result in turn_results:
                        detailed_results.append({
                            "method": method_dir.name,
                            "category": category_name,
                            "dialogue_id": dialogue_id,
                            "turn_id": result["turn_id"],
                            "gold_title": target_title,
                            "overlap": result["overlap"],
                            "hit": result["hit"]
                        })

                        per_turn_hits[result["turn_id"]] += result["hit"]
                        per_turn_total[result["turn_id"]] += 1
                        total_hits += result["hit"]
                        total_turns += 1

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Error reading {json_file}: {e}")
                continue

    # Calculate per-turn accuracy
    per_turn_acc = {}
    for turn_id in sorted(per_turn_total.keys()):
        if per_turn_total[turn_id] > 0:
            per_turn_acc[turn_id] = per_turn_hits[turn_id] / per_turn_total[turn_id]
        else:
            per_turn_acc[turn_id] = 0.0

    overall_acc = total_hits / total_turns if total_turns > 0 else 0.0

    stats = {
        "per_turn_acc": per_turn_acc,
        "overall_acc": overall_acc,
        "total_hits": total_hits,
        "total_turns": total_turns,
        "matched_dialogues": matched_dialogues,
        "unmatched_dialogues": unmatched_dialogues
    }

    return detailed_results, stats


def print_summary(all_stats: dict, methods: list):
    """Print summary table to console."""
    print("\n" + "=" * 100)
    print("TARGET PRODUCT ACCURACY EVALUATION RESULTS")
    print("=" * 100)

    # Find max turn number across all methods
    max_turn = 0
    for method in methods:
        if method in all_stats:
            turns = all_stats[method]["per_turn_acc"].keys()
            if turns:
                max_turn = max(max_turn, max(turns))

    # Print header
    header = f"{'Method':<40}"
    for t in range(1, max_turn + 1):
        header += f"{'T' + str(t):>8}"
    header += f"{'Overall':>10}{'Dialogues':>12}"
    print(header)
    print("-" * 100)

    # Print each method's results
    for method in methods:
        if method not in all_stats:
            continue

        stats = all_stats[method]
        row = f"{method:<40}"

        for t in range(1, max_turn + 1):
            acc = stats["per_turn_acc"].get(t, 0.0)
            row += f"{acc:>8.3f}"

        row += f"{stats['overall_acc']:>10.3f}"
        row += f"{stats['matched_dialogues']:>12}"
        print(row)

    print("=" * 100)

    # Print additional info
    for method in methods:
        if method in all_stats:
            stats = all_stats[method]
            print(f"\n{method}:")
            print(f"  Matched dialogues: {stats['matched_dialogues']}")
            print(f"  Unmatched dialogues: {stats['unmatched_dialogues']}")
            print(f"  Total turns evaluated: {stats['total_turns']}")
            print(f"  Total hits: {stats['total_hits']}")


def save_csv(all_results: list, output_file: str):
    """Save detailed results to CSV."""
    if not all_results:
        print("No results to save.")
        return

    fieldnames = ["method", "category", "dialogue_id", "turn_id", "gold_title", "overlap", "hit"]

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nDetailed results saved to: {output_file}")


def save_summary_csv(all_stats: dict, methods: list, output_file: str):
    """Save summary statistics to CSV."""
    if not all_stats:
        print("No summary to save.")
        return

    # Find max turn number across all methods
    max_turn = 0
    for method in methods:
        if method in all_stats:
            turns = all_stats[method]["per_turn_acc"].keys()
            if turns:
                max_turn = max(max_turn, max(turns))

    # Build fieldnames
    fieldnames = ["method"]
    for t in range(1, max_turn + 1):
        fieldnames.append(f"turn_{t}_acc")
    fieldnames.extend(["overall_acc", "total_hits", "total_turns", "matched_dialogues", "unmatched_dialogues"])

    rows = []
    for method in methods:
        if method not in all_stats:
            continue

        stats = all_stats[method]
        row = {"method": method}

        for t in range(1, max_turn + 1):
            row[f"turn_{t}_acc"] = round(stats["per_turn_acc"].get(t, 0.0), 4)

        row["overall_acc"] = round(stats["overall_acc"], 4)
        row["total_hits"] = stats["total_hits"]
        row["total_turns"] = stats["total_turns"]
        row["matched_dialogues"] = stats["matched_dialogues"]
        row["unmatched_dialogues"] = stats["unmatched_dialogues"]
        rows.append(row)

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summary results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate target product accuracy for multi-turn dialogues"
    )
    parser.add_argument(
        "--dialogue-dir",
        type=str,
        default=None,
        help="Path to dialogue_outputs directory"
    )
    parser.add_argument(
        "--questions-dir",
        type=str,
        default=None,
        help="Path to questions_v3 directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="target_product_acc_results.csv",
        help="Output CSV file path for detailed results"
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default="target_product_acc_summary.csv",
        help="Output CSV file path for summary results"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Token overlap threshold for hit (default: {DEFAULT_THRESHOLD})"
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=METHODS,
        help="Methods to evaluate"
    )

    args = parser.parse_args()

    # Determine base directory
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]

    dialogue_dir = Path(args.dialogue_dir) if args.dialogue_dir else script_dir / "dialogue_outputs"
    questions_dir = Path(args.questions_dir) if args.questions_dir else repo_root / "data" / "questions_v3"

    print(f"Dialogue outputs directory: {dialogue_dir}")
    print(f"Questions directory: {questions_dir}")
    print(f"Threshold: {args.threshold}")

    # Load questions mapping
    print("\nLoading questions mapping...")
    questions_mapping = load_questions_mapping(str(questions_dir))
    print(f"Loaded {len(questions_mapping)} user-to-target mappings")

    # Evaluate each method
    all_results = []
    all_stats = {}

    for method in args.methods:
        method_path = dialogue_dir / method
        if not method_path.exists():
            print(f"\nWarning: Method directory not found: {method_path}")
            continue

        print(f"\nEvaluating {method}...")
        results, stats = evaluate_method(method_path, questions_mapping, args.threshold)
        all_results.extend(results)
        all_stats[method] = stats

    # Print summary
    print_summary(all_stats, args.methods)

    # Save detailed results
    output_path = script_dir / args.output if not os.path.isabs(args.output) else Path(args.output)
    save_csv(all_results, str(output_path))

    # Save summary results
    summary_path = script_dir / args.summary_output if not os.path.isabs(args.summary_output) else Path(args.summary_output)
    save_summary_csv(all_stats, args.methods, str(summary_path))


if __name__ == "__main__":
    main()
