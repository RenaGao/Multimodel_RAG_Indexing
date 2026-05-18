"""
summarize_eval_results.py
-----------------------------------
Summarize and compare evaluation scores across different RAG configurations.

Usage:
    python summarize_eval_results.py --output-dir eval_summary
"""

import json
import os
import argparse
import glob
from collections import defaultdict

# -----------------------------------
# Configuration
# -----------------------------------
EVAL_DIRS = {
    "baseline": "eval_results/baseline_eval",
    "textrag": "eval_results/textrag_eval",
    "multimodal_no_review": "eval_results/multimodal_no_user_review_eval",
    "multimodalrag": "eval_results/multimodalrag_eval"
}

METRICS = [
    "GlobalPersonalizationConsistency",
    "CumulativeInformationSynthesis",
    "InteractionTrajectory",
    "ToneConsistency",
    "OverallScore"
]

METRIC_SHORT_NAMES = {
    "GlobalPersonalizationConsistency": "GPC",
    "CumulativeInformationSynthesis": "CIS",
    "InteractionTrajectory": "IT",
    "ToneConsistency": "TC",
    "OverallScore": "Overall"
}

PAIRWISE_DIRS = {
    "multimodal_vs_textrag": "eval_results/Pairwise_Eval/multimodal_vs_textrag",
    "textrag_vs_multimodal": "eval_results/Pairwise_Eval/textrag_vs_multimodal",
    "mmodal_full_vs_no_review": "eval_results/Pairwise_Eval/mmodal_full_vs_mmodal_no_user_review",
    "mmodal_no_review_vs_full": "eval_results/Pairwise_Eval/mmodal_no_review_vs_mmodal_full"
}

PAIRWISE_CRITERIA = [
    "GlobalPersonalizationConsistency",
    "CumulativeInformationSynthesis",
    "InteractionTrajectory",
    "ToneConsistency"
]


def load_eval_results(eval_dir):
    """
    Load all evaluation results from a directory.

    Returns:
        dict: {category_name: [list of evaluation results]}
    """
    results = {}

    json_files = glob.glob(os.path.join(eval_dir, "*_eval.json"))

    for file_path in json_files:
        filename = os.path.basename(file_path)
        category = filename.replace("_eval.json", "")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            results[category] = data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return results


def extract_scores(eval_results):
    """
    Extract scores from evaluation results.

    Args:
        eval_results: List of dialogue evaluations

    Returns:
        dict: {metric_name: [list of scores]}
    """
    scores = defaultdict(list)
    flaws = defaultdict(int)

    for entry in eval_results:
        if "error" in entry:
            continue

        evaluation = entry.get("evaluation", {})

        for metric in METRICS:
            if metric == "OverallScore":
                score = evaluation.get(metric)
                if score is not None:
                    scores[metric].append(score)
            else:
                metric_data = evaluation.get(metric, {})
                score = metric_data.get("score")
                if score is not None:
                    scores[metric].append(score)

        # Count flaws
        detected_flaws = evaluation.get("Major_Flaws_Detected", [])
        for flaw in detected_flaws:
            if flaw != "None":
                flaws[flaw] += 1

    return scores, flaws


def calculate_averages(scores):
    """Calculate average for each metric."""
    averages = {}
    for metric, score_list in scores.items():
        if score_list:
            averages[metric] = sum(score_list) / len(score_list)
        else:
            averages[metric] = None
    return averages


def summarize_config(config_name, eval_dir):
    """
    Summarize all categories for a single configuration.

    Returns:
        dict: {
            "per_category": {category: {metric: avg_score}},
            "overall": {metric: avg_score},
            "flaws": {flaw_name: count}
        }
    """
    results = load_eval_results(eval_dir)

    per_category = {}
    all_scores = defaultdict(list)
    all_flaws = defaultdict(int)
    total_dialogues = 0

    for category, eval_results in results.items():
        scores, flaws = extract_scores(eval_results)
        averages = calculate_averages(scores)
        per_category[category] = {
            "averages": averages,
            "num_dialogues": len(eval_results)
        }

        # Aggregate for overall
        for metric, score_list in scores.items():
            all_scores[metric].extend(score_list)

        for flaw, count in flaws.items():
            all_flaws[flaw] += count

        total_dialogues += len(eval_results)

    overall_averages = calculate_averages(all_scores)

    return {
        "config": config_name,
        "per_category": per_category,
        "overall": overall_averages,
        "flaws": dict(all_flaws),
        "total_dialogues": total_dialogues
    }


def print_comparison_table(summaries):
    """Print a formatted comparison table."""
    configs = list(summaries.keys())

    # Print header
    print("\n" + "=" * 100)
    print("OVERALL EVALUATION COMPARISON")
    print("=" * 100)

    # Header row
    header = f"{'Metric':<12}"
    for config in configs:
        header += f"{config:>22}"
    print(header)
    print("-" * 100)

    # Data rows
    for metric in METRICS:
        short_name = METRIC_SHORT_NAMES.get(metric, metric)
        row = f"{short_name:<12}"
        for config in configs:
            avg = summaries[config]["overall"].get(metric)
            if avg is not None:
                row += f"{avg:>22.3f}"
            else:
                row += f"{'N/A':>22}"
        print(row)

    print("-" * 100)

    # Dialogue counts
    row = f"{'Dialogues':<12}"
    for config in configs:
        count = summaries[config]["total_dialogues"]
        row += f"{count:>22}"
    print(row)
    print()


def print_category_comparison(summaries, metric="OverallScore"):
    """Print per-category comparison for a specific metric."""
    configs = list(summaries.keys())

    # Get all categories
    all_categories = set()
    for summary in summaries.values():
        all_categories.update(summary["per_category"].keys())
    categories = sorted(all_categories)

    print("\n" + "=" * 120)
    print(f"PER-CATEGORY COMPARISON ({METRIC_SHORT_NAMES.get(metric, metric)})")
    print("=" * 120)

    # Header
    header = f"{'Category':<35}"
    for config in configs:
        header += f"{config:>22}"
    print(header)
    print("-" * 120)

    # Data rows
    for category in categories:
        # Truncate long category names
        display_name = category[:33] if len(category) > 33 else category
        row = f"{display_name:<35}"

        for config in configs:
            cat_data = summaries[config]["per_category"].get(category, {})
            avg = cat_data.get("averages", {}).get(metric)
            if avg is not None:
                row += f"{avg:>22.3f}"
            else:
                row += f"{'N/A':>22}"
        print(row)

    print()


def print_flaw_comparison(summaries):
    """Print flaw frequency comparison."""
    configs = list(summaries.keys())

    # Get all flaws
    all_flaws = set()
    for summary in summaries.values():
        all_flaws.update(summary["flaws"].keys())
    flaws = sorted(all_flaws)

    if not flaws:
        return

    print("\n" + "=" * 120)
    print("FLAW FREQUENCY COMPARISON")
    print("=" * 120)

    # Header
    header = f"{'Flaw Type':<30}"
    for config in configs:
        header += f"{config:>22}"
    print(header)
    print("-" * 120)

    for flaw in flaws:
        row = f"{flaw:<30}"
        for config in configs:
            count = summaries[config]["flaws"].get(flaw, 0)
            row += f"{count:>22}"
        print(row)

    print()


def load_pairwise_summary(pairwise_dir):
    """Load pairwise evaluation summary from a directory."""
    summary_path = os.path.join(pairwise_dir, "pairwise_summary.json")
    if not os.path.exists(summary_path):
        return None

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {summary_path}: {e}")
        return None


def print_pairwise_comparison(pairwise_summaries):
    """Print pairwise comparison results."""
    if not pairwise_summaries:
        return

    print("\n" + "=" * 120)
    print("PAIRWISE EVALUATION COMPARISON")
    print("=" * 120)

    # Overall win rates table
    print("\n--- Overall Win Rates ---")
    print(f"{'Comparison':<35} {'A (Label)':<20} {'B (Label)':<20} {'A Wins':>10} {'B Wins':>10} {'Tie':>10} {'A Win%':>10}")
    print("-" * 120)

    for name, summary in pairwise_summaries.items():
        label_a = summary.get("label_a", "A")
        label_b = summary.get("label_b", "B")
        a_wins = summary.get("A_wins", 0)
        b_wins = summary.get("B_wins", 0)
        tie = summary.get("Tie", 0)
        a_rate = summary.get("A_win_rate", 0) * 100

        print(f"{name:<35} {label_a:<20} {label_b:<20} {a_wins:>10} {b_wins:>10} {tie:>10} {a_rate:>9.1f}%")

    # Per-criterion breakdown
    print("\n--- Per-Criterion Win Counts (A / B / Tie) ---")
    header = f"{'Comparison':<35}"
    for criterion in PAIRWISE_CRITERIA:
        short_name = METRIC_SHORT_NAMES.get(criterion, criterion[:10])
        header += f"{short_name:>20}"
    print(header)
    print("-" * 120)

    for name, summary in pairwise_summaries.items():
        row = f"{name:<35}"
        per_criterion = summary.get("per_criterion", {})
        for criterion in PAIRWISE_CRITERIA:
            crit_data = per_criterion.get(criterion, {})
            a = crit_data.get("A_wins", 0)
            b = crit_data.get("B_wins", 0)
            t = crit_data.get("Tie", 0)
            row += f"{a:>6}/{b:>5}/{t:>5}"
        print(row)

    print()


def save_pairwise_summary(pairwise_summaries, output_dir):
    """Save pairwise comparison summary to CSV."""
    if not pairwise_summaries:
        return

    csv_path = os.path.join(output_dir, "pairwise_comparison.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        # Header
        f.write("Comparison,Label_A,Label_B,Total_Pairs,A_Wins,B_Wins,Tie,A_Win_Rate,B_Win_Rate\n")

        for name, summary in pairwise_summaries.items():
            label_a = summary.get("label_a", "A")
            label_b = summary.get("label_b", "B")
            total = summary.get("total_pairs", 0)
            a_wins = summary.get("A_wins", 0)
            b_wins = summary.get("B_wins", 0)
            tie = summary.get("Tie", 0)
            a_rate = summary.get("A_win_rate", 0)
            b_rate = summary.get("B_win_rate", 0)

            f.write(f"{name},{label_a},{label_b},{total},{a_wins},{b_wins},{tie},{a_rate:.4f},{b_rate:.4f}\n")

    print(f"Pairwise comparison CSV saved to: {csv_path}")

    # Per-criterion CSV
    crit_csv_path = os.path.join(output_dir, "pairwise_per_criterion.csv")
    with open(crit_csv_path, "w", encoding="utf-8") as f:
        # Header
        header = ["Comparison"]
        for criterion in PAIRWISE_CRITERIA:
            short_name = METRIC_SHORT_NAMES.get(criterion, criterion)
            header.extend([f"{short_name}_A", f"{short_name}_B", f"{short_name}_Tie"])
        f.write(",".join(header) + "\n")

        for name, summary in pairwise_summaries.items():
            row = [name]
            per_criterion = summary.get("per_criterion", {})
            for criterion in PAIRWISE_CRITERIA:
                crit_data = per_criterion.get(criterion, {})
                row.append(str(crit_data.get("A_wins", 0)))
                row.append(str(crit_data.get("B_wins", 0)))
                row.append(str(crit_data.get("Tie", 0)))
            f.write(",".join(row) + "\n")

    print(f"Pairwise per-criterion CSV saved to: {crit_csv_path}")


def save_summary(summaries, output_dir):
    """Save summary to JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "evaluation_comparison.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"Summary saved to: {output_path}")

    # Also save a CSV for easy viewing
    csv_path = os.path.join(output_dir, "overall_comparison.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        configs = list(summaries.keys())

        # Header
        f.write("Metric," + ",".join(configs) + "\n")

        # Data
        for metric in METRICS:
            short_name = METRIC_SHORT_NAMES.get(metric, metric)
            row = [short_name]
            for config in configs:
                avg = summaries[config]["overall"].get(metric)
                row.append(f"{avg:.3f}" if avg is not None else "N/A")
            f.write(",".join(row) + "\n")

        # Dialogue counts
        row = ["Dialogues"]
        for config in configs:
            row.append(str(summaries[config]["total_dialogues"]))
        f.write(",".join(row) + "\n")

    print(f"CSV saved to: {csv_path}")

    # Save per-category CSV
    category_csv_path = os.path.join(output_dir, "per_category_comparison.csv")
    with open(category_csv_path, "w", encoding="utf-8") as f:
        configs = list(summaries.keys())

        # Get all categories
        all_categories = set()
        for summary in summaries.values():
            all_categories.update(summary["per_category"].keys())
        categories = sorted(all_categories)

        # Header with all metrics for each config
        header = ["Category"]
        for config in configs:
            for metric in METRICS:
                short_name = METRIC_SHORT_NAMES.get(metric, metric)
                header.append(f"{config}_{short_name}")
        f.write(",".join(header) + "\n")

        # Data
        for category in categories:
            row = [category]
            for config in configs:
                cat_data = summaries[config]["per_category"].get(category, {})
                for metric in METRICS:
                    avg = cat_data.get("averages", {}).get(metric)
                    row.append(f"{avg:.3f}" if avg is not None else "N/A")
            f.write(",".join(row) + "\n")

    print(f"Per-category CSV saved to: {category_csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize and compare evaluation results across RAG configurations"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_summary",
        help="Output directory for summary files"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Base directory containing eval_results"
    )
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 80)

    summaries = {}

    for config_name, rel_path in EVAL_DIRS.items():
        eval_dir = os.path.join(args.base_dir, rel_path)

        if not os.path.isdir(eval_dir):
            print(f"Warning: Directory not found: {eval_dir}")
            continue

        print(f"\nProcessing: {config_name} ({eval_dir})")
        summary = summarize_config(config_name, eval_dir)
        summaries[config_name] = summary
        print(f"  Loaded {len(summary['per_category'])} categories, {summary['total_dialogues']} dialogues")

    if not summaries:
        print("No evaluation results found!")
        return

    # Print comparison tables
    print_comparison_table(summaries)
    print_category_comparison(summaries, "OverallScore")
    print_flaw_comparison(summaries)

    # Save to files
    output_dir = os.path.join(args.base_dir, args.output_dir)
    save_summary(summaries, output_dir)

    # Load and print pairwise evaluation results
    print("\n" + "=" * 80)
    print("PAIRWISE EVALUATION RESULTS")
    print("=" * 80)

    pairwise_summaries = {}
    for name, rel_path in PAIRWISE_DIRS.items():
        pairwise_dir = os.path.join(args.base_dir, rel_path)
        if os.path.isdir(pairwise_dir):
            summary = load_pairwise_summary(pairwise_dir)
            if summary:
                pairwise_summaries[name] = summary
                print(f"Loaded pairwise: {name}")

    if pairwise_summaries:
        print_pairwise_comparison(pairwise_summaries)
        save_pairwise_summary(pairwise_summaries, output_dir)
    else:
        print("No pairwise evaluation results found.")


if __name__ == "__main__":
    main()
