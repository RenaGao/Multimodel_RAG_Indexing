"""
analyze_eval_results.py
-----------------------------------
Compare and analyze evaluation results from different runs.
Supports A vs B comparison (e.g., baseline vs textrag, textrag vs multimodal).

Usage:
    python analyze_eval_results.py \
        --eval-a eval_results/baseline_eval \
        --eval-b eval_results/textrag_eval \
        --label-a baseline \
        --label-b textrag \
        --output-dir comparison_results
"""

import json
import os
import argparse
import glob


def load_eval_results(eval_dir):
    """
    Load all *_eval.json files from a directory.

    Returns:
        Dict of category -> list of evaluation results
    """
    results = {}
    eval_files = glob.glob(os.path.join(eval_dir, "*_eval.json"))

    for file_path in sorted(eval_files):
        filename = os.path.basename(file_path)
        # Extract category name (e.g., "Electronics_eval.json" -> "Electronics")
        category = filename.replace("_eval.json", "")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            results[category] = data
            print(f"✓ Loaded {len(data)} results from {filename}")
        except Exception as e:
            print(f"✗ Error loading {filename}: {str(e)}")

    return results


def generate_paired_comparison(results_a, results_b, label_a, label_b):
    """
    Generate paired comparison between two sets of results.

    Args:
        results_a: List of eval results for variant A
        results_b: List of eval results for variant B
        label_a: Label for variant A
        label_b: Label for variant B

    Returns:
        Dict with paired comparison statistics
    """
    # Index by dialogue_id
    index_a = {r.get("dialogue_id"): r for r in results_a if "error" not in r}
    index_b = {r.get("dialogue_id"): r for r in results_b if "error" not in r}

    # Find common dialogue_ids
    common_ids = set(index_a.keys()) & set(index_b.keys())

    if not common_ids:
        return {"error": "No common dialogue_ids found", "label_a": label_a, "label_b": label_b}

    pairs = []
    for dialogue_id in common_ids:
        eval_a = index_a[dialogue_id]["evaluation"]
        eval_b = index_b[dialogue_id]["evaluation"]

        pair_diff = {
            "dialogue_id": dialogue_id,
            "consistency_diff": eval_b["GlobalPersonalizationConsistency"]["score"] - eval_a["GlobalPersonalizationConsistency"]["score"],
            "synthesis_diff": eval_b["CumulativeInformationSynthesis"]["score"] - eval_a["CumulativeInformationSynthesis"]["score"],
            "trajectory_diff": eval_b["InteractionTrajectory"]["score"] - eval_a["InteractionTrajectory"]["score"],
            "tone_diff": eval_b["ToneConsistency"]["score"] - eval_a["ToneConsistency"]["score"],
            "overall_diff": eval_b.get("OverallScore", 0) - eval_a.get("OverallScore", 0),
            f"{label_b}_scores": eval_b,
            f"{label_a}_scores": eval_a
        }
        pairs.append(pair_diff)

    n = len(pairs)
    stats = {
        "total_pairs": n,
        "label_a": label_a,
        "label_b": label_b,
        label_b: {
            "avg_consistency": sum(p[f"{label_b}_scores"]["GlobalPersonalizationConsistency"]["score"] for p in pairs) / n,
            "avg_synthesis": sum(p[f"{label_b}_scores"]["CumulativeInformationSynthesis"]["score"] for p in pairs) / n,
            "avg_trajectory": sum(p[f"{label_b}_scores"]["InteractionTrajectory"]["score"] for p in pairs) / n,
            "avg_tone": sum(p[f"{label_b}_scores"]["ToneConsistency"]["score"] for p in pairs) / n,
            "avg_overall": sum(p[f"{label_b}_scores"].get("OverallScore", 0) for p in pairs) / n
        },
        label_a: {
            "avg_consistency": sum(p[f"{label_a}_scores"]["GlobalPersonalizationConsistency"]["score"] for p in pairs) / n,
            "avg_synthesis": sum(p[f"{label_a}_scores"]["CumulativeInformationSynthesis"]["score"] for p in pairs) / n,
            "avg_trajectory": sum(p[f"{label_a}_scores"]["InteractionTrajectory"]["score"] for p in pairs) / n,
            "avg_tone": sum(p[f"{label_a}_scores"]["ToneConsistency"]["score"] for p in pairs) / n,
            "avg_overall": sum(p[f"{label_a}_scores"].get("OverallScore", 0) for p in pairs) / n
        },
        f"{label_b}_better_count": sum(1 for p in pairs if p["overall_diff"] > 0.1),
        f"{label_a}_better_count": sum(1 for p in pairs if p["overall_diff"] < -0.1),
        "tie_count": sum(1 for p in pairs if abs(p["overall_diff"]) <= 0.1),
        "pairs": pairs
    }

    return stats


def print_comparison_report(stats, label_a, label_b):
    """Print comparison report to console."""
    if "error" in stats:
        print(f"\n⚠️  {stats['error']}")
        return

    print("\n" + "="*60)
    print(f"📊 COMPARISON: {label_b.upper()} vs {label_a.upper()}")
    print("="*60)

    total = stats["total_pairs"]
    print(f"\n🔗 Total Paired Dialogues: {total}")

    variant_b = stats[label_b]
    variant_a = stats[label_a]

    print(f"\n📈 {label_b.upper()} Average Scores:")
    print(f"   Global Consistency:     {variant_b['avg_consistency']:.2f}")
    print(f"   Information Synthesis:  {variant_b['avg_synthesis']:.2f}")
    print(f"   Interaction Trajectory: {variant_b['avg_trajectory']:.2f}")
    print(f"   Tone Consistency:       {variant_b['avg_tone']:.2f}")
    print(f"   Overall Score:          {variant_b['avg_overall']:.2f}")

    print(f"\n📉 {label_a.upper()} Average Scores:")
    print(f"   Global Consistency:     {variant_a['avg_consistency']:.2f}")
    print(f"   Information Synthesis:  {variant_a['avg_synthesis']:.2f}")
    print(f"   Interaction Trajectory: {variant_a['avg_trajectory']:.2f}")
    print(f"   Tone Consistency:       {variant_a['avg_tone']:.2f}")
    print(f"   Overall Score:          {variant_a['avg_overall']:.2f}")

    b_better = stats[f"{label_b}_better_count"]
    a_better = stats[f"{label_a}_better_count"]
    tie = stats["tie_count"]

    print(f"\n🏆 Win/Loss Breakdown:")
    print(f"   {label_b.upper()} Better:      {b_better:3d} ({b_better/total*100:.1f}%)")
    print(f"   {label_a.upper()} Better:      {a_better:3d} ({a_better/total*100:.1f}%)")
    print(f"   Tie (±0.1):             {tie:3d} ({tie/total*100:.1f}%)")


def compare_eval_dirs(eval_dir_a, eval_dir_b, label_a, label_b, output_dir):
    """
    Compare evaluation results from two directories.

    Args:
        eval_dir_a: Directory containing eval results for variant A
        eval_dir_b: Directory containing eval results for variant B
        label_a: Label for variant A
        label_b: Label for variant B
        output_dir: Directory to save comparison results
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"🔍 ANALYZING EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Variant A ({label_a}): {eval_dir_a}")
    print(f"Variant B ({label_b}): {eval_dir_b}")
    print(f"Output: {output_dir}")

    # Load results
    print(f"\n📂 Loading {label_a} results...")
    results_a = load_eval_results(eval_dir_a)

    print(f"\n📂 Loading {label_b} results...")
    results_b = load_eval_results(eval_dir_b)

    # Find common categories
    common_categories = set(results_a.keys()) & set(results_b.keys())
    print(f"\n✓ Found {len(common_categories)} common categories")

    if not common_categories:
        print("❌ No common categories found!")
        return

    # Compare each category
    all_category_stats = []
    all_pairs = []

    for category in sorted(common_categories):
        print(f"\n--- {category} ---")
        stats = generate_paired_comparison(
            results_a[category],
            results_b[category],
            label_a,
            label_b
        )

        if "error" not in stats:
            # Save category comparison
            output_path = os.path.join(output_dir, f"{category}_comparison.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)

            all_category_stats.append({
                "category": category,
                "total_pairs": stats["total_pairs"],
                label_a: stats[label_a],
                label_b: stats[label_b],
                f"{label_b}_better_count": stats[f"{label_b}_better_count"],
                f"{label_a}_better_count": stats[f"{label_a}_better_count"],
                "tie_count": stats["tie_count"]
            })
            all_pairs.extend(stats["pairs"])

            print_comparison_report(stats, label_a, label_b)
        else:
            print(f"⚠️  {stats['error']}")

    # Generate overall summary
    if all_pairs:
        n = len(all_pairs)
        overall_stats = {
            "total_categories": len(all_category_stats),
            "total_pairs": n,
            "label_a": label_a,
            "label_b": label_b,
            label_b: {
                "avg_consistency": sum(p[f"{label_b}_scores"]["GlobalPersonalizationConsistency"]["score"] for p in all_pairs) / n,
                "avg_synthesis": sum(p[f"{label_b}_scores"]["CumulativeInformationSynthesis"]["score"] for p in all_pairs) / n,
                "avg_trajectory": sum(p[f"{label_b}_scores"]["InteractionTrajectory"]["score"] for p in all_pairs) / n,
                "avg_tone": sum(p[f"{label_b}_scores"]["ToneConsistency"]["score"] for p in all_pairs) / n,
                "avg_overall": sum(p[f"{label_b}_scores"].get("OverallScore", 0) for p in all_pairs) / n
            },
            label_a: {
                "avg_consistency": sum(p[f"{label_a}_scores"]["GlobalPersonalizationConsistency"]["score"] for p in all_pairs) / n,
                "avg_synthesis": sum(p[f"{label_a}_scores"]["CumulativeInformationSynthesis"]["score"] for p in all_pairs) / n,
                "avg_trajectory": sum(p[f"{label_a}_scores"]["InteractionTrajectory"]["score"] for p in all_pairs) / n,
                "avg_tone": sum(p[f"{label_a}_scores"]["ToneConsistency"]["score"] for p in all_pairs) / n,
                "avg_overall": sum(p[f"{label_a}_scores"].get("OverallScore", 0) for p in all_pairs) / n
            },
            f"{label_b}_better_count": sum(1 for p in all_pairs if p["overall_diff"] > 0.1),
            f"{label_a}_better_count": sum(1 for p in all_pairs if p["overall_diff"] < -0.1),
            "tie_count": sum(1 for p in all_pairs if abs(p["overall_diff"]) <= 0.1),
            "per_category": all_category_stats
        }

        # Save overall summary
        summary_path = os.path.join(output_dir, "all_categories_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(overall_stats, f, ensure_ascii=False, indent=2)

        # Print overall summary
        print(f"\n{'='*60}")
        print(f"📊 OVERALL SUMMARY (All Categories)")
        print(f"{'='*60}")
        print_comparison_report(overall_stats, label_a, label_b)
        print(f"\n💾 Summary saved to: {summary_path}")


# -----------------------------------
# Main
# -----------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare and analyze evaluation results from different runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare baseline vs textrag
  python %(prog)s \\
      --eval-a eval_results/baseline_eval \\
      --eval-b eval_results/textrag_eval \\
      --label-a baseline \\
      --label-b textrag \\
      --output-dir comparison_results

  # Compare textrag vs multimodal
  python %(prog)s \\
      --eval-a eval_results/textrag_eval \\
      --eval-b eval_results/multimodal_eval \\
      --label-a textrag \\
      --label-b multimodal \\
      --output-dir textrag_vs_multimodal
"""
    )
    parser.add_argument(
        "--eval-a",
        type=str,
        required=True,
        help="Directory containing evaluation results for variant A"
    )
    parser.add_argument(
        "--eval-b",
        type=str,
        required=True,
        help="Directory containing evaluation results for variant B"
    )
    parser.add_argument(
        "--label-a",
        type=str,
        required=True,
        help="Label for variant A (e.g., 'baseline', 'textrag')"
    )
    parser.add_argument(
        "--label-b",
        type=str,
        required=True,
        help="Label for variant B (e.g., 'textrag', 'multimodal')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for comparison results"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.eval_a):
        raise ValueError(f"Eval directory A not found: {args.eval_a}")
    if not os.path.isdir(args.eval_b):
        raise ValueError(f"Eval directory B not found: {args.eval_b}")

    compare_eval_dirs(args.eval_a, args.eval_b, args.label_a, args.label_b, args.output_dir)
