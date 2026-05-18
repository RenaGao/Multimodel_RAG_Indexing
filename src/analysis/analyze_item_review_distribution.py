"""
analyze_item_review_distribution.py
-----------------------------------
Analyze item and review count distribution across Amazon product categories.
Generates two seaborn bar charts: one for items, one for reviews.
"""

import json
import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set USER_REVIEWS_DIR in your environment, or override with --data-dir on the CLI.
DATA_DIR = os.environ.get("USER_REVIEWS_DIR", "")


def analyze_files(data_dir=DATA_DIR):
    """
    Analyze all category files and return statistics.

    Args:
        data_dir: Directory containing the JSON files

    Returns:
        DataFrame with columns: category, items, reviews
    """
    stats = []

    data_path = Path(data_dir)
    files = list(data_path.glob("*_top_100_user_itemReview.json"))

    if not files:
        print(f"No files found in {data_dir}")
        return pd.DataFrame()

    print(f"Found {len(files)} category files")
    print("-" * 50)

    for f in sorted(files):
        # Extract category name from filename
        category = f.stem.replace("_top_100_user_itemReview", "")

        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        item_count = len(data)  # Number of unique items (keys)
        review_count = sum(len(reviews) for reviews in data.values())  # Total reviews

        stats.append({
            "category": category,
            "items": item_count,
            "reviews": review_count,
            "avg_reviews_per_item": round(review_count / item_count, 2) if item_count > 0 else 0
        })

        print(f"{category}: {item_count} items, {review_count} reviews")

    return pd.DataFrame(stats)


def plot_distributions(df, output_dir="."):
    """
    Plot item and review distributions using seaborn.

    Args:
        df: DataFrame with category, items, reviews columns
        output_dir: Directory to save output images
    """
    if df.empty:
        print("No data to plot")
        return

    # Set seaborn style
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams['figure.facecolor'] = 'white'

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # Figure 1: Item Count Distribution
    # ==========================================
    df_by_items = df.sort_values("items", ascending=True)

    fig1, ax1 = plt.subplots(figsize=(14, 12))

    bars1 = sns.barplot(
        data=df_by_items,
        y="category",
        x="items",
        palette="viridis",
        ax=ax1
    )

    # Add value labels on bars
    for i, (idx, row) in enumerate(df_by_items.iterrows()):
        ax1.text(row['items'] + 50, i, f"{row['items']:,}",
                 va='center', fontsize=9)

    ax1.set_title("Item Count Distribution by Category", fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel("Number of Items", fontsize=12)
    ax1.set_ylabel("Category", fontsize=12)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    plt.tight_layout()
    fig1.savefig(output_path / "item_distribution.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path / 'item_distribution.png'}")

    # ==========================================
    # Figure 2: Review Count Distribution
    # ==========================================
    df_by_reviews = df.sort_values("reviews", ascending=True)

    fig2, ax2 = plt.subplots(figsize=(14, 12))

    bars2 = sns.barplot(
        data=df_by_reviews,
        y="category",
        x="reviews",
        palette="magma",
        ax=ax2
    )

    # Add value labels on bars
    for i, (idx, row) in enumerate(df_by_reviews.iterrows()):
        ax2.text(row['reviews'] + 500, i, f"{row['reviews']:,}",
                 va='center', fontsize=9)

    ax2.set_title("Review Count Distribution by Category", fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel("Number of Reviews", fontsize=12)
    ax2.set_ylabel("Category", fontsize=12)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    plt.tight_layout()
    fig2.savefig(output_path / "review_distribution.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path / 'review_distribution.png'}")

    plt.show()


def print_summary(df):
    """Print summary statistics."""
    if df.empty:
        return

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total categories: {len(df)}")
    print(f"Total items: {df['items'].sum():,}")
    print(f"Total reviews: {df['reviews'].sum():,}")
    print(f"\nItems per category:")
    print(f"  Min: {df['items'].min():,} ({df.loc[df['items'].idxmin(), 'category']})")
    print(f"  Max: {df['items'].max():,} ({df.loc[df['items'].idxmax(), 'category']})")
    print(f"  Mean: {df['items'].mean():,.0f}")
    print(f"  Median: {df['items'].median():,.0f}")
    print(f"\nReviews per category:")
    print(f"  Min: {df['reviews'].min():,} ({df.loc[df['reviews'].idxmin(), 'category']})")
    print(f"  Max: {df['reviews'].max():,} ({df.loc[df['reviews'].idxmax(), 'category']})")
    print(f"  Mean: {df['reviews'].mean():,.0f}")
    print(f"  Median: {df['reviews'].median():,.0f}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze item and review distribution")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DATA_DIR,
        help="Directory containing the JSON files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save output images"
    )
    args = parser.parse_args()

    # Analyze files
    df = analyze_files(args.data_dir)

    if not df.empty:
        # Print full table
        print("\n" + "=" * 60)
        print("DETAILED STATISTICS")
        print("=" * 60)
        print(df.sort_values("reviews", ascending=False).to_string(index=False))

        # Print summary
        print_summary(df)

        # Save to CSV
        csv_path = Path(args.output_dir) / "item_review_stats.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved CSV: {csv_path}")

        # Plot
        plot_distributions(df, args.output_dir)
