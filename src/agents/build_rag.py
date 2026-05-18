#!/usr/bin/env python
"""
Main entry point for RAG index building.

This script uses the modular buildRag_openrouter package to build
RAG indexes for Amazon product data with OpenRouter API embeddings.

Usage:
    python build_rag.py --category Amazon_Fashion
    python build_rag.py --all
    python build_rag.py --category All_Beauty --questions-file ../questions/All_Beauty_questions.json
    python build_rag.py --captions --category All_Beauty
    python build_rag.py --captions --all
"""

import argparse
from pathlib import Path

# Import from modular package
from buildRag_openrouter import (
    DATA_DIR,
    check_category_complete,
    find_questions_file_for_category,
    build_item_index_for_category,
    build_caption_embeddings_for_category
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(description="Build RAG index for Amazon product data")
    parser.add_argument("--category", help="e.g. All_Beauty or Amazon_Fashion")
    parser.add_argument("--all", action="store_true", help="Build index for all available categories")
    parser.add_argument("--captions", action="store_true", help="Build caption embeddings from image captions")
    parser.add_argument(
        "--questions-file",
        help="Path to questions JSON file for excluding target product reviews"
    )

    args = parser.parse_args()

    # Parse questions file path if provided
    questions_path = Path(args.questions_file) if args.questions_file else None

    # Handle --captions mode
    if args.captions:
        if args.all:
            # Build caption embeddings for all categories
            cat2file = {}
            for p in DATA_DIR.glob("*_top_100_user_itemReview.json"):
                stem = p.stem
                suffix = "_top_100_user_itemReview"
                if not stem.endswith(suffix):
                    continue
                cat = stem[: -len(suffix)]
                cat2file[cat] = p.name

            print(f"\n{'='*80}")
            print(f"Building caption embeddings for {len(cat2file)} categories...")
            print(f"{'='*80}\n")

            for cat in sorted(cat2file.keys()):
                print(f"\n===== Building caption embeddings: {cat} =====")
                build_caption_embeddings_for_category(cat)

            print(f"\n{'='*80}")
            print(f"Caption embedding generation complete for all {len(cat2file)} categories!")
            print(f"{'='*80}\n")

        elif args.category:
            # Build caption embeddings for single category
            print(f"\n===== Building caption embeddings for category: {args.category} =====")
            build_caption_embeddings_for_category(args.category)

        else:
            parser.print_help()
            print("\nFor --captions mode, please specify --category or --all")
            print("Example usage:")
            print("  python build_rag.py --captions --category All_Beauty")
            print("  python build_rag.py --captions --all")

        return

    if args.all:
        # Build indexes for all categories
        cat2file = {}
        for p in DATA_DIR.glob("*_top_100_user_itemReview.json"):
            stem = p.stem
            suffix = "_top_100_user_itemReview"
            if not stem.endswith(suffix):
                continue
            cat = stem[: -len(suffix)]
            cat2file[cat] = p.name

        print(f"\n{'='*80}")
        print(f"Checking completeness for {len(cat2file)} categories...")
        print(f"Check items: 1. Item Meta  2. Product Review Embedding  3. User Review Embedding")
        print(f"{'='*80}\n")

        # Check completion status for all categories
        to_process = []
        already_complete = []

        for cat in sorted(cat2file.keys()):
            status = check_category_complete(cat)

            if status["complete"]:
                already_complete.append(cat)
                print(f"[SUCCESS] {cat:30s} - {status['details']}")
            else:
                to_process.append(cat)
                print(f"[WARN]  {cat:30s} - {status['details']}")

        # Display statistics
        print(f"\n{'='*80}")
        print(f"Complete:   {len(already_complete)}/{len(cat2file)}")
        print(f"To process: {len(to_process)}/{len(cat2file)}")
        print(f"{'='*80}\n")

        if not to_process:
            print("[INFO] All categories are complete!")
        else:
            print(f"[INFO] Processing {len(to_process)} incomplete categories...\n")

            # Process incomplete categories
            for cat in to_process:
                fname = cat2file[cat]
                print(f"\n===== Building: {cat} (from {fname}) =====")

                # Auto-find questions file for this category
                questions_file = find_questions_file_for_category(cat)

                build_item_index_for_category(cat, questions_file)

    elif args.category:
        # Build index for single category
        print(f"\n===== Building item index for category: {args.category} =====")

        # Determine questions file
        if args.questions_file:
            # Use user-specified questions file
            questions_path = Path(args.questions_file)
        else:
            # Auto-find questions file
            questions_path = find_questions_file_for_category(args.category)

        build_item_index_for_category(args.category, questions_path)

    else:
        # No arguments provided
        parser.print_help()
        print("\nExample usage:")
        print("  python build_rag.py --category Amazon_Fashion")
        print("  python build_rag.py --all")


if __name__ == "__main__":
    main()
