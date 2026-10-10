"""Build the Hugging Face release files from data/questions_v3 and data/user_profiles.

Merges the 33 per-category JSON files of each subset into a single JSONL file
with a `category` column, and normalizes the mixed-type product-metadata fields
so the files load with `datasets.load_dataset` and the Hub's Dataset Viewer.

Usage:
    python src/data_prep/build_hf_release.py [--out hf_release]
"""

import argparse
import glob
import json
import math
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QUESTIONS_DIR = os.path.join(REPO_ROOT, "data", "questions_v3")
PROFILES_DIR = os.path.join(REPO_ROOT, "data", "user_profiles")

PROFILE_FIELDS = [
    "Preferences",
    "Style",
    "Budget",
    "Size",
    "Quality_Priorities",
    "Dislikes",
    "Pain_Points",
    "Additional_Notes",
]

# product_meta fields kept as-is apart from NaN -> null
META_STR_FIELDS = ["main_category", "title", "store", "parent_asin", "subtitle", "author"]
META_LIST_FIELDS = ["features", "description", "categories"]


def _is_nan(value):
    return isinstance(value, float) and math.isnan(value)


def _none_if_nan(value):
    return None if _is_nan(value) else value


def _parse_price(value):
    """Source prices are floats, numeric strings, NaN, 'None' or '—'."""
    if value is None or _is_nan(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except ValueError:
        return None


def _as_int(value):
    if value is None or _is_nan(value):
        return None
    if float(value) != int(value):
        raise ValueError(f"expected an integer-valued number, got {value!r}")
    return int(value)


def normalize_product_meta(meta):
    out = {k: _none_if_nan(meta.get(k)) for k in META_STR_FIELDS}
    for k in META_LIST_FIELDS:
        out[k] = meta.get(k) or []
    out["average_rating"] = _none_if_nan(meta.get("average_rating"))
    out["rating_number"] = _as_int(meta.get("rating_number"))
    out["price"] = _parse_price(meta.get("price"))
    # `details` has a different key set per product (340+ keys overall, with
    # string / list / dict values), so it is stored as a JSON string.
    out["details"] = json.dumps(meta.get("details") or {}, ensure_ascii=False, sort_keys=True)
    # `bought_together` is empty for every product in the source and is dropped.
    return out


def build_questions():
    rows = []
    for path in sorted(glob.glob(os.path.join(QUESTIONS_DIR, "*_questions_*.json"))):
        category = os.path.basename(path).split("_questions_")[0]
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        for rec in records:
            target = rec["target_product"]
            rows.append(
                {
                    "category": category,
                    "user_id": rec["user_id"],
                    "target_product": {
                        "product_meta": normalize_product_meta(target["product_meta"]),
                        "rating": _as_int(target["rating"]),
                        "review_title": target["review_title"],
                        "review_text": target["review_text"],
                    },
                    "user_profile": {k: rec["user_profile"][k] for k in PROFILE_FIELDS},
                    "generated_questions": rec["generated_questions"],
                }
            )
    return rows


def build_user_profiles():
    rows = []
    for path in sorted(glob.glob(os.path.join(PROFILES_DIR, "*_user_profiles.json"))):
        category = os.path.basename(path)[: -len("_user_profiles.json")]
        with open(path, encoding="utf-8") as f:
            profiles = json.load(f)
        for user_id, profile in profiles.items():
            row = {"category": category, "user_id": user_id}
            row.update({k: profile[k] for k in PROFILE_FIELDS})
            rows.append(row)
    return rows


def write_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            # allow_nan=False guarantees strict JSON (no bare NaN tokens)
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "hf_release"))
    args = parser.parse_args()

    questions = build_questions()
    profiles = build_user_profiles()
    write_jsonl(questions, os.path.join(args.out, "questions", "test.jsonl"))
    write_jsonl(profiles, os.path.join(args.out, "user_profiles", "test.jsonl"))

    n_questions = sum(len(r["generated_questions"]) for r in questions)
    print(f"questions:     {len(questions)} rows, {n_questions} questions, "
          f"{len({r['category'] for r in questions})} categories, "
          f"{len({r['user_id'] for r in questions})} unique user ids")
    print(f"user_profiles: {len(profiles)} rows, "
          f"{len({r['category'] for r in profiles})} categories, "
          f"{len({r['user_id'] for r in profiles})} unique user ids")


if __name__ == "__main__":
    main()
