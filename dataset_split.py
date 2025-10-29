import json
import os
import random
from pprint import pprint
import pandas as pd

def load_json_dataset(path):
    """Load the dataset JSON file and return it as a list of dicts."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("The JSON file must contain a list of objects.")
    return data


def summarize_dataset(data):
    """Print a summary of the dataset content."""
    df = pd.DataFrame(data)
    print(f"\nLoaded {len(df)} samples.")

    if "split" in df.columns:
        print("\nSplit counts:")
        print(df["split"].value_counts())

    if "task" in df.columns:
        print("\nTask distribution:")
        print(df["task"].value_counts())

    if "is_vc" in df.columns:
        print("\nis_vc distribution:")
        print(df["is_vc"].value_counts())


def show_samples(data, n=2):
    """Pretty-print a few random examples from the dataset."""
    print(f"\nShowing {n} sample(s):\n")
    for i, sample in enumerate(data[:n]):
        print(f"--- Sample {i+1} ---")
        pprint(sample)
        print()


def sample_dataset(data, fraction=0.001, seed=42, save_path=None):
    """
    Randomly sample a fraction (default 1/1000) of the dataset.
    If save_path is provided, saves the subset to that file.
    """
    if not 0 < fraction <= 1:
        raise ValueError("Fraction must be between 0 and 1.")
    random.seed(seed)
    sample_size = max(1, int(len(data) * fraction))
    sampled = random.sample(data, sample_size)
    print(f"\nSampled {sample_size} entries out of {len(data)} (~{fraction*100:.2f}%).")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(sampled, f, ensure_ascii=False, indent=2)
        print(f"Saved to {save_path}")
    return sampled


if __name__ == "__main__":
    dataset_path = "data/dataset.json"
    subset_path = "data/dataset_1-10.json"

    data = load_json_dataset(dataset_path)
    summarize_dataset(data)

    sampled_data = sample_dataset(data, fraction=0.1, save_path=subset_path)
    summarize_dataset(sampled_data)
    show_samples(sampled_data, n=2)
