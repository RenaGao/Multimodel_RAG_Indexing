#!/usr/bin/env python3
"""
Rerun problematic dialogues (multi-threaded version)
Based on problematic_dialogues.csv
"""

import os
import subprocess
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import threading

# Max parallel workers
MAX_WORKERS = 5

# Thread-safe print
print_lock = threading.Lock()

def safe_print(msg):
    with print_lock:
        print(msg)

# ==========================================
# Problematic dialogue list
# ==========================================

# multimodal_dialogues_output problems (6)
MULTIMODAL_PROBLEMS = [
    ("Beauty_and_Personal_Care", "AEZP6Z2C5AVQDZAJECQYZWQRNG3Q"),
    ("Sports_and_Outdoors", "AGZZXSMMS4WRHHJRBUJZI4FZDHKQ"),
    ("Amazon_Fashion", "AG73BVBKUOH22USSFJA5ZWL7AKXA"),
    ("Office_Products", "AG375WAXLZ7PIOQKIQ6KQB4J3JVQ"),
    ("Health_and_Household", "AG73BVBKUOH22USSFJA5ZWL7AKXA"),
    ("Health_and_Personal_Care", "AGNUU44ETNXLBOUB53LJGFJP3SQA"),
]

# multimdoal_no_user_review_output problems (7)
NO_USER_REVIEW_PROBLEMS = [
    ("Pet_Supplies", "AEYVPPWR4CIKWX4BGYKCBCDL2CZQ"),
    ("Pet_Supplies", "AG73BVBKUOH22USSFJA5ZWL7AKXA"),
    ("Toys_and_Games", "AEIIRIHLIYKQGI7ZOCIJTRDF5NPQ"),
    ("Office_Products", "AEIIRIHLIYKQGI7ZOCIJTRDF5NPQ"),
    ("Digital_Music", "AGAFM74L2RIJ5O36NNYH4Z5ISQNQ"),
    ("Patio_Lawn_and_Garden", "AGUTZC4GHLTGYHA3KBEDRF6MHB6A"),
    ("Health_and_Household", "AHPGHDFIU3BUB3RQBP56RQQA7W4Q"),
]


def delete_old_files(output_dir: str, json_pattern: str, jsonl_pattern: str):
    """Delete old output files"""
    if not os.path.isdir(output_dir):
        return

    # Delete JSON files
    for f in glob.glob(os.path.join(output_dir, json_pattern)):
        os.remove(f)
        safe_print(f"  Deleted: {f}")

    # Delete JSONL files
    for f in glob.glob(os.path.join(output_dir, jsonl_pattern)):
        os.remove(f)
        safe_print(f"  Deleted: {f}")


def process_multimodal(category: str, user_id: str, log_dir: str) -> tuple:
    """Process multimodal dialogue"""
    safe_print(f"[multimodal] Starting: {category} / {user_id}")

    # Delete old files
    output_dir = f"./dialogue_outputs/multimodal_dialogues_output/{category}"
    delete_old_files(
        output_dir,
        f"multimodal_rag_dialogues_{user_id}_*.json",
        f"{category}_{user_id}_*.jsonl"
    )

    # Run script
    log_file = os.path.join(log_dir, f"multimodal_{category}_{user_id}.log")
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                ["python", "imageRagBaseAgent.py", "--category", category, "--user-id", user_id],
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=1800  # 30 min timeout
            )
        success = result.returncode == 0
    except subprocess.TimeoutExpired:
        success = False
        safe_print(f"[multimodal] Timeout: {category} / {user_id}")
    except Exception as e:
        success = False
        safe_print(f"[multimodal] Error: {category} / {user_id}: {e}")

    status = "Done" if success else "Failed"
    safe_print(f"[multimodal] {status}: {category} / {user_id}")
    return ("multimodal", category, user_id, success)


def process_no_user_review(category: str, user_id: str, log_dir: str) -> tuple:
    """Process no_user_review dialogue"""
    safe_print(f"[no_user_review] Starting: {category} / {user_id}")

    # Delete old files
    output_dir = f"./dialogue_outputs/multimdoal_no_user_review_output/{category}"
    delete_old_files(
        output_dir,
        f"multimodal_no_user_review_dialogues_{user_id}_*.json",
        f"{category}_{user_id}_*.jsonl"
    )

    # Run script
    log_file = os.path.join(log_dir, f"no_user_review_{category}_{user_id}.log")
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                ["python", "imageRagBaseAgent_ablation_no_user_review.py", "--category", category, "--user-id", user_id],
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=1800  # 30 min timeout
            )
        success = result.returncode == 0
    except subprocess.TimeoutExpired:
        success = False
        safe_print(f"[no_user_review] Timeout: {category} / {user_id}")
    except Exception as e:
        success = False
        safe_print(f"[no_user_review] Error: {category} / {user_id}: {e}")

    status = "Done" if success else "Failed"
    safe_print(f"[no_user_review] {status}: {category} / {user_id}")
    return ("no_user_review", category, user_id, success)


def main():
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/rerun_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 60)
    print("Rerun Problematic Dialogues (Multi-threaded)")
    print(f"Max workers: {MAX_WORKERS}")
    print(f"Log dir: {log_dir}")
    print("=" * 60)

    results = []

    # ==========================================
    # Process multimodal_dialogues_output (6)
    # ==========================================
    print("\n[1/2] Processing multimodal_dialogues_output...")
    print(f"Total tasks: {len(MULTIMODAL_PROBLEMS)}\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_multimodal, cat, uid, log_dir): (cat, uid)
            for cat, uid in MULTIMODAL_PROBLEMS
        }
        for future in as_completed(futures):
            results.append(future.result())

    print("\nMultimodal batch done!")

    # ==========================================
    # Process multimdoal_no_user_review_output (7)
    # ==========================================
    print("\n[2/2] Processing multimdoal_no_user_review_output...")
    print(f"Total tasks: {len(NO_USER_REVIEW_PROBLEMS)}\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_no_user_review, cat, uid, log_dir): (cat, uid)
            for cat, uid in NO_USER_REVIEW_PROBLEMS
        }
        for future in as_completed(futures):
            results.append(future.result())

    print("\nNo_user_review batch done!")

    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    success_count = sum(1 for r in results if r[3])
    fail_count = len(results) - success_count

    print(f"\nSuccess: {success_count}, Failed: {fail_count}")

    if fail_count > 0:
        print("\nFailed tasks:")
        for method, cat, uid, success in results:
            if not success:
                print(f"  - [{method}] {cat} / {uid}")

    print(f"\nLog dir: {log_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
