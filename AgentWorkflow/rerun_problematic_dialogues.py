#!/usr/bin/env python3
"""
重新处理有问题的对话脚本 (多线程版本)
根据 problematic_dialogues.csv 中记录的问题对话
"""

import os
import subprocess
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import threading

# 最大并行数
MAX_WORKERS = 5

# 线程安全打印
print_lock = threading.Lock()

def safe_print(msg):
    with print_lock:
        print(msg)

# ==========================================
# 问题对话列表
# ==========================================

# multimodal_dialogues_output 的问题对话 (6个)
MULTIMODAL_PROBLEMS = [
    ("Beauty_and_Personal_Care", "AEZP6Z2C5AVQDZAJECQYZWQRNG3Q"),
    ("Sports_and_Outdoors", "AGZZXSMMS4WRHHJRBUJZI4FZDHKQ"),
    ("Amazon_Fashion", "AG73BVBKUOH22USSFJA5ZWL7AKXA"),
    ("Office_Products", "AG375WAXLZ7PIOQKIQ6KQB4J3JVQ"),
    ("Health_and_Household", "AG73BVBKUOH22USSFJA5ZWL7AKXA"),
    ("Health_and_Personal_Care", "AGNUU44ETNXLBOUB53LJGFJP3SQA"),
]

# multimdoal_no_user_review_output 的问题对话 (7个)
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
    """删除旧的输出文件"""
    if not os.path.isdir(output_dir):
        return

    # 删除JSON文件
    for f in glob.glob(os.path.join(output_dir, json_pattern)):
        os.remove(f)
        safe_print(f"  删除: {f}")

    # 删除JSONL文件
    for f in glob.glob(os.path.join(output_dir, jsonl_pattern)):
        os.remove(f)
        safe_print(f"  删除: {f}")


def process_multimodal(category: str, user_id: str, log_dir: str) -> tuple:
    """处理 multimodal 对话"""
    safe_print(f"[multimodal] 开始: {category} / {user_id}")

    # 删除旧文件
    output_dir = f"./dialogue_outputs/multimodal_dialogues_output/{category}"
    delete_old_files(
        output_dir,
        f"multimodal_rag_dialogues_{user_id}_*.json",
        f"{category}_{user_id}_*.jsonl"
    )

    # 运行脚本
    log_file = os.path.join(log_dir, f"multimodal_{category}_{user_id}.log")
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                ["python", "imageRagBaseAgent.py", "--category", category, "--user-id", user_id],
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=1800  # 30分钟超时
            )
        success = result.returncode == 0
    except subprocess.TimeoutExpired:
        success = False
        safe_print(f"[multimodal] 超时: {category} / {user_id}")
    except Exception as e:
        success = False
        safe_print(f"[multimodal] 错误: {category} / {user_id}: {e}")

    status = "成功" if success else "失败"
    safe_print(f"[multimodal] {status}: {category} / {user_id}")
    return ("multimodal", category, user_id, success)


def process_no_user_review(category: str, user_id: str, log_dir: str) -> tuple:
    """处理 no_user_review 对话"""
    safe_print(f"[no_user_review] 开始: {category} / {user_id}")

    # 删除旧文件
    output_dir = f"./dialogue_outputs/multimdoal_no_user_review_output/{category}"
    delete_old_files(
        output_dir,
        f"multimodal_no_user_review_dialogues_{user_id}_*.json",
        f"{category}_{user_id}_*.jsonl"
    )

    # 运行脚本
    log_file = os.path.join(log_dir, f"no_user_review_{category}_{user_id}.log")
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                ["python", "imageRagBaseAgent_ablation_no_user_review.py", "--category", category, "--user-id", user_id],
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=1800  # 30分钟超时
            )
        success = result.returncode == 0
    except subprocess.TimeoutExpired:
        success = False
        safe_print(f"[no_user_review] 超时: {category} / {user_id}")
    except Exception as e:
        success = False
        safe_print(f"[no_user_review] 错误: {category} / {user_id}: {e}")

    status = "成功" if success else "失败"
    safe_print(f"[no_user_review] {status}: {category} / {user_id}")
    return ("no_user_review", category, user_id, success)


def main():
    # 切换到脚本所在目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # 创建日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/rerun_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 60)
    print("重新处理有问题的对话 (多线程版本)")
    print(f"最大并行数: {MAX_WORKERS}")
    print(f"日志目录: {log_dir}")
    print("=" * 60)

    results = []

    # ==========================================
    # 处理 multimodal_dialogues_output (6个)
    # ==========================================
    print("\n[1/2] 并行处理 multimodal_dialogues_output 的问题对话...")
    print(f"共 {len(MULTIMODAL_PROBLEMS)} 个任务\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_multimodal, cat, uid, log_dir): (cat, uid)
            for cat, uid in MULTIMODAL_PROBLEMS
        }
        for future in as_completed(futures):
            results.append(future.result())

    print("\nmultimodal 批次完成!")

    # ==========================================
    # 处理 multimdoal_no_user_review_output (7个)
    # ==========================================
    print("\n[2/2] 并行处理 multimdoal_no_user_review_output 的问题对话...")
    print(f"共 {len(NO_USER_REVIEW_PROBLEMS)} 个任务\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_no_user_review, cat, uid, log_dir): (cat, uid)
            for cat, uid in NO_USER_REVIEW_PROBLEMS
        }
        for future in as_completed(futures):
            results.append(future.result())

    print("\nno_user_review 批次完成!")

    # ==========================================
    # 汇总结果
    # ==========================================
    print("\n" + "=" * 60)
    print("处理结果汇总")
    print("=" * 60)

    success_count = sum(1 for r in results if r[3])
    fail_count = len(results) - success_count

    print(f"\n成功: {success_count}, 失败: {fail_count}")

    if fail_count > 0:
        print("\n失败的任务:")
        for method, cat, uid, success in results:
            if not success:
                print(f"  - [{method}] {cat} / {uid}")

    print(f"\n日志目录: {log_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
