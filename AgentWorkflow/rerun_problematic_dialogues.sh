#!/bin/bash
# 重新处理有问题的对话脚本
# 根据 problematic_dialogues.csv 中记录的问题对话

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "重新处理有问题的对话"
echo "=========================================="

# ==========================================
# multimodal_dialogues_output 的问题对话 (6个)
# 使用 imageRagBaseAgent.py
# ==========================================
echo ""
echo "[1/2] 处理 multimodal_dialogues_output 的问题对话..."
echo ""

# 定义问题对话: category|user_id
MULTIMODAL_PROBLEMS=(
    "Beauty_and_Personal_Care|AEZP6Z2C5AVQDZAJECQYZWQRNG3Q"
    "Sports_and_Outdoors|AGZZXSMMS4WRHHJRBUJZI4FZDHKQ"
    "Amazon_Fashion|AG73BVBKUOH22USSFJA5ZWL7AKXA"
    "Office_Products|AG375WAXLZ7PIOQKIQ6KQB4J3JVQ"
    "Health_and_Household|AG73BVBKUOH22USSFJA5ZWL7AKXA"
    "Health_and_Personal_Care|AGNUU44ETNXLBOUB53LJGFJP3SQA"
)

for item in "${MULTIMODAL_PROBLEMS[@]}"; do
    IFS='|' read -r category user_id <<< "$item"

    echo "----------------------------------------"
    echo "处理: multimodal - $category / $user_id"
    echo "----------------------------------------"

    # 删除旧的输出文件
    OUTPUT_DIR="./dialogue_outputs/multimodal_dialogues_output/${category}"
    if [ -d "$OUTPUT_DIR" ]; then
        # JSON文件: multimodal_rag_dialogues_{user_id}_{timestamp}.json
        echo "删除旧JSON文件: ${OUTPUT_DIR}/multimodal_rag_dialogues_${user_id}_*.json"
        rm -f "${OUTPUT_DIR}"/multimodal_rag_dialogues_${user_id}_*.json

        # JSONL文件: {category}_{user_id}_{timestamp}.jsonl
        echo "删除旧JSONL文件: ${OUTPUT_DIR}/${category}_${user_id}_*.jsonl"
        rm -f "${OUTPUT_DIR}"/${category}_${user_id}_*.jsonl
    fi

    # 重新运行
    echo "重新运行: python imageRagBaseAgent.py --category $category --user-id $user_id"
    python imageRagBaseAgent.py --category "$category" --user-id "$user_id"

    echo ""
done

# ==========================================
# multimdoal_no_user_review_output 的问题对话 (7个)
# 使用 imageRagBaseAgent_ablation_no_user_review.py
# ==========================================
echo ""
echo "[2/2] 处理 multimdoal_no_user_review_output 的问题对话..."
echo ""

# 定义问题对话: category|user_id
NO_USER_REVIEW_PROBLEMS=(
    "Pet_Supplies|AEYVPPWR4CIKWX4BGYKCBCDL2CZQ"
    "Pet_Supplies|AG73BVBKUOH22USSFJA5ZWL7AKXA"
    "Toys_and_Games|AEIIRIHLIYKQGI7ZOCIJTRDF5NPQ"
    "Office_Products|AEIIRIHLIYKQGI7ZOCIJTRDF5NPQ"
    "Digital_Music|AGAFM74L2RIJ5O36NNYH4Z5ISQNQ"
    "Patio_Lawn_and_Garden|AGUTZC4GHLTGYHA3KBEDRF6MHB6A"
    "Health_and_Household|AHPGHDFIU3BUB3RQBP56RQQA7W4Q"
)

for item in "${NO_USER_REVIEW_PROBLEMS[@]}"; do
    IFS='|' read -r category user_id <<< "$item"

    echo "----------------------------------------"
    echo "处理: no_user_review - $category / $user_id"
    echo "----------------------------------------"

    # 删除旧的输出文件
    OUTPUT_DIR="./dialogue_outputs/multimdoal_no_user_review_output/${category}"
    if [ -d "$OUTPUT_DIR" ]; then
        # JSON文件: multimodal_no_user_review_dialogues_{user_id}_{timestamp}.json
        echo "删除旧JSON文件: ${OUTPUT_DIR}/multimodal_no_user_review_dialogues_${user_id}_*.json"
        rm -f "${OUTPUT_DIR}"/multimodal_no_user_review_dialogues_${user_id}_*.json

        # JSONL文件: {category}_{user_id}_{timestamp}.jsonl
        echo "删除旧JSONL文件: ${OUTPUT_DIR}/${category}_${user_id}_*.jsonl"
        rm -f "${OUTPUT_DIR}"/${category}_${user_id}_*.jsonl
    fi

    # 重新运行
    echo "重新运行: python imageRagBaseAgent_ablation_no_user_review.py --category $category --user-id $user_id"
    python imageRagBaseAgent_ablation_no_user_review.py --category "$category" --user-id "$user_id"

    echo ""
done

echo "=========================================="
echo "全部完成!"
echo "=========================================="
