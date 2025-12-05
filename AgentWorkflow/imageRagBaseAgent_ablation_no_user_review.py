"""
[ABLATION STUDY] Multimodal RAG Agent WITHOUT User Review Personalization

This is an ablation study version that removes user-level review personalization.
- Keeps: Triple fusion RAG (metadata + item_review + caption)
- Removes: User's relevant reviews selection and personalization in prompts

Compare with imageRagBaseAgent.py to measure the impact of user review personalization.
"""
from dotenv import load_dotenv
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import os
import shutil
from helpFunctions import load_all_questions, load_all_users, export_dialogues_for_eval
from datetime import datetime
from pathlib import Path
import json
import sys
import argparse
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from ragFunctions import (
    save_chat_history_openrouter,
    rag_search_for_recommend_agent_multimodal_no_user_review,  # [ABLATION] No user review version
    rag_search_for_dialogue_history_agent_openrouter,
    append_agent_message,
    save_round_summary,
    load_previous_round_summaries,
    get_current_round_number,
    set_dataset_category,
    _user_paths
)
from agent_prompts import (
    get_dialogue_history_prompt,
    get_recommender_prompt_no_personalization,  # [ABLATION] No personalization
    get_reasoner_prompt_no_personalization,      # [ABLATION] No personalization
    get_reply_prompt_no_personalization          # [ABLATION] No personalization
)

# 多线程配置
NUM_WORKERS = 5

# 线程安全的打印锁
_print_lock = threading.Lock()

def thread_safe_print(*args, **kwargs):
    """线程安全的打印函数"""
    with _print_lock:
        print(*args, **kwargs)

# Custom logger class to capture all print outputs
class TeeLogger:
    """Captures print output to both console and file"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')
        self.lock = threading.Lock()

    def write(self, message):
        with self.lock:
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

    def flush(self):
        with self.lock:
            self.terminal.flush()
            self.log.flush()

    def close(self):
        self.log.close()

def get_all_question_files(questions_dir="../questions_v3"):
    """Get all question files from questions_v3 folder."""
    files = []
    if os.path.exists(questions_dir):
        for f in os.listdir(questions_dir):
            if f.endswith(".json") and "_questions_" in f:
                files.append(os.path.join(questions_dir, f))
    return sorted(files)

def extract_category(filename):
    """Extract category from filename."""
    basename = os.path.basename(filename)
    return basename.split("_questions_")[0]

def find_category_file(category_name, questions_dir="../questions_v3"):
    """Find question file for a specific category."""
    question_files = get_all_question_files(questions_dir)
    for qf in question_files:
        if extract_category(qf) == category_name:
            return qf
    return None

def get_completed_users(category):
    """Get set of user IDs that have completed output files."""
    # [ABLATION] Use different output directory
    output_dir = Path(f"./ablation_no_user_review_output/{category}")
    if not output_dir.exists():
        return set()

    completed = set()
    for f in output_dir.glob("ablation_no_review_dialogues_*.json"):
        stem = f.stem
        parts = stem.split("_")
        dialogues_idx = parts.index("dialogues") if "dialogues" in parts else -1
        if dialogues_idx >= 0 and len(parts) >= dialogues_idx + 4:
            user_id = "_".join(parts[dialogues_idx + 1 : -2])
            if user_id:
                completed.add(user_id)

    return completed

def is_category_completed(questions_path, category):
    """Check if all users in questions file have output files."""
    users = set(load_all_users(questions_path))
    completed = get_completed_users(category)
    return users.issubset(completed)

def clear_single_user_storage(user_name: str):
    """Clear storage for a single user (thread-safe)."""
    from ragFunctions import _safe_user_dir, USERS_DIR
    udir = USERS_DIR / _safe_user_dir(user_name)
    if udir.exists():
        shutil.rmtree(udir)
        thread_safe_print(f"[CLEAR] Cleared storage for user: {user_name}")

def make_simple_flow(user_agent, dh_agent, rec_agent, rsn_agent, reply_agent,
                     agent_retry_counts, agent_tool_called, max_retries=3):
    """
    Flow:
      User -> DialogueHistory -> Recommender -> Reasoner -> Reply -> End
    """
    tool_required_agents = {dh_agent, rec_agent}

    def _last(gc):
        return (gc.messages or [None])[-1] or {}

    def _has_tool_calls(msg: dict) -> bool:
        return isinstance(msg, dict) and bool(msg.get("tool_calls"))

    def _is_tool_return(msg: dict) -> bool:
        return isinstance(msg, dict) and msg.get("role") == "tool"

    def select_next(last_speaker, groupchat):
        last = _last(groupchat)

        if last_speaker is None and not groupchat.messages:
            return user_agent

        if _is_tool_return(last):
            if last_speaker and last_speaker in tool_required_agents:
                agent_tool_called.add(last_speaker.name)
            return last_speaker

        if _has_tool_calls(last):
            return last_speaker

        if last_speaker in tool_required_agents:
            agent_name = last_speaker.name
            if agent_name in agent_tool_called:
                agent_tool_called.discard(agent_name)
                agent_retry_counts[agent_name] = 0
            else:
                current_retries = agent_retry_counts.get(agent_name, 0)
                if current_retries < max_retries:
                    agent_retry_counts[agent_name] = current_retries + 1
                    thread_safe_print(f"[RETRY] {agent_name} didn't call tool, retry {current_retries + 1}/{max_retries}")
                    return last_speaker
                else:
                    thread_safe_print(f"[WARNING] {agent_name} failed to call tool after {max_retries} retries, continuing...")
                    agent_retry_counts[agent_name] = 0

        if last_speaker is user_agent:
            return dh_agent
        if last_speaker is dh_agent:
            return rec_agent
        if last_speaker is rec_agent:
            return rsn_agent
        if last_speaker is rsn_agent:
            return reply_agent
        if last_speaker is reply_agent:
            return None

        return user_agent

    return select_next


def process_single_user(user: str, questions: list, category: str, llm_config: dict, per_user_limit: int = None):
    """处理单个用户的所有问题（在独立线程中运行）"""
    thread_safe_print(f"\n[THREAD-{threading.current_thread().name}] Starting user: {user}")

    if per_user_limit:
        questions = questions[:per_user_limit]

    agent_retry_counts = {}
    agent_tool_called = set()

    # Dialogue History Agent - same as original
    dialogue_history_agent = AssistantAgent(
        name=f"Dialogue_History_{user}_agent",
        llm_config=llm_config,
        system_message=get_dialogue_history_prompt(user),
    )

    # [ABLATION] Recommender Agent - NO PERSONALIZATION
    recommend_agent = AssistantAgent(
        name=f"Recommender_{user}_agent",
        llm_config=llm_config,
        system_message=get_recommender_prompt_no_personalization(user),
    )

    # [ABLATION] Reasoner Agent - NO PERSONALIZATION
    reason_agent = AssistantAgent(
        name=f"Reasoner_{user}_agent",
        llm_config=llm_config,
        system_message=get_reasoner_prompt_no_personalization(user),
    )

    # [ABLATION] Reply Agent - NO PERSONALIZATION
    reply_agent = AssistantAgent(
        name=f"Reply_{user}_agent",
        llm_config=llm_config,
        system_message=get_reply_prompt_no_personalization(),
    )

    user_proxy_agent = UserProxyAgent(
        name=f"User_{user}_proxy_agent",
        human_input_mode="NEVER",
        code_execution_config=False,
    )

    # Dialogue History tool wrapper
    def make_dh_tool_wrapper(user_id: str, llm_cfg: dict):
        def wrapper(current_query: str):
            return rag_search_for_dialogue_history_agent_openrouter(
                user_name=user_id,
                current_query=current_query,
                llm_config=llm_cfg
            )
        return wrapper

    dh_wrapper = make_dh_tool_wrapper(user, llm_config)

    dialogue_history_agent.register_for_llm(
        name="rag_search_for_dialogue_history_agent",
        description=(
            "Retrieve relevant user dialogue history and preference records from the vector database. "
            "Use it to infer user interests, habits, or preference. "
            "Only requires 'current_query' parameter (user_name is automatically provided)."
        ),
    )(dh_wrapper)

    dialogue_history_agent.register_for_execution(
        name="rag_search_for_dialogue_history_agent"
    )(dh_wrapper)

    # [ABLATION] Recommender tool wrapper - NO USER REVIEW VERSION
    def make_rec_tool_wrapper(user_id: str):
        def wrapper(user_query: str):
            return rag_search_for_recommend_agent_multimodal_no_user_review(
                user_name=user_id,
                user_query=user_query
            )
        return wrapper

    rec_wrapper = make_rec_tool_wrapper(user)

    recommend_agent.register_for_llm(
        name="rag_search_for_recommend_agent",
        description=(
            "Search the vector database for product, item, or knowledge information relevant to the user's query. "
            "Uses multimodal fusion (metadata + reviews + image captions) for recommendations. "
            "Only requires 'user_query' parameter (user_name is automatically provided)."
        ),
    )(rec_wrapper)

    recommend_agent.register_for_execution(
        name="rag_search_for_recommend_agent"
    )(rec_wrapper)

    user_history = []

    thread_safe_print(f"\n{'='*80}")
    thread_safe_print(f"[ABLATION] Processing User: {user}")
    thread_safe_print(f"[ABLATION] Mode: NO USER REVIEW PERSONALIZATION")
    thread_safe_print(f"[THREAD] Total Questions: {len(questions)}")
    thread_safe_print(f"{'='*80}\n")

    for i, q in enumerate(questions, start=1):
        thread_safe_print(f"\n[{user}] Round {i}/{len(questions)}: {q[:50]}...")

        groupchat = GroupChat(
            agents=[user_proxy_agent, dialogue_history_agent, recommend_agent, reason_agent, reply_agent],
            messages=[],
            max_round=25,
            speaker_selection_method=make_simple_flow(
                user_proxy_agent, dialogue_history_agent, recommend_agent, reason_agent, reply_agent,
                agent_retry_counts, agent_tool_called
            )
        )

        manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

        save_chat_history_openrouter(f"[USER_QUESTION_{i}] [{user_proxy_agent.name}] {q}", user)
        user_proxy_agent.initiate_chat(manager, message=q)

        new_messages = groupchat.messages

        for idx, msg in enumerate(new_messages):
            if not isinstance(msg, dict):
                continue
            msg_name = msg.get("name", "")
            msg_role = msg.get("role", "")
            msg_content = msg.get("content", "")

            if msg_role in ("tool", "function"):
                continue
            if not msg_content or str(msg_content).strip() in ("", "None"):
                continue
            if msg_name == user_proxy_agent.name:
                continue

            if msg_name in [dialogue_history_agent.name, recommend_agent.name, reason_agent.name, reply_agent.name]:
                append_agent_message(msg_content, "assistant", msg_name, user)

        user_history.extend(new_messages)
        thread_safe_print(f"[{user}] Round {i} completed")

    thread_safe_print(f"\n[THREAD] Completed User: {user}, Total Rounds: {len(questions)}")

    # [ABLATION] Export to different directory
    eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = Path(f"./ablation_no_user_review_output/{category}")
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_output_path = eval_dir / f"ablation_no_review_dialogues_{user}_{eval_timestamp}.json"

    export_result = export_dialogues_for_eval(user, output_path=str(eval_output_path))
    if export_result is None:
        thread_safe_print(f"[WARNING] Failed to export dialogues for user {user}")
    else:
        thread_safe_print(f"[SUCCESS] Exported dialogues to: {export_result}")

    # Backup chat_history.jsonl
    up = _user_paths(user)
    if up["chat_history"].exists():
        backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = eval_dir / f"{category}_{user}_{backup_timestamp}.jsonl"
        shutil.copy2(up["chat_history"], backup_path)
        thread_safe_print(f"[BACKUP] Chat history saved to: {backup_path}")

    clear_single_user_storage(user)

    return user_history


def run_single_category(json_path, category, target_user=None):
    """Run agent workflow for a single category with multi-threading."""
    set_dataset_category(category)

    all_questions_dict = load_all_questions(json_path)
    all_users_list = load_all_users(json_path)

    if target_user:
        if target_user not in all_users_list:
            print(f"User '{target_user}' not found in {json_path}")
            return {}
        selected_users = [target_user]
        completed_users = set()
    else:
        completed_users = get_completed_users(category)
        selected_users = [u for u in all_users_list if u not in completed_users]

    per_user_limit = None

    print(f"Configuration:")
    print(f"  Category: {category}")
    print(f"  Total users: {len(all_users_list)}")
    print(f"  Completed users: {len(completed_users)}")
    print(f"  Remaining users: {len(selected_users)}")
    print(f"  Per user limit: {per_user_limit or 'All questions'}")
    print(f"  Question file: {json_path}")
    print(f"  [ABLATION] Mode: NO USER REVIEW PERSONALIZATION")
    print(f"  RAG Mode: MULTIMODAL (metadata + review + caption) - NO user review selection")
    print(f"  Parallel Workers: {NUM_WORKERS}")
    print()

    llm_config = {
        "config_list": [{
            "model": "qwen/qwen3-8b",
            "api_key": os.getenv("OPENROUTER_API_KEY"),
            "base_url": "https://openrouter.ai/api/v1",
            "price": [0.001, 0.002],
            "timeout": 120,
            "temperature": 0.3,
        }],
        "timeout": 120,
        "cache_seed": None,
    }

    results = {}

    if len(selected_users) == 0:
        print("No users to process.")
        return results

    print(f"\n{'='*80}")
    print(f"Starting parallel processing with {NUM_WORKERS} workers...")
    print(f"{'='*80}\n")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(
                process_single_user,
                user,
                all_questions_dict[user],
                category,
                llm_config,
                per_user_limit
            ): user
            for user in selected_users
        }

        completed_count = 0
        total_count = len(futures)

        for future in as_completed(futures):
            user = futures[future]
            completed_count += 1
            try:
                result = future.result()
                results[user] = result
                print(f"\n[PROGRESS] {completed_count}/{total_count} users completed. Latest: {user}")
            except Exception as e:
                print(f"\n[ERROR] User {user} failed with exception: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n{'='*80}")
    print(f"All users completed: {len(results)}/{len(selected_users)}")
    print(f"{'='*80}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="[ABLATION] Multimodal RAG Agent WITHOUT User Review Personalization"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all categories from questions_v3/ folder"
    )
    parser.add_argument(
        "--questions-dir",
        type=str,
        default="../questions_v3",
        help="Directory containing question files (default: ../questions_v3)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to a specific questions JSON file (for single category mode)"
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Process a specific category by name (e.g., 'Appliances')"
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default=None,
        help="Process a specific user ID (requires --category)"
    )
    args = parser.parse_args()

    load_dotenv()
    assert os.getenv("OPENROUTER_API_KEY"), "Missing OPENROUTER_API_KEY in .env"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"ablation_no_user_review_run_{ts}.log"

    logger = TeeLogger(log_file)
    sys.stdout = logger
    sys.stderr = logger

    print(f"{'='*80}")
    print(f"[ABLATION STUDY] Agent Workflow WITHOUT User Review Personalization")
    print(f"RAG Mode: MULTIMODAL (metadata + review + caption) - NO user review selection")
    print(f"Parallel Workers: {NUM_WORKERS}")
    print(f"Timestamp: {ts}")
    print(f"Log file: {log_file}")
    print(f"{'='*80}\n")

    # Test API connection
    print("\n[TEST] Testing OpenRouter API connection...")
    try:
        import requests
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "qwen/qwen3-8b",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10
            },
            timeout=30
        )
        print(f"[TEST] API responded with status: {response.status_code}")
        if response.status_code != 200:
            print(f"[TEST] API Error: {response.text}")
    except Exception as e:
        print(f"[TEST] API Test Failed: {e}")
    print()

    if args.all:
        question_files = get_all_question_files(args.questions_dir)
        if not question_files:
            print(f"No question files found in {args.questions_dir}")
            sys.exit(1)

        print(f"Found {len(question_files)} categories to process:")
        for qf in question_files:
            print(f"  - {extract_category(qf)}: {qf}")
        print()

        for qf in question_files:
            category = extract_category(qf)

            if is_category_completed(qf, category):
                print(f"Skipping {category} (already completed)")
                continue

            print(f"\n{'#'*80}")
            print(f"# [ABLATION] Processing Category: {category}")
            print(f"# Question file: {qf}")
            print(f"{'#'*80}\n")

            try:
                run_single_category(qf, category)
            except Exception as e:
                print(f"Error processing category {category}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n{'='*80}")
        print(f"[ABLATION] All Categories Completed")
        print(f"Output saved to: ./multimdoal_no_user_review_output/{{category}}/")
        print(f"Log saved to: {log_file}")
        print(f"{'='*80}\n")

    elif args.category:
        qf = find_category_file(args.category, args.questions_dir)
        if not qf:
            print(f"Category '{args.category}' not found in {args.questions_dir}")
            sys.exit(1)

        category = args.category
        print(f"Category mode: {category}")
        print(f"Question file: {qf}")
        if args.user_id:
            print(f"Target user: {args.user_id}")
        print()

        run_single_category(qf, category, target_user=args.user_id)

        print(f"\n{'='*80}")
        print(f"[ABLATION] Category Completed: {category}")
        print(f"Output saved to: ./ablation_no_user_review_output/{category}/")
        print(f"Log saved to: {log_file}")
        print(f"{'='*80}\n")

    else:
        if args.input:
            json_path = args.input
        else:
            json_path = "../questions_v3/Clothing_Shoes_and_Jewelry_questions_20251127_235753.json"
            if not os.path.exists(json_path):
                question_files = get_all_question_files(args.questions_dir)
                if question_files:
                    json_path = question_files[0]
                else:
                    print(f"No question files found. Please specify --input or use --all mode.")
                    sys.exit(1)

        category = extract_category(json_path)
        print(f"Single category mode: {category}")
        print(f"Question file: {json_path}")
        print()

        run_single_category(json_path, category)

        print(f"\n{'='*80}")
        print(f"[ABLATION] Execution Completed")
        print(f"Output saved to: ./ablation_no_user_review_output/{category}/")
        print(f"Log saved to: {log_file}")
        print(f"{'='*80}\n")

    sys.stdout = logger.terminal
    sys.stderr = logger.terminal
    logger.close()

    print(f"Execution complete. Check {log_file} for detailed logs.")

if __name__ == "__main__":
    main()
