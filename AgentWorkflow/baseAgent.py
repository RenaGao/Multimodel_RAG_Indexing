from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import os
import sys
import argparse
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from helpFunctions import load_all_questions, load_all_users, export_dialogues_for_eval
from datetime import datetime
from pathlib import Path
import json

# Thread-safe print lock
_print_lock = threading.Lock()

def thread_safe_print(*args, **kwargs):
    """Thread-safe print function."""
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
    output_dir = Path(f"./baseline_dialogues_output/{category}")
    if not output_dir.exists():
        return set()

    completed = set()
    for f in output_dir.glob("baseline_dialogues_*.json"):
        stem = f.stem
        parts = stem.split("_")
        # Format: baseline_dialogues_{user_id}_{date}_{time}
        if len(parts) >= 4:
            # user_id is between 'dialogues' and timestamp (last 2 parts)
            user_id = "_".join(parts[2:-2])
            if user_id:
                completed.add(user_id)

    return completed


def is_category_completed(questions_path, category):
    """Check if all users in questions file have output files."""
    users = set(load_all_users(questions_path))
    completed = get_completed_users(category)
    return users.issubset(completed)


def save_baseline_chat_history(user_name: str, category: str, rounds_data: list, output_dir: Path):
    """
    Save chat history in JSONL format similar to ragFunctions.py.

    Args:
        user_name: User identifier
        category: Dataset category
        rounds_data: List of round data dicts
        output_dir: Output directory path
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chat_history_path = output_dir / f"{category}_{user_name}_{timestamp}.jsonl"

    with chat_history_path.open("w", encoding="utf-8") as f:
        for round_data in rounds_data:
            f.write(json.dumps(round_data, ensure_ascii=False) + "\n")

    thread_safe_print(f"[CHAT_HISTORY] Saved to: {chat_history_path}")
    return chat_history_path


def make_simple_flow(user_agent, dh_agent, recommender, reasoner, reply_agent):
    """User -> DialogueHistory -> Recommender -> Reasoner -> Reply -> End"""
    def select_next(last_speaker, groupchat):
        if last_speaker is user_agent:
            return dh_agent
        if last_speaker is dh_agent:
            return recommender
        if last_speaker is recommender:
            return reasoner
        if last_speaker is reasoner:
            return reply_agent
        if last_speaker is reply_agent:
            return None
        return user_agent
    return select_next


def process_single_user(user: str, questions: list, category: str, llm_config: dict, output_dir: Path, per_user_limit: int = None):
    """
    Process a single user's questions through the agent workflow.
    Thread-safe function for parallel processing.

    Args:
        user: User identifier
        questions: List of questions for this user
        category: Dataset category
        llm_config: LLM configuration dict
        output_dir: Output directory for results
        per_user_limit: Optional limit on questions per user

    Returns:
        Tuple of (user, results, rounds_data)
    """
    if per_user_limit:
        questions = questions[:per_user_limit]

    thread_safe_print(f"\n{'='*80}")
    thread_safe_print(f"Processing User: {user}")
    thread_safe_print(f"Total Questions: {len(questions)}")
    thread_safe_print(f"{'='*80}\n")

    # Create agents for this user
    dialogue_history_agent = AssistantAgent(
        name=f"DialogueHistory_{user}_agent",
        llm_config=llm_config,
        system_message=(
            "You are a Dialogue History Agent.\n\n"
            "Your task is to review the accumulated conversation history and provide a concise summary.\n\n"
            "INSTRUCTIONS:\n"
            "1. Review ALL previous messages in the conversation history.\n"
            "2. Identify patterns in user preferences, interests, and past interactions.\n"
            "3. Generate a concise summary (2-4 sentences) focusing on:\n"
            "   - What the user has asked about in previous rounds\n"
            "   - User preferences or patterns observed\n"
            "   - Relevant context for the current recommendation\n\n"
            "IMPORTANT:\n"
            "- If this is the FIRST round (no previous messages), simply state: "
            "'This is the first interaction. No previous conversation history available.'\n"
            "- Use ONLY the information available in the conversation history.\n"
            "- Do NOT call any tools or functions.\n"
            "- End your turn immediately after generating the summary."
        ),
    )

    recommend_agent = AssistantAgent(
        name=f"Recommender_{user}_agent",
        llm_config=llm_config,
        system_message=(
            "You are a Shopping Recommender Agent.\n\n"
            f"STEP 1: Review the DialogueHistory_{user}_agent's summary (if available) "
            "to understand the user's conversation context and preferences.\n\n"
            "STEP 2: Based on the user's current query and any historical context, "
            "provide product recommendations.\n\n"
            "STEP 3: For each recommendation, briefly explain why it matches the user's needs.\n\n"
            "IMPORTANT:\n"
            "- Use ONLY information from the conversation history.\n"
            "- Do NOT call any tools or functions.\n"
            "- Provide 3-5 recommendations.\n"
            "- End your turn immediately after generating recommendations."
        ),
    )

    reason_agent = AssistantAgent(
        name=f"Reasoner_{user}_agent",
        llm_config=llm_config,
        system_message=(
            "You are a Reasoning Agent.\n\n"
            "Your task is to explain WHY the product recommendations were generated.\n\n"
            "INSTRUCTIONS:\n"
            "1. Review the current round's messages to extract:\n"
            f"   - User query: from User_{user}\n"
            f"   - Dialogue summary: from DialogueHistory_{user}_agent\n"
            f"   - Recommendations: from Recommender_{user}_agent\n"
            "2. Explain clearly and concisely WHY these recommendations were produced.\n"
            "3. Connect the recommendations to user preferences or query intent.\n\n"
            "IMPORTANT:\n"
            "- Use ONLY information from the conversation history.\n"
            "- Do NOT call any tools or functions.\n"
            "- Focus on the CURRENT round's context.\n"
            "- End your turn immediately after generating the explanation."
        ),
    )

    reply_agent = AssistantAgent(
        name=f"Reply_{user}_agent",
        llm_config=llm_config,
        system_message=(
            "You are a Reply Agent.\n\n"
            "Your task is to generate a friendly, natural response to the user based on the "
            "recommendations and reasoning provided by previous agents.\n\n"
            "INSTRUCTIONS:\n"
            "1. Review the conversation history to determine if this is a continuation:\n"
            "   - FIRST ROUND: Use a friendly greeting like 'Hey there!' or 'Hi!'\n"
            "   - CONTINUATION: Acknowledge context with phrases like 'Based on what we discussed earlier...'\n"
            "2. Present the recommendations in a conversational way\n"
            "3. Highlight key features that match the user's needs\n"
            "4. Keep the response friendly and concise (4-6 sentences)\n"
            "5. End with an offer to help further\n\n"
            "IMPORTANT:\n"
            "- Use ONLY information from the current conversation history.\n"
            "- Do NOT call any tools or functions.\n"
            "- End your turn immediately after generating the response."
        ),
    )

    user_proxy_agent = UserProxyAgent(
        name=f"User_{user}",
        human_input_mode="NEVER",
        code_execution_config=False,
    )

    category_history = []
    rounds_data = []
    eval_dialogue_history = []
    user_queries = []

    for i, q in enumerate(questions, start=1):
        thread_safe_print(f"\n{'-'*80}")
        thread_safe_print(f"[{user}] [ROUND {i}/{len(questions)}] Question: {q}")
        thread_safe_print(f"{'-'*80}\n")

        groupchat = GroupChat(
            agents=[user_proxy_agent, dialogue_history_agent, recommend_agent, reason_agent, reply_agent],
            messages=list(category_history),
            max_round=10,
            speaker_selection_method=make_simple_flow(user_proxy_agent, dialogue_history_agent, recommend_agent, reason_agent, reply_agent)
        )

        manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

        user_proxy_agent.initiate_chat(manager, message=q)

        # Add rate limiting delay
        time.sleep(0.5)

        new_msgs = groupchat.messages[:]

        # Build round data for chat history
        round_data = {
            "round_id": f"round_{i:03d}",
            "round_number": i,
            "user_query": q,
            "dataset_category": category,
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "conversation": []
        }

        # Extract messages for round data
        reply_content = None
        for msg in new_msgs:
            if isinstance(msg, dict):
                msg_name = msg.get("name", "")
                msg_role = msg.get("role", "")
                msg_content = msg.get("content", "")

                if msg_role in ("tool", "function"):
                    continue
                if not msg_content or str(msg_content).strip() in ("", "None"):
                    continue

                # Add to round conversation
                round_data["conversation"].append({
                    "agent": msg_name,
                    "role": "user" if msg_name == user_proxy_agent.name else "assistant",
                    "content": msg_content
                })

                if "Reply" in msg_name:
                    reply_content = msg_content

        rounds_data.append(round_data)

        # Add to eval data
        if reply_content:
            eval_dialogue_history.append({"role": "user", "content": q})
            eval_dialogue_history.append({"role": "assistant", "content": reply_content})
            user_queries.append(q)

        category_history.extend(new_msgs)
        thread_safe_print(f"[{user}] [ROUND {i} COMPLETED]\n")

    # Prepare results
    results = {
        "user": user,
        "category": category,
        "total_rounds": len(questions),
        "eval_dialogue_history": eval_dialogue_history,
        "user_queries": user_queries,
        "category_history": category_history
    }

    thread_safe_print(f"\n{'='*80}")
    thread_safe_print(f"Completed User: {user}")
    thread_safe_print(f"Total Rounds: {len(questions)}")
    thread_safe_print(f"{'='*80}\n")

    return user, results, rounds_data


def run_single_category(json_path: str, category: str, target_user: str = None, num_threads: int = 10):
    """
    Run baseline agent workflow for a single category with multi-threading.

    Args:
        json_path: Path to questions JSON file
        category: Category name
        target_user: Optional specific user to process
        num_threads: Number of parallel threads (default: 10)
    """
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

    per_user_limit = None  # run all questions

    # Create output directory
    output_dir = Path(f"./baseline_dialogues_output/{category}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Configuration:")
    print(f"  Category: {category}")
    print(f"  Total users: {len(all_users_list)}")
    print(f"  Completed users: {len(completed_users)}")
    print(f"  Remaining users: {len(selected_users)}")
    print(f"  Threads: {num_threads}")
    print(f"  Per user limit: {per_user_limit or 'All questions'}")
    print(f"  Question file: {json_path}")
    print(f"  Output dir: {output_dir}")
    print()

    if not selected_users:
        print(f"No users to process for category {category}")
        return {}

    # LLM config
    llm_config = {
        "config_list": [{
            "model": "qwen/qwen3-8b",
            "api_key": os.getenv("OPENROUTER_API_KEY"),
            "base_url": "https://openrouter.ai/api/v1",
            "price": [0.001, 0.002]
        }]
    }

    all_results = {}

    # Process users with thread pool
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {}
        for user in selected_users:
            questions = all_questions_dict[user]
            future = executor.submit(
                process_single_user,
                user, questions, category, llm_config, output_dir, per_user_limit
            )
            futures[future] = user

        for future in as_completed(futures):
            user = futures[future]
            try:
                user_id, results, rounds_data = future.result()
                all_results[user_id] = results

                # Save chat history
                save_baseline_chat_history(user_id, category, rounds_data, output_dir)

                # Save dialogue output for evaluation
                if results.get("user_queries"):
                    num_queries = len(results["user_queries"])
                    query_summary = ", ".join([f"({i+1}) {q}" for i, q in enumerate(results["user_queries"])])
                    user_goal = f"Multi-round shopping session with {num_queries} queries: {query_summary}"

                    eval_entry = {
                        "dialogue_id": user_id,
                        "user_goal": user_goal,
                        "dialogue_history": results["eval_dialogue_history"]
                    }

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    eval_path = output_dir / f"baseline_dialogues_{user_id}_{timestamp}.json"
                    with open(eval_path, "w", encoding="utf-8") as f:
                        json.dump([eval_entry], f, ensure_ascii=False, indent=2)

                    thread_safe_print(f"[EXPORT] Saved evaluation data to: {eval_path}")

            except Exception as e:
                thread_safe_print(f"[ERROR] Failed to process user {user}: {e}")
                import traceback
                traceback.print_exc()

    return all_results


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Baseline Agent for Shopping Recommendations")
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
    parser.add_argument(
        "--threads",
        type=int,
        default=10,
        help="Number of parallel threads (default: 10)"
    )
    args = parser.parse_args()

    load_dotenv()
    assert os.getenv("OPENROUTER_API_KEY"), "Missing OPENROUTER_API_KEY in .env"

    # Setup logging
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"baseline_agent_run_{ts}.log"

    # Redirect stdout to both console and log file
    logger = TeeLogger(log_file)
    sys.stdout = logger
    sys.stderr = logger

    print(f"{'='*80}")
    print(f"Baseline Agent Workflow Execution Started")
    print(f"Timestamp: {ts}")
    print(f"Log file: {log_file}")
    print(f"Threads: {args.threads}")
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

    # Run based on mode
    if args.all:
        # Batch mode: process all categories
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
            print(f"# Processing Category: {category}")
            print(f"# Question file: {qf}")
            print(f"{'#'*80}\n")

            try:
                run_single_category(qf, category, num_threads=args.threads)
            except Exception as e:
                print(f"Error processing category {category}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n{'='*80}")
        print(f"All Categories Completed")
        print(f"Output saved to: ./baseline_dialogues_output/{{category}}/")
        print(f"Log saved to: {log_file}")
        print(f"{'='*80}\n")

    elif args.category:
        # Category mode: find and process specific category
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

        run_single_category(qf, category, target_user=args.user_id, num_threads=args.threads)

        print(f"\n{'='*80}")
        print(f"Category Completed: {category}")
        print(f"Output saved to: ./baseline_dialogues_output/{category}/")
        print(f"Log saved to: {log_file}")
        print(f"{'='*80}\n")

    else:
        # Single category mode
        if args.input:
            json_path = args.input
        else:
            # Default: try to find any question file
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

        run_single_category(json_path, category, num_threads=args.threads)

        print(f"\n{'='*80}")
        print(f"Baseline Agent Workflow Execution Completed")
        print(f"Output saved to: ./baseline_dialogues_output/{category}/")
        print(f"Log saved to: {log_file}")
        print(f"{'='*80}\n")

    # Restore stdout and close log file
    sys.stdout = logger.terminal
    sys.stderr = logger.terminal
    logger.close()

    print(f"Execution complete. Check {log_file} for detailed logs.")


if __name__ == "__main__":
    main()
