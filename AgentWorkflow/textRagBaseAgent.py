from dotenv import load_dotenv
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import os
from helpFunctions import load_all_questions, load_all_users, export_dialogues_for_eval
from datetime import datetime
from pathlib import Path
import json
import sys
import argparse
from ragFunctions import save_chat_history, rag_search_for_recommend_agent, rag_search_for_dialogue_history_agent, append_agent_message, save_round_summary, load_previous_round_summaries, get_current_round_number
from agent_prompts import get_dialogue_history_prompt, get_recommender_prompt, get_reasoner_prompt, get_reply_prompt

# Custom logger class to capture all print outputs
class TeeLogger:
    """Captures print output to both console and file"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure immediate write

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

def make_simple_flow(user_agent, dh_agent, rec_agent, rsn_agent, reply_agent):
    """
    Flow:
      User -> DialogueHistory (handles its tool call cycles)
           -> Recommender     (handles its tool call cycles)
           -> Reasoner        (handles its tool call cycles)
           -> Reply           (generates final user response)
           -> End
    """
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

        if _has_tool_calls(last) or _is_tool_return(last):
            return last_speaker

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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Text RAG Base Agent for Shopping Recommendations")
    parser.add_argument(
        "--reply-prompt-version",
        type=str,
        default="v1",
        choices=["v1", "v2"],
        help="Version of reply agent prompt to use (default: v1, v2: adaptive/flexible)"
    )
    args = parser.parse_args()
    reply_prompt_version = args.reply_prompt_version

    load_dotenv()
    assert os.getenv("QWEN_API_KEY"), "Missing QWEN_API_KEY in .env"

    # Setup logging
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"agent_run_{ts}.log"

    # Redirect stdout to both console and log file
    logger = TeeLogger(log_file)
    sys.stdout = logger
    sys.stderr = logger  # Also capture error messages

    print(f"{'='*80}")
    print(f"Agent Workflow Execution Started")
    print(f"Timestamp: {ts}")
    print(f"Log file: {log_file}")
    print(f"{'='*80}\n")

    # question json file path
    json_path = Path("../questions_v2/Clothing_Shoes_and_Jewelry_questions_20251127_235753.json")

    all_questions_dict = load_all_questions(json_path)

    all_users_list = load_all_users(json_path)
    selected_users = all_users_list[:]

    # how many questions can run for one category
    per_user_limit = None # run all questions in the category

    # store output for json format
    results = {}

    print(f"Configuration:")
    print(f"  Reply Prompt Version: {reply_prompt_version}")
    print(f"  Selected users: {selected_users}")
    print(f"  Per user limit: {per_user_limit or 'All questions'}")
    print(f"  Question file: {json_path}")
    print()

    # Test API connection
    print("\n[TEST] Testing OpenRouter API connection...")
    try:
        import requests
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('QWEN_API_KEY')}",
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

    # Qwen3 VL 8B Instruct
    llm_config = {
            "config_list": [{
                # "model": "qwen/qwen3-vl-8b-instruct",
                "model": "qwen/qwen3-8b", # use this since all text only
                "api_key": os.getenv("QWEN_API_KEY"),
                "base_url": "https://openrouter.ai/api/v1",
                "price": [0.001, 0.002], # remove autogen price warning
                "timeout": 120,  # 2 minutes timeout
            }],
            "timeout": 120,
            "cache_seed": None,  # Disable caching for debugging
        }

    for user in selected_users:
        questions = all_questions_dict[user]
        if per_user_limit:
            questions = questions[:per_user_limit]

        # TODO: Should ask llm to summarize previous history
        dialogue_history_agent = AssistantAgent(
            name=f"Dialogue_History_{user}_agent",
            llm_config=llm_config,
            system_message=get_dialogue_history_prompt(user),
        )

        print(f"[DEBUG] Dialogue_History_agent created for user: {user}")

        recommend_agent = AssistantAgent(
            name=f"Recommender_{user}_agent",
            llm_config=llm_config,
            system_message=get_recommender_prompt(user),
        )

        reason_agent = AssistantAgent(
            name=f"Reasoner_{user}_agent",
            llm_config=llm_config,
            system_message=get_reasoner_prompt(user),
        )

        reply_agent = AssistantAgent(
            name=f"Reply_{user}_agent",
            llm_config=llm_config,
            system_message=get_reply_prompt(version=reply_prompt_version),
        )

        # Create wrapper that captures user_id for reply_agent and auto-detects round number
        def make_reply_tool_wrapper(user_id: str):
            def wrapper():
                # Auto-detect current round number at execution time
                current_round = get_current_round_number(user_id)
                summaries = load_previous_round_summaries(user_id, current_round)

                if not summaries:
                    return {
                        "user_name": user_id,
                        "current_round": current_round,
                        "previous_summaries": [],
                        "message": "This is the first round. No previous conversation history exists."
                    }
                return {
                    "user_name": user_id,
                    "current_round": current_round,
                    "previous_summaries": summaries,
                    "message": f"Found {len(summaries)} previous round(s) of conversation."
                }
            return wrapper

        reply_wrapper = make_reply_tool_wrapper(user)

        reply_agent.register_for_llm(
            name="get_previous_round_summaries",
            description=(
                "Retrieve summaries of all previous rounds in this conversation session. "
                "Use this to determine if this is a continuation and to understand the conversation context. "
                "No parameters required (user_name and round_number are automatically provided)."
            ),
        )(reply_wrapper)

        reply_agent.register_for_execution(
            name="get_previous_round_summaries"
        )(reply_wrapper)

        print(f"[DEBUG] Tool 'get_previous_round_summaries' registered for {user}")

        user_proxy_agent = UserProxyAgent(
            name=f"User_{user}_proxy_agent",
            human_input_mode="NEVER",
            code_execution_config=False,
        )

        # Create wrapper that captures user_id and llm_config for dialogue_history_agent
        def make_dh_tool_wrapper(user_id: str, llm_cfg: dict):
            def wrapper(current_query: str):
                return rag_search_for_dialogue_history_agent(
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

        print(f"[DEBUG] Tool 'rag_search_for_dialogue_history_agent' registered for {user}")

        # Create wrapper that captures user_id for recommend_agent
        def make_rec_tool_wrapper(user_id: str):
            def wrapper(user_query: str):
                return rag_search_for_recommend_agent(
                    user_name=user_id,
                    user_query=user_query
                )
            return wrapper

        rec_wrapper = make_rec_tool_wrapper(user)

        recommend_agent.register_for_llm(
            name="rag_search_for_recommend_agent",
            description=(
                "Search the vector database for product, item, or knowledge information relevant to the user's query. "
                "Use it to generate personalized shopping recommendations. "
                "Only requires 'user_query' parameter (user_name is automatically provided)."
            ),
        )(rec_wrapper)

        recommend_agent.register_for_execution(
            name="rag_search_for_recommend_agent"
        )(rec_wrapper)

        # output separated by category
        results[user] = []

        # store user chat history
        user_history = []

        print(f"\n{'='*80}")
        print(f"Processing User: {user}")
        print(f"Total Questions: {len(questions)}")
        print(f"{'='*80}\n")

        for i, q in enumerate(questions, start=1):
            print(f"\n{'-'*80}")
            print(f"[ROUND {i}/{len(questions)}] Question: {q}")
            print(f"{'-'*80}\n")

            groupchat = GroupChat(
                agents=[user_proxy_agent, dialogue_history_agent, recommend_agent, reason_agent, reply_agent],
                messages=[],  # Each round is independent; dialogue_history_agent uses RAG for history
                max_round=10, # leave a number here
                speaker_selection_method=make_simple_flow(user_proxy_agent, dialogue_history_agent, recommend_agent, reason_agent, reply_agent)
            )

            manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

            save_chat_history(f"[USER_QUESTION_{i}] [{user_proxy_agent.name}] {q}", user)
            user_proxy_agent.initiate_chat(manager, message=q)

            # Extract and save new messages from this round
            new_messages = groupchat.messages  # All messages from this independent round

            # Save agent outputs to chat history
            print(f"\n[DEBUG] Processing {len(new_messages)} new messages for Round {i}")
            for idx, msg in enumerate(new_messages):
                if not isinstance(msg, dict):
                    print(f"[DEBUG] Message {idx}: Skipped (not a dict)")
                    continue

                msg_name = msg.get("name", "")
                msg_role = msg.get("role", "")
                msg_content = msg.get("content", "")

                print(f"[DEBUG] Message {idx}: name='{msg_name}', role='{msg_role}', content_length={len(str(msg_content))}")

                # Skip tool messages
                if msg_role in ("tool", "function"):
                    print(f"[DEBUG] Message {idx}: Skipped (tool message)")
                    continue

                # Skip empty messages
                if not msg_content or str(msg_content).strip() in ("", "None"):
                    print(f"[DEBUG] Message {idx}: Skipped (empty content)")
                    continue

                # Skip user_proxy messages (already saved in line 296)
                if msg_name == user_proxy_agent.name:
                    print(f"[DEBUG] Message {idx}: Skipped (user_proxy)")
                    continue

                # Save assistant agent outputs
                if msg_name in [dialogue_history_agent.name, recommend_agent.name, reason_agent.name, reply_agent.name]:
                    print(f"[DEBUG] Message {idx}: Saving to chat_history")
                    append_agent_message(msg_content, "assistant", msg_name, user)
                else:
                    print(f"[DEBUG] Message {idx}: Not in agent list, skipped")

            # Update history for next round
            user_history.extend(new_messages)

            print(f"\n[ROUND {i} COMPLETED]\n")

        results[user] = user_history

        print(f"\n{'='*80}")
        print(f"Completed User: {user}")
        print(f"Total Rounds: {len(questions)}")
        print(f"{'='*80}\n")

        # Export dialogues for evaluation
        eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_output_path = f"./Clothing_Shoes_and_Jewelry_dial/eval_dialogues_{user}_{reply_prompt_version}_{eval_timestamp}.json"
        export_dialogues_for_eval(user, output_path=eval_output_path)

    output_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("./")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = f"shopping_text_rag_agent_output_qv2_ragreview_{reply_prompt_version}_{output_ts}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*80}")
    print(f"Agent Workflow Execution Completed")
    print(f"Output saved to: {out_path}")
    print(f"Log saved to: {log_file}")
    print(f"{'='*80}\n")

    # Restore stdout and close log file
    sys.stdout = logger.terminal
    sys.stderr = logger.terminal
    logger.close()

    print(f"Execution complete. Check {log_file} for detailed logs.")

if __name__ == "__main__":
    main()