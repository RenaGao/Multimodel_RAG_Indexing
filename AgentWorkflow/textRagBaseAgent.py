from dotenv import load_dotenv
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import os
from helpFunctions import load_all_questions, load_all_users
from datetime import datetime
from pathlib import Path
import json
from ragFunctions import save_chat_history, rag_search_for_recommend_agent, rag_search_for_dialogue_history_agent

# TODO: can be changed based on Rag need
def make_save_hook(agent_name: str, user_name: str):
    """Save a compact version of the latest inbound event for this agent before it replies."""
    def _hook(recipient, messages, sender, config):
        try:
            if not messages:
                return False, None

            msg = messages[-1]  
            role = msg.get("role", "")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls") or msg.get("function_call")

            if role in ("tool", "function"):
                return False, None

            if (not content or str(content).strip() in ("", "None")) and tool_calls:
                return False, None

            text = ""
            if isinstance(content, str):
                text = content

            elif isinstance(content, list):
                parts = []
                for c in content:
                    if not isinstance(c, dict):
                        continue
                    if c.get("type") == "text" and "text" in c:
                        parts.append(str(c["text"]))
                    elif "content" in c:
                        parts.append(str(c["content"]))
                text = " ".join(parts)

            else:
                text = str(content)

            text = (text or "").strip()
            if not text:
                return False, None 
            
            clean = f"[{role}] {text}" if role else text

            save_chat_history(clean, user_name)

        except Exception as e:
            save_chat_history(f"[hook_error] {type(e).__name__}: {e}", user_name)

        return False, None

    return _hook

def make_simple_flow(user_agent, dh_agent, rec_agent, rsn_agent):
    """
    Flow:
      User -> DialogueHistory (handles its tool call cycles)
           -> Recommender     (handles its tool call cycles)
           -> Reasoner        (handles its tool call cycles)
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
            return None

        return user_agent

    return select_next

def main():

    load_dotenv()
    assert os.getenv("QWEN_API_KEY"), "Missing QWEN_API_KEY in .env"

    # question json file path
    json_path = Path("../questions/Amazon_Fashion_questions_20251110_214618.json")

    all_questions_dict = load_all_questions(json_path)

    all_users_list = load_all_users(json_path)
    selected_users = all_users_list[:1]
    
    # how many questions can run for one category
    per_user_limit = None # run all questions in the category

    # store output for json format
    results = {}

    # Qwen3 VL 8B Instruct
    llm_config = {
            "config_list": [{
                # "model": "qwen/qwen3-vl-8b-instruct",
                "model": "qwen/qwen3-8b", # use this since all text only
                "api_key": os.getenv("QWEN_API_KEY"),
                "base_url": "https://openrouter.ai/api/v1",
                "price": [0.001, 0.002] # remove autogen price warning
            }]
        }

    for user in selected_users:
        questions = all_questions_dict[user]
        if per_user_limit:
            questions = questions[:per_user_limit]

        dialogue_history_agent = AssistantAgent(
            name=f"Dialogue_History_{user}_agent",
            llm_config=llm_config,
            system_message=(
                    "You are a Dialogue History Agent. "
                    "Your task has exactly two steps and must stop after completing them:\n"
                    "Step 1: Immediately call the tool 'rag_search_for_dialogue_history_agent' once "
                    f"to retrieve past dialogue and preference information for the user '{user}'.\n"
                    "Step 2: Using only the tool result, generate concise summary of preferences. "
                    "Do not call any other tools, and do not produce any response before the tool result is obtained. "
                    "End your turn after generating the final response."
            ),
        )

        recommend_agent = AssistantAgent(
            name=f"Recommender_{user}_agent",
            llm_config=llm_config,
            system_message=(
                "You are a Shopping Recommender Agent.\n"
                "Your task has exactly THREE steps and must stop after completing them.\n\n"

                "Step 1: Immediately and automatically call the tool 'rag_search_for_recommend_agent' ONCE. "
                "Use the user's query exactly as provided.\n"

                "STEP 2: Based on the tool result, generate the product recommendation(s) for the user. "
                "Do NOT call any additional tools.\n"

                f"STEP 3: Use the Dialogue_History_{user}_agent's final output included earlier in the conversation "
                "to FILTER or RERANK the recommendations.\n"

                "FINAL OUTPUT REQUIREMENTS:\n"
                "1. Provide the final, reranked recommendation list.\n"
                f"2. Use ONLY information from: (1) the RAG tool result, and (2) the Dialogue_History_{user}_agent summary.\n"
                "End your turn immediately after producing the recommendation output."
            ),
        )

        reason_agent = AssistantAgent(
            name=f"Reasoner_{user}_agent",
            llm_config=llm_config,
            system_message=(
                "You are a Reasoning Agent. "
                "Your task is to explain WHY the product recommendations were generated.\n"

                "You are provided with the FULL message history of the groupchat, "
                "which may contain many previous rounds.\n\n"

                "IMPORTANT RULES:\n"
                "1. Only consider messages that belong to the **current question round**.\n"
                "2. Ignore ALL messages that appear before this boundary.\n\n"

                "Within the current round, extract exactly these fields:\n"
                "- user_query:   from the first User_*_proxy_agent message after the boundary\n"
                "- dialogue_tool_result:    from the tool message of Dialogue_History_*_agent\n"
                f"- dialogue_summary:         from the assistant output of Dialogue_History_{user}_agent\n"
                "- recommendation_tool_result:   from the tool message of Recommender_*_agent\n"
                f"- final_recommendation:         from the assistant output of Recommender_{user}_agent\n\n"

                "FINAL OUTPUT REQUIREMENTS:\n"
                "1. Read ONLY these extracted fields from this round.\n"
                "2. Explain clearly and concisely WHY the recommendation was produced.\n"
                "3. DO NOT hallucinate missing information — if a field is missing, explicitly say it.\n"
                "4. DO NOT use or reference any information from previous rounds.\n\n"
                
                "End your turn after producing the explanation."
            ),
        )

        user_proxy_agent = UserProxyAgent(
            name=f"User_{user}_proxy_agent",
            human_input_mode="NEVER",
            code_execution_config=False,
        )

        dialogue_history_agent.register_for_llm(
            name="rag_search_for_dialogue_history_agent",
            description=(
                "Retrieve relevant user dialogue history and preference records from the vector database. "
                "Use it to infer user interests, habits, or preference."
            ),
        )(rag_search_for_dialogue_history_agent)

        dialogue_history_agent.register_for_execution(
            name="rag_search_for_dialogue_history_agent"
        )(rag_search_for_dialogue_history_agent)

        recommend_agent.register_for_llm(
            name="rag_search_for_recommend_agent",
            description=(
                "Search the vector database for product, item, or knowledge information relevant to the user's query. "
                "Use it to generate personalized shopping recommendations."
            ),
        )(rag_search_for_recommend_agent)

        recommend_agent.register_for_execution(
            name="rag_search_for_recommend_agent"
        )(rag_search_for_recommend_agent)

        user_proxy_agent.register_reply(
            [autogen.Agent, None],
            reply_func=make_save_hook(user_proxy_agent.name, user),
            config={},
        )

        dialogue_history_agent.register_reply(
            [autogen.Agent, None],
            reply_func=make_save_hook(dialogue_history_agent.name, user),
            config={},
        )

        recommend_agent.register_reply(
            [autogen.Agent, None],
            reply_func=make_save_hook(recommend_agent.name, user),
            config={},
        )

        reason_agent.register_reply(
            [autogen.Agent, None],
            reply_func=make_save_hook(reason_agent.name, user),
            config={},
        )

        # output separated by category
        results[user] = []

        # store user chat history
        user_history = []

        for i, q in enumerate(questions, start=1):
            if i == 2:
                break
            groupchat = GroupChat(
                agents=[user_proxy_agent, dialogue_history_agent, recommend_agent, reason_agent],
                messages=list(user_history),
                max_round=10, # leave a number here
                speaker_selection_method=make_simple_flow(user_proxy_agent, dialogue_history_agent, recommend_agent, reason_agent)
            )

            manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

            save_chat_history(f"[USER_QUESTION_{i}] [{user_proxy_agent.name}] {q}", user)
            user_proxy_agent.initiate_chat(manager, message=q)

            user_history.extend(groupchat.messages[len(user_history)])

        results[user] = user_history
            
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("./")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = f"shopping_text_rag_agent_output_{ts}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved agent outputs.")

if __name__ == "__main__":
    main()