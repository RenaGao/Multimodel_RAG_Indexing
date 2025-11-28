from dotenv import load_dotenv
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import os
from helpFunctions import load_all_questions, load_all_users
from datetime import datetime
from pathlib import Path
import json
from ragFunctions import save_chat_history, rag_search_include_image_for_recommend_agent, rag_search_for_dialogue_history_agent, rag_search_for_reason_agent

def make_save_hook(agent_name: str):
    """Save the latest inbound event for this agent before it replies."""
    def _hook(recipient, messages, sender, config):
        try:
            if not messages:
                return False, None
            
            msg = messages[-1]  # the newest inbound message for this agent
            sender_name = getattr(sender, "name", "Unknown")
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            tool_bits = []
            if msg.get("tool_calls"):
                tool_bits.append(f"tool_calls={str(msg['tool_calls'])}")
            if msg.get("function_call"):
                tool_bits.append(f"function_call={str(msg['function_call'])}")
            if role in ("tool", "function"):
                tool_bits.append(f"tool_return(name={msg.get('name')}, content={str(content)[:400]})")

            history_str = (
                f"[AGENT:{agent_name}] "
                f"[FROM:{sender_name}] "
                f"[ROLE:{role}] "
                f"[TEXT:{str(content)[:]}] "
                + (f"[TOOLS:{' | '.join(tool_bits)}]" if tool_bits else "")
            )

            save_chat_history(history_str)
        except Exception as e:
            save_chat_history(f"[AGENT:{agent_name}] [SAVE_ERROR:{e}]")

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

    json_path = "temp_user_based_question_list.json"

    all_questions_dict = load_all_questions(json_path)

    all_users_list = load_all_users(json_path)

    selected_users = all_users_list[:2]
    
    # how many questions can run for one category
    per_user_limit = None # run all questions in the category

    # store output for json format
    results = {}

    # replace Qwen3 VL 8B Instruct
    llm_config = {
            "config_list": [{
                "model": "qwen/qwen3-vl-8b-instruct",
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
                    "Your task has exactly two steps and must stop after completing them:\n"
                    "Step 1: Immediately call the tool 'rag_search_include_image_for_recommend_agent' once "
                    "to retrieve recommendation candidates relevant to the user's query and labels. "
                    "The tool returns structured JSON with text fields and image URLs for each candidate.\n"
                    "Step 2: Based solely on the tool result, generate the product recommendation(s) for the user. "
                    "Use the textual fields (title, summary, tags, price) and the provided image URLs "
                    "to support your reasoning and presentation. Do not call any other tools, and do not respond "
                    "before the tool result is returned. End your turn after providing the recommendation."
            ),
        )

        reason_agent = AssistantAgent(
            name=f"Reasoner_{user}_agent",
            llm_config=llm_config,
            system_message=(
                    "You are a Reasoning Agent that provides the rationale behind previous labels and recommendations. "
                    "Your task has exactly two steps and must stop after completing them:\n"
                    "Step 1: Immediately call the tool 'rag_search_for_reason_agent' once "
                    "to retrieve relevant dialogue and recommendation history for reasoning.\n"
                    "Step 2: Using only the tool result, explain the reasoning or justification for the prior outputs. "
                    "Do not call any other tools, and do not produce any response before the tool result is obtained. "
                    "End your turn after giving the explanation."
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
                "Use it to infer user interests, habits, or contextual labels."
            ),
        )(rag_search_for_dialogue_history_agent)

        dialogue_history_agent.register_for_execution(
            name="rag_search_for_dialogue_history_agent"
        )(rag_search_for_dialogue_history_agent)

        recommend_agent.register_for_llm(
            name="rag_search_include_image_for_recommend_agent",
            description=(
                "RAG search that returns candidates with text fields and image URLs. "
                "Use it to produce shopping recommendations enriched with visuals."
            ),
        )(rag_search_include_image_for_recommend_agent)

        recommend_agent.register_for_execution(
            name="rag_search_include_image_for_recommend_agent"
        )(rag_search_include_image_for_recommend_agent)

        reason_agent.register_for_llm(
            name="rag_search_for_reason_agent",
            description=(
                "Search the vector database for previous dialogue and recommendation records "
                "to explain the rationale behind user labels and system recommendations."
            ),
        )(rag_search_for_reason_agent)

        reason_agent.register_for_execution(
            name="rag_search_for_reason_agent"
        )(rag_search_for_reason_agent)

        user_proxy_agent.register_reply(
            [autogen.Agent, None],
            reply_func=make_save_hook(user_proxy_agent.name),
            config={},
        )

        dialogue_history_agent.register_reply(
            [autogen.Agent, None],
            reply_func=make_save_hook(dialogue_history_agent.name),
            config={},
        )

        recommend_agent.register_reply(
            [autogen.Agent, None],
            reply_func=make_save_hook(recommend_agent.name),
            config={},
        )

        reason_agent.register_reply(
            [autogen.Agent, None],
            reply_func=make_save_hook(reason_agent.name),
            config={},
        )

        # output separated by category
        results[user] = []

        # store user chat history
        user_history = []

        for i, q in enumerate(questions, start=1):

            groupchat = GroupChat(
                agents=[user_proxy_agent, dialogue_history_agent, recommend_agent, reason_agent],
                messages=list(user_history),
                max_round=10, # leave a number here
                speaker_selection_method=make_simple_flow(user_proxy_agent, dialogue_history_agent, recommend_agent, reason_agent)
            )

            manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

            save_chat_history(f"[USER_QUESTION_{i}] [{user_proxy_agent.name}] {q}")
            user_proxy_agent.initiate_chat(manager, message=q)

            new_msgs = groupchat.messages[:]
            user_history.extend(new_msgs)

        results[user] = user_history
            
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("./")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = f"shopping_image_text_rag_agent_output_{ts}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved agent outputs.")

if __name__ == "__main__":
    main()