from dotenv import load_dotenv
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import os
from helpFunctions import load_all_questions, load_all_users
from datetime import datetime
from pathlib import Path
import json
from ragFunctions import save_chat_history, rag_search_for_recommend_agent, rag_search_for_dialogue_history_agent, rag_search_for_reason_agent


def make_save_hook(agent_name: str):
    """save chat into vector database"""
    def _hook(recipient, messages, sender, config):
        if messages:
            msg = messages[-1]
            save_chat_history(f"[{agent_name}] {str(msg)}")
        return False, None
    return _hook

def make_simple_flow(user_agent, dh_agent, rec_agent, rsn_agent):
    """User -> DialogueHistory -> Recommend -> Reason -> End"""
    def select_next(last_speaker, _groupchat):
        if last_speaker is user_agent:
            return dh_agent
        if last_speaker is dh_agent:
            return rec_agent
        if last_speaker is rec_agent:
            return rsn_agent
        if last_speaker is rsn_agent:
            return None
        # first time user agent speak
        return user_agent
    return select_next


def main():

    load_dotenv()
    assert os.getenv("QWEN_API_KEY"), "Missing QWEN_API_KEY in .env"

    # question json file path
    # TODO: need this format of question list
    json_path = "temp_user_based_question_list.json"

    all_questions_dict = load_all_questions(json_path)

    all_users_list = load_all_users(json_path)

    # TODO: selected_users = all_users_list, select first 2 users for now
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
                "base_url": "https://openrouter.ai/api/v1"
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
                "You are a dialogue history agent. "
                f"You MUST call the tool 'rag_search_for_dialogue_history_agent' once to search {user} preference. "
                "Using the tool's result, then give label(s) to user"
            ),
        )

        recommend_agent = AssistantAgent(
            name=f"Recommender_{user}_agent",
            llm_config=llm_config,
            system_message=(
                "You are a shopping recommender. "
                "When the user asks for recommendations, you MUST call the tool 'rag_search_for_recommend_agent' once. "
                "Using the tool's result then give user recommendation."
            ),
        )

        reason_agent = AssistantAgent(
            name=f"Reasoner_{user}_agent",
            llm_config=llm_config,
            system_message=(
                "You only provide explanation of the rationale for the previous user label and recommendations. "
                "You MUST call the tool 'rag_search_for_reason_agent' once. "
                "Using the tool's result, then give reason(s) to user"
            ),
        )

        user_proxy_agent = UserProxyAgent(
            name=f"User_{user}_proxy_agent",
            human_input_mode="NEVER",
            code_execution_config=False,
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

        user_proxy_agent.register_reply(
            [autogen.Agent, None],
            reply_func=make_save_hook(user_proxy_agent.name),
            config={},
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
            name="rag_search_for_recommend_agent",
            description=(
                "Search the vector database for product, item, or knowledge information relevant to the user's query. "
                "Use it to generate personalized shopping recommendations."
            ),
        )(rag_search_for_recommend_agent)

        recommend_agent.register_for_execution(
            name="rag_search_for_recommend_agent"
        )(rag_search_for_recommend_agent)


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

            user_proxy_agent.initiate_chat(manager, message=q)

            new_msgs = groupchat.messages[:]
            user_history.extend(new_msgs)

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