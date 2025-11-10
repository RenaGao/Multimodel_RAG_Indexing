from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import os
from helpFunctions import load_all_users_and_questions
from datetime import datetime
from pathlib import Path
import json

def make_simple_flow(user_agent, recommender, reasoner):
    # must keep groupchat here as it required by autogen framework
    def select_next(last_speaker, groupchat):
        """User -> Recommender -> Reasoner -> End"""
        if last_speaker is user_agent:
            return recommender
        if last_speaker is recommender:
            return reasoner
        if last_speaker is reasoner:
            return None
        # first time user agent
        return user_agent
    return select_next

def main():

    load_dotenv()
    assert os.getenv("QWEN_API_KEY"), "Missing QWEN_API_KEY in .env"

    # question json file path
    # TODO: need this format of question list
    json_path = "../questions/All_Beauty_questions_20251110_212053.json"

    all_questions_dict, all_users_list = load_all_users_and_questions(json_path)

    selected_users = all_users_list[:]
    
    # how many questions can run for one user
    per_user_limit = None # run all questions from the user

    # store output for json format
    results = {}

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

        # separate agent based on category
        recommend_agent = AssistantAgent(
            name=f"Recommender_{user}_agent",
            llm_config=llm_config,
            system_message=(
                "You are a shopping recommender."
            ),
        )
        reason_agent = AssistantAgent(
            name=f"Reasoner_{user}_agent",
            llm_config=llm_config,
            system_message=(
                f"You only provide explanation of the rationale for the previous recommendations from Recommender_{user}_agent."
            ),
        )
        user_proxy_agent = UserProxyAgent(
            name=f"User_{user}",
            human_input_mode="NEVER",
            code_execution_config=False,
        )

        # output separated by category
        results[user] = []

        category_history = []

        for i, q in enumerate(questions, start=1):

            groupchat = GroupChat(
                agents=[user_proxy_agent, recommend_agent, reason_agent],
                messages=list(category_history),
                max_round=6, # no need to restrict as we already defined the workflow, still put a lao liu here
                speaker_selection_method=make_simple_flow(user_proxy_agent, recommend_agent, reason_agent)
            )

            manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

            user_proxy_agent.initiate_chat(manager, message=q)

            new_msgs = groupchat.messages[:]
            category_history.extend(new_msgs)

        results[user] = category_history
            
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("./")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = f"shopping_agent_output_{ts}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved agent outputs.")

if __name__ == "__main__":
    main()