from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import os
from helpFunctions import load_all_questions, load_all_categories
from datetime import datetime
from pathlib import Path
import json
from ragFunction import rag_search


def main():

    load_dotenv()
    assert os.getenv("OPENAI_API_KEY"), "Missing OPENAI_API_KEY in .env"

    # question json file
    json_path = "../outputs/shopping_user_questions_trimmed.json"

    all_questions_dict = load_all_questions(json_path)

    all_categories_list = load_all_categories(json_path)

    # TODO: selected_categories = all_categories_list, select first 2 category for now
    selected_categories = all_categories_list[:2]
    
    # how many questions can run for one category
    per_category_limit = None # run all questions in the category

    # store output for json format
    results = {}

    llm_config = {"config_list": [{"model": "gpt-5-nano", "api_key": None}]}

    for category in selected_categories:
        questions = all_questions_dict[category]
        if per_category_limit:
            questions = questions[:per_category_limit]

        # separate agent based on category
        recommend_agent = AssistantAgent(
            name=f"Recommender_{category}",
            llm_config=llm_config,
            system_message=(
                "You are a shopping recommender. "
                "When the user asks for recommendations, you MUST call the tool 'ragSearch' and ONLY once. "
                "Using the tool's result then give user recommendation."
            ),
        )

        reason_agent = AssistantAgent(
            name=f"Reasoner_{category}",
            llm_config=llm_config,
            system_message=(
                f"You only provide explanation of the rationale for the previous recommendations from Recommender_{category}."
            ),
        )
        user = UserProxyAgent(
            name=f"User_{category}",
            human_input_mode="NEVER",
            code_execution_config=False,
        )


        recommend_agent.register_for_llm(name="ragSearch", description="A search tool to retrive data from a vector database.")(rag_search)
        user.register_for_execution(name="ragSearch")(rag_search)

        groupchat = GroupChat(
            agents=[user, recommend_agent, reason_agent],
            messages=[],
            max_round= 3,
            speaker_selection_method="round_robin"
        )

        manager = GroupChatManager(
            groupchat=groupchat,
            llm_config=llm_config,
        )

        # output separated by category
        results[category] = []

        category_history = []

        for i, q in enumerate(questions, start=1):

            groupchat = GroupChat(
                agents=[user, recommend_agent, reason_agent],
                messages=list(category_history),
                max_round=5,
                speaker_selection_method="round_robin",
            )

            manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

            user.initiate_chat(manager, message=q)

            new_msgs = groupchat.messages[:]
            category_history.extend(new_msgs)

        results[category] = category_history
            
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("./")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = f"shopping_agent_output_{ts}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved agent outputs.")

if __name__ == "__main__":
    main()