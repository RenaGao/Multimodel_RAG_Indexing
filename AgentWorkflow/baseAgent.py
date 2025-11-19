from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import os
from helpFunctions import load_all_questions, load_all_users
from datetime import datetime
from pathlib import Path
import json

def make_simple_flow(user_agent, dh_agent, recommender, reasoner, reply_agent):
    # must keep groupchat here as it required by autogen framework
    def select_next(last_speaker, groupchat):
        """User -> DialogueHistory -> Recommender -> Reasoner -> Reply -> End"""
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
        # first time user agent
        return user_agent
    return select_next

def main():

    load_dotenv()
    assert os.getenv("QWEN_API_KEY"), "Missing QWEN_API_KEY in .env"

    # question json file path
    # TODO: need this format of question list
    json_path = "../questions/Amazon_Fashion_questions_20251110_214618.json"

    all_questions_dict = load_all_questions(json_path)

    all_users_list = load_all_users(json_path)

    selected_users = all_users_list[:1]
    
    # how many questions can run for one user
    per_user_limit = None # run all questions from the user

    # store output for json format
    results = {}

    llm_config = {
            "config_list": [{
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

        # separate agent based on category
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

        # output separated by category
        results[user] = []

        category_history = []

        # Track user queries and reply responses for evaluation
        eval_dialogue_history = []
        user_queries = []

        for i, q in enumerate(questions, start=1):

            groupchat = GroupChat(
                agents=[user_proxy_agent, dialogue_history_agent, recommend_agent, reason_agent, reply_agent],
                messages=list(category_history),
                max_round=10, # increased from 6 to accommodate more agents
                speaker_selection_method=make_simple_flow(user_proxy_agent, dialogue_history_agent, recommend_agent, reason_agent, reply_agent)
            )

            manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

            # Track length before chat to extract only new messages
            old_history_len = len(category_history)

            user_proxy_agent.initiate_chat(manager, message=q)

            # Extract ONLY new messages from this round
            new_msgs = groupchat.messages[old_history_len:]
            print(f"[DEBUG Round {i}] old_history_len={old_history_len}, new_msgs count={len(new_msgs)}")

            # Extract reply agent response for evaluation
            reply_content = None
            for msg in new_msgs:
                if isinstance(msg, dict):
                    msg_name = msg.get("name", "")
                    if "Reply" in msg_name:
                        reply_content = msg.get("content", "")
                        break

            print(f"[DEBUG Round {i}] reply_content found: {reply_content is not None}, length: {len(reply_content) if reply_content else 0}")

            # Add to eval data
            if reply_content:
                eval_dialogue_history.append({"role": "user", "content": q})
                eval_dialogue_history.append({"role": "assistant", "content": reply_content})
                user_queries.append(q)
                print(f"[DEBUG Round {i}] Added to eval. Total queries: {len(user_queries)}")
            else:
                print(f"[DEBUG Round {i}] WARNING: reply_content is None, NOT added to eval!")

            category_history.extend(new_msgs)

        results[user] = category_history

        # Generate evaluation export
        if user_queries:
            num_queries = len(user_queries)
            query_summary = ", ".join([f"({i+1}) {q}" for i, q in enumerate(user_queries)])
            user_goal = f"Multi-round shopping session with {num_queries} queries: {query_summary}"

            eval_entry = {
                "dialogue_id": user,
                "user_goal": user_goal,
                "dialogue_history": eval_dialogue_history
            }

            # Save eval file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            eval_dir = Path("./")
            eval_dir.mkdir(parents=True, exist_ok=True)
            eval_path = eval_dir / f"baseline_eval_dialogues_{user}_{timestamp}.json"
            with open(eval_path, "w", encoding="utf-8") as f:
                json.dump([eval_entry], f, ensure_ascii=False, indent=2)

            print(f"[EXPORT] Evaluation data saved to: {eval_path}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("./")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = f"shopping_agent_output_{ts}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved agent outputs.")

if __name__ == "__main__":
    main()