from typing import List
from datetime import datetime
from pathlib import Path

# TODO: replace with real rag save
def save_chat_history(chat_history: str) -> bool:
    """Append chat history to a local text file with timestamp."""
    try:
        log_path = Path("chat_history_log.txt")

        with log_path.open("a", encoding="utf-8") as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{ts}] {chat_history}\n")
            f.write("-" * 100 + "\n")

        print(f"[SAVE] -> {log_path}")
        return True
    except Exception as e:
        print(f"[SAVE ERROR] {e}")
        return False

# TODO:
def rag_search_for_dialogue_history_agent(user_name:str) -> str:
    
    return "function found and executed."

# TODO:
def rag_search_for_recommend_agent(user_query: str, dialogue_summary: str, top_k: int) -> str:
    
    return """
    [
      {"item_id": "A1", "title": "CeraVe Moisturizing Cream", "price": "$15-20", "rating": 4.7},
      {"item_id": "A2", "title": "La Roche-Posay Lipikar Balm AP+", "price": "$20-30", "rating": 4.6}
    ]
    """

# TODO:
def rag_search_for_reason_agent() -> str:
    return """"""
