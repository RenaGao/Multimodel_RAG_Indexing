from datetime import datetime
from pathlib import Path
from typing import Dict, Any

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
def rag_search_include_image_for_recommend_agent(
    user_query: str,
    dialogue_summary: str,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Perform RAG search and return products (with image URLs) for the recommender agent.

    Args:
        user_query (str): User's query (e.g., "moisturizer for dry skin").
        dialogue_summary (str): User preference summary from DialogueHistoryAgent.
        top_k (int): Number of retrieved items to return.

    Returns:
        dict: Structured data containing query, context, items (with images), and metadata.
    """

    # example output
    retrieved_items = [
        {
            "id": "cerave-123",
            "title": "CeraVe Moisturizing Cream 340g",
            "summary": (
                "Fragrance-free moisturizing cream with ceramides and hyaluronic acid. "
                "Ideal for very dry or eczema-prone skin."
            ),
            "price": {"value": 18.99, "currency": "AUD"},
            "images": [
                {"url": "https://cdn.example.com/img/cerave-1.jpg", "alt": "Front jar shot"},
                {"url": "https://cdn.example.com/img/cerave-2.jpg", "alt": "Texture close-up"}
            ],
            "tags": ["dry-skin", "hydration", "fragrance-free"]
        },
        {
            "id": "laroche-001",
            "title": "La Roche-Posay Lipikar Baume AP+M 400ml",
            "summary": (
                "Soothing repairing balm for atopic, itchy, and very dry skin. "
                "Fragrance-free and suitable for babies and adults."
            ),
            "price": {"value": 28.50, "currency": "AUD"},
            "images": [
                {"url": "https://cdn.example.com/img/laroche-1.jpg", "alt": "Bottle packaging"}
            ],
            "tags": ["dry-skin", "sensitive-skin", "repairing"]
        }
    ][:top_k]

    result = {
        "query": user_query,
        "dialogue_context": dialogue_summary,
        "items": retrieved_items,
    }

    return result

# TODO:
def rag_search_for_reason_agent() -> str:
    return """"""
