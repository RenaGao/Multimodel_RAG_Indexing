from typing import List

def save_chat_history(chat_history: str) -> bool:
    # TODO: save into vector database
    return True


# TODO:
def rag_search_for_dialogue_history_agent(user_name:str) -> str:
    
    return """
    [
      {"item_id": "A1", "title": "CeraVe Moisturizing Cream", "price": "$15-20", "rating": 4.7},
      {"item_id": "A2", "title": "La Roche-Posay Lipikar Balm AP+", "price": "$20-30", "rating": 4.6}
    ]
    """


# TODO:
def rag_search_for_recommend_agent(user_query: str, user_labels: List[str], top_k: int) -> str:
    
    return """
    [
      {"item_id": "A1", "title": "CeraVe Moisturizing Cream", "price": "$15-20", "rating": 4.7},
      {"item_id": "A2", "title": "La Roche-Posay Lipikar Balm AP+", "price": "$20-30", "rating": 4.6}
    ]
    """

# TODO:
def rag_search_for_reason_agent() -> str:
    return """"""
