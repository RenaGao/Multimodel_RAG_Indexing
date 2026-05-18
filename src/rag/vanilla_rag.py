from argparse import ArgumentParser
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

def build_product_context(item_data):

    meta_data = item_data["meta"]
    context = ""
    context += f"Product Title: {meta_data['title']}.\n" if 'title' in meta_data else ""
    context += f"Category: {meta_data['main_category']}.\n" if 'main_category' in meta_data else ""
    context += f"Description: {meta_data['description']}.\n" if 'description' in meta_data else ""
    context += f"Price: {meta_data['price']}.\n" if 'price' in meta_data else ""
    context += f"Store: {meta_data['store']}.\n" if 'store' in meta_data else ""
    context += f"Average Rating: {meta_data['average_rating']}.\n" if 'average_rating' in meta_data else ""
    context += f"Rating Number: {meta_data['rating_number']}.\n" if 'rating_number' in meta_data else ""
    features = meta_data.get('features', [])
    
    if len(features) > 0:
        context += "Features:\n"
        for feature in features:
            context += f"- {feature}\n"

    details = meta_data.get('details', {})
    for detail_key, detail_value in details.items():
        context += f"{detail_key}: {detail_value}.\n"

    return context



def index(all_item_data, save_path):

    if os.path.exists(save_path):
        print(f"Index already exists at {save_path}. Skipping indexing.")
        index = faiss.read_index(save_path)
        return index

    print("Indexing documents...")
    if not os.path.exists("index"):
        os.makedirs("index")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    embeddings = []

    for item_id in tqdm(all_item_data):
        item_data = all_item_data[item_id]
        item_context = build_product_context(item_data)

        emb = model.encode(item_context, normalize_embeddings=True)
        embeddings.append(emb)

    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  
    index.add(embeddings)

    print(f"Indexed {embeddings.shape[0]} products with dimension {dim}")

    faiss.write_index(index, save_path)

    print(f"Index saved to {save_path}")

    return index


def search(query, top_k=5, index_path="product_index.faiss"):

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    index = faiss.read_index(index_path)

    q_emb = model.encode(query, normalize_embeddings=True)
    D, I = index.search(np.array([q_emb], dtype="float32"), top_k)

    return D, I


def main(args):

    with open(f"AmazonReviews/Processed/{args.category}_top_100_item_data.json", "r") as f:
        all_item_data = json.load(f)

    index_path = f"index/{args.category}_product_index.faiss"

    faiss_index = index(all_item_data, index_path)

    query = "Can you help me find a good moisturizer for dry skin?"

    D, I = search(query, top_k=5, index_path=index_path)

    print("Query:", query)
    print()
    print("Search Results:")
    for rank, (dist, idx) in enumerate(zip(D[0], I[0])):
        item_id = list(all_item_data.keys())[idx]
        item_data = all_item_data[item_id]
        print(f"Rank {rank + 1}:")
        print(f"Item ID: {item_id}")
        print(f"Title: {item_data['meta'].get('title', 'N/A')}")
        print(f"Distance: {dist}")
        print()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--category", "-c", type=str, default="All_Beauty", help="Category of the documents to index")

    args = parser.parse_args()

    main(args)