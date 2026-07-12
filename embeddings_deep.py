from langchain_huggingface import HuggingFaceEmbeddings

import numpy as np
from dotenv import load_dotenv

load_dotenv()

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def basic_embeddings():
    # Single text
    text = "What is Machine Learning?"
    single_embedding = embedding_model.embed_query(text)
    print(f"Vector dimensions: {len(single_embedding)}")
    print(f"First 5 values: {single_embedding[:5]}")
    print(f"Vector norm: {np.linalg.norm(single_embedding):.4f}")
    
    # embedding_model.embed_query - Convert one user query into an embedding for retrieval.


def batch_embeddings():
    
    text = [
        "What is Machine Learning ?",
        "Explain the concept of overfitting in ML",
        "How does a neural network work ?",
    ]

    batch_embedding = embedding_model.embed_documents(text)
    for i, emb in enumerate(batch_embedding):
        print(f"Text {i+1} - Vector dimensions: {len(emb)}")
        print(f"Text {i+1} - First 5 values: {emb[:5]}")
        print(f"Text {i+1} - Vector norm: {np.linalg.norm(emb):.4f}")

    
    # embedding_model.embed_documents - Convert one user query into an embedding for retrieval.
    # Processes a single string, Optimized for fast inference, Often uses query‑optimized pooling, Used every time the user asks a question


if __name__ == "__main__":
    # print("=== Single Text Embedding ===")
    # basic_embeddings()
    print("=== Batch Embedding ===")
    batch_embeddings()