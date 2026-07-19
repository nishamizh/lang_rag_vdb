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


def similarity_search():
    docs = [
        "Python is a programming language",
        "Javascript is used for web development",
        "Machine learning enables AI applications",
        "Deep learning uses neural networks",
        "Cats are popular pets",
    ]

    query = "What programming languages exist?"

    doc_vector = embedding_model.embed_documents(docs)
    query_vector = embedding_model.embed_query(query)

    def cosine_similarity(vec1,vec2):
        return np.dot(vec1,vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    similarities = [cosine_similarity(query_vector, doc_vector) for doc_vector in doc_vector]

    # rank doocuments by similarity
    ranked_docs = sorted(zip(docs,similarities), key=lambda x:x[1], reverse = True)

    print(f"Query: {query}\n")
    print("Ranked by similarity: ")
    for doc,score in ranked_docs:
        print(f" {score:.4f}: {doc}")

# Caching ---
def embedding_caching():
    from langchain_classic.embeddings.cache import CacheBackedEmbeddings

    from langchain_classic.storage import LocalFileStore
    import tempfile
    import time

    with tempfile.TemporaryDirectory() as tempdir:
        store = LocalFileStore(root_path=tempdir)

        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=embedding_model,
            document_embedding_cache=store,
            namespace="exercise",
        )
        """
        this code builds a tiny, local, on‑disk embedding cache, and LangChain automatically 
        checks that cache before calling the embedding model again. That’s why the second call is instant.
        """
        text = "What is Reinforcement Learning?"

        # First call - hits API
        print("First call (API):")
        start = time.time()
        vectors1 = cached_embeddings.embed_documents([text])
        print("First call time:", time.time() - start)
        print(f"  Embedded {len(vectors1)} documents")
        

        # Second call - from cache
        print("\nSecond call (Cache):")
        start = time.time()
        vectors2 = cached_embeddings.embed_documents([text])
        print("Second call time:", time.time() - start)
        print(f"  Embedded {len(vectors2)} documents")

        # Verify same results
        print(f"\nSame vectors: {np.allclose(vectors1[0], vectors2[0])}")

if __name__ == "__main__":
    # print("=== Single Text Embedding ===")
    # basic_embeddings()
    # print("=== Batch Embedding ===")
    # batch_embeddings()
    # print("=== Similarity Search ===")
    # similarity_search()
    print("=== Caching ===")
    embedding_caching()
