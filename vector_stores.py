from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

import tempfile

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Sample documents
SAMPLE_DOCS = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "langchain_docs", "topic": "overview"},
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        metadata={"source": "langgraph_docs", "topic": "overview"},
    ),
    Document(
        page_content="Vector stores are databases optimized for storing and searching embeddings.",
        metadata={"source": "vector_guide", "topic": "database"},
    ),
    Document(
        page_content="RAG combines retrieval with generation for more accurate LLM responses.",
        metadata={"source": "rag_guide", "topic": "architecture"},
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors for semantic similarity.",
        metadata={"source": "embeddings_guide", "topic": "fundamentals"},
    ),
    Document(
        page_content="Chroma is an open-source embedding database for AI applications.",
        metadata={"source": "chroma_docs", "topic": "database"},
    ),
    Document(
        page_content="FAISS is a library for efficient similarity search developed by Facebook.",
        metadata={"source": "faiss_docs", "topic": "database"},
    ),
    Document(
        page_content="Pinecone is a managed vector database service for production workloads.",
        metadata={"source": "pinecone_docs", "topic": "database"},
    ),
]

# https://reference.langchain.com/python/langchain-chroma/vectorstores/Chroma/from_documents
"""If you don’t specify a collection name, LangChain automatically creates a Chroma collection named "langchain",
 and stores your documents there — nothing is fetched from anywhere else."""


def chroma_basics():
    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS, embedding=embedding_model, persist_directory=tmpdir
        )
        print(
            f"Vector store created with Collection name - {vectorstore._collection.name} - {vectorstore._collection.count()} Documents persisted"
        )

        query = "What is LangChain?"
        results_basic = vectorstore.similarity_search(query=query, k=2)

        print(f"Top 2 results for query '{query}': ")
        for i, doc in enumerate(results_basic):
            print(
                f"Result {i+1}: {doc.page_content}  , (Source: {doc.metadata['source']} , (Similarity_search - {doc.metadata}))"
            )
        # To find out metadata details various ways
        # print(doc.metadata.keys())
        #print(doc.metadata)
        # print(results_basic[1].metadata)



if __name__ == "__main__":
    chroma_basics()