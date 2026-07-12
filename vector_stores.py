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
# Create a Chroma vectorstore from a list of documents.
"""If you don’t specify a collection name, LangChain automatically creates a Chroma collection named "langchain",
 and stores your documents there — nothing is fetched from anywhere else."""


def chroma_basics():
    with tempfile.TemporaryDirectory() as tmpdir:
        #create a new vector store from documents
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

def similarity_search_with_scores():
    with tempfile.TemporaryDirectory() as tempdir:
        #create a new vector store from documents
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS, embedding=embedding_model, persist_directory=tempdir
            )
        
        query = "Explain vector stores"

        results_with_scores = vectorstore.similarity_search_with_score(query=query, k=3)

        print(f"Top 3 results with scores for query '{query}':")

        for i, (doc,score) in enumerate(results_with_scores):
            final_score = 1/(1+score)  #convert distance to similarity
            print(f" {i+1}: {doc.page_content} (Score: {final_score:.4f}, Source: {doc.metadata['source']})")


def metadata_filtering():
    with tempfile.TemporaryDirectory() as tempdir:
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS, embedding=embedding_model, persist_directory=tempdir
        )

        query = "What databases are available"

        results = vectorstore.similarity_search(query=query, k=5)

        for i,doc in enumerate(results):
            print(f" {i+1} -  {doc.page_content} , (Source : {doc.metadata["source"]})")

        # Adding Metadata Filtering
        filter_criteria = {"topic": "database"}
        filtered_results = vectorstore.similarity_search(query=query, k=5, filter=filter_criteria)

        for i,doc in enumerate(filtered_results):
            print(f" {i+1} : {doc.page_content} , (Source: {doc.metadata["source"]})")


    




if __name__ == "__main__":
    #chroma_basics()
    #similarity_search_with_scores()
    metadata_filtering()



#
# Vector Similarity Search — Step‑by‑Step
#   1. Load or create your documents
#       Each document has:
#           page_content
#           metadata (optional but useful)
#   2. Embed the documents
#       Use an embedding model to convert text → vectors.
#   3. Store embeddings in a vector database
#       Examples:Chroma, Pinecone, FAISS
# 4. Run a similarity search
#      similarity_search_with_score(query, k)
#           This returns:
#               top‑k documents
#               distance scores (not similarity scores)
#   5. Interpret the scores
#       Distance score meaning:
#            Lower = more similar
#            Higher = less similar
# Example:
# 0.66 → very relevant
# 1.34 → least relevant
#   6. Convert distance → similarity (optional)
#       If you want similarity instead of distance:
#       similarity=1/(1+distance)

#  Metadata Filtering — Step‑by‑Step
#   7. Add metadata to each document
#   Example:
#   {"topic": "database"}

#   8. Define a filter
#   Example:
#       filter_criteria = {"topic": "database"}

#   9. Run similarity search WITH filter
#       similarity_search(query, k, filter=filter_criteria)

#   10. Vector store first filters documents
#       It keeps only documents whose metadata matches the filter.

#   11. Then it performs similarity search on the filtered subset
#       This gives:
#           fewer documents
#           more relevant results
#           metadata‑aware retrieval

#   12. Compare results with and without filter
#    Without filter → all relevant docs
#   With filter → only docs matching metadata