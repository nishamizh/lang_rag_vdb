# Reference - https://copilot.microsoft.com/shares/jva2KypixBzVB4CSbUmJ1 - hybridSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# pip install langchain langchain-openai langchain-chroma rank_bm25 
# uv add rank_bm25x   
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# uv add bm25, uv add langchain-text-splitters

from dotenv import load_dotenv

load_dotenv()

documents = [

    Document(
        page_content='Product SKU-7742X is our flagship router. It supports'
        'gigabit speeds and advanced QoS features',
        metadata = {'type' : 'product'}
    ),

    Document(
        page_content='for network connectivity issues, first check the'
        'ethernet cable and router status lights.',
        metadata = {'type' : 'troubleshooting'}
    ),

    Document(
        page_content='Error code E_CONN_REFUSED indicates the server    '
        'rejected the connection. Check firewall settings.',
        metadata = {'type' : 'error'}
    ),

    Document(
        page_content='The authentication process requires valid credentials. '
        'Use OAuth2 for secure API access.',
        metadata = {'type' : 'auth'}
    ),

    Document(
        page_content='Router configuration guide: Access the admin panel'
        'at 192.168.1.1 to modify settings',
        metadata = {'type' : 'config'}
    ),

    Document(
        page_content='WCAG 2.1 compliance requires all images to have '
        'alt text and sufficient color contrast.',
        metadata = {'type' : 'compliance'}
    ),
]

print(f"Laoded {len(documents)} documents")

# 1. Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

vector_store = Chroma.from_documents(
    documents,
    embeddings,
    collection_name="hybrid_test"
)

#2. Create vector retriever
vector_retriever = vector_store.as_retriever(
    search_kwargs = {'k':3} # returns top 3
)

print('Vector retriever ready')

#3. BM25 retriever - BM25 works on the raw text
bm25_retriever = BM25Retriever.from_documents(
    documents,
    k=3 # returns top 3
)

print('BM25 retriever ready')

#ensemble_retriever = EnsembleRetriever(
#    retrievers = [bm25_retriever, vector_retriever],
#    weights=[0.5,0.5] # Equal weight to both
#)

# Hybrid retrieval using REciproal Rank Fusion (RRF)
# This is what EnsembleRetriever works internally
def reciprocal_rank_fusion(query, retrievers, weights, k=3, rrf_k=60):
    """
    Combine multiple retrievers using weighted Reciprocal Rank Fusion
    results_lists: List[List[Document]]
    weights: optional list of floats, same length as results_lists
    """
    doc_scores = {}

    for retriever, weight in zip(retrievers, weights):
        results = retriever.invoke(query)

        for rank, doc in enumerate(results):
            key = doc.page_content
            rrf_score = 1.0 / (rrf_k + rank)   # FIXED
            weighted = weight * rrf_score

            if key in doc_scores:
                doc_scores[key] = (doc_scores[key][0] + weighted, doc)
            else:
                doc_scores[key] = (weighted, doc)

    sorted_docs = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in sorted_docs[:k]]



    #if weights is None:
    #    weights = [1.0] * len(results_lists)

    #scores = defaultdict(float)

    #for retr_idx, docs in enumerate(results_lists):
    #    w = weights[retr_idx]
    #    for rank, doc in enumerate(docs):
    #        # RRF score: w / (k + rank)
    #        scores[id(doc)] += w / (k + rank)

    # deduplicate by id, keep highest scoring instance
    #id_to_doc = {}
    #for docs in results_lists:
    #    for d in docs:
    #        id_to_doc[id(d)] = d

    #ranked = sorted(id_to_doc.values(), key=lambda d: scores[id(d)], reverse=True)
    #return ranked

def hybrid_retriever(query):
    return reciprocal_rank_fusion(
        query,
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
        k=3
    )

print('Hybrid retriever ready')


def test_query(query, name, retriever):
    '''Test a query and show results'''
    results = retriever.invoke(query)
    print(f'\\n{name} - Query: \" {query}\"')
    for i, doc in enumerate(results[:3]):
        preview = doc.page_content[:80] + '...'
        print(f'  {i+1}. {preview}')
    return results

test_queries = [
    'SKU-7742X specifications',     # Exact product code
    'E_CONN_REFUSED error',         # Error code
    'How do I authenticate?',       # Semantic question
    'WCAG compliance',              # Acronym
    'router configuration',         # General semantic
]

if __name__ == "__main__":
    print("\nRunning hybrid search tests...\n")

    for q in test_queries:
        print(f"\n==============================")
        print(f"Query: \"{q}\"")
        print(f"==============================")
        # Vector
        vector_results = vector_retriever.invoke(q)
        print(f"\nVector - Query: \"{q}\"")
        for i, doc in enumerate(vector_results):
            preview = doc.page_content[:80] + "..."
            print(f"  {i+1}. {preview}")

        # BM25
        bm25_results = bm25_retriever.invoke(q)
        print(f"\nBM25 - Query: \"{q}\"")
        for i, doc in enumerate(bm25_results):
            preview = doc.page_content[:80] + "..."
            print(f"  {i+1}. {preview}")

        # Hybrid          
        hybrid_results = hybrid_retriever(q)
        print(f"\nHybrid - Query: \"{q}\"")
        for i, doc in enumerate(hybrid_results):
            preview = doc.page_content[:80] + "..."
            print(f"  {i+1}. {preview}")