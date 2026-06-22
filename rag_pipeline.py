from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_groq import ChatGroq

from langchain.chat_models import init_chat_model

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List

from dotenv import load_dotenv
import tempfile

load_dotenv()
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Sample knowledge base
KNOWLEDGE_BASE = """# LangChain Framework

LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.

## Core Components

1. **Models**: LangChain supports various LLM providers including OpenAI, Anthropic, and local models.

2. **Prompts**: Templates for structuring inputs to language models.

3. **Chains**: Sequences of calls to models and other components.

4. **Agents**: Systems that use LLMs to determine which actions to take.

5. **Memory**: Components for persisting state between chain/agent calls.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications. Key features:
- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.

## Getting Started

Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""

# Create a Knowledge Base
def create_kb():
    """Create a Vector Store from Knowledge Base."""
    #  KB - Transfers all raw knowledge(pdf,files,..etc) converting them into embeddings, and storing those embeddings inside a vector database like Chroma
    
    # 1. We are just defining how splitter should be, what size and how many overlap, not actually splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    #2. Document - Class for storing a piece of text and associated metadata.
    doc = Document(page_content=KNOWLEDGE_BASE,
                   metadata = {"source": "langchain_knowledge_base.md"})

    #3. Split the Documents into chunks using the splitter defined in 1.
    chunks  = splitter.split_documents([doc])

    #4. Create a vector store from the chunks. Chroma.from_documents, Creates a Chroma vectorstore from a list of documents.
    vector_store = Chroma.from_documents(
        documents= chunks,
        embedding=embedding_model,
        persist_directory=tempfile.mkdtemp(),
    )

    return vector_store


    #chat_models - Entrypoint to using chat models in LangChain.
    # init_chat_model - Initialize a chat model from any supported provider using a unified interface.

    #langchain_groq - ChatGroq --> No init_chat_model. 2.No wrappers. 3.No extra config.


def demo_basic_rag():
    # 1. Fetch the VectorStore
    vector_store = create_kb()

    # 2. From the Vector_store - fetch/retrieve the documents based on similarity and top-2 results
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs = {"k":2}
    )

    # 3. Define the LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    #---- Let's do RAG ----

    #4. RAG Prompt template

    # 4.a Define the prompt

    #Chat models (like ChatGroq, ChatOpenAI, Anthropic, etc.) do not accept plain text.
    #They expect structured chat messages: system, user, assistant, tool

    #ChatPromptTemplate builds these messages correctly. 
    # Without it, your RAG chain cannot pass {context} and {question} into the LLM in the correct chat format.

    #ChatPromptTemplate.from_messages - Create a chat prompt template from a variety of message formats.
    #ChatPromptTemplate.from_template -  Use it when you want one single user message.
    
    prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}

Question: {question}

Answer:

Make sure to answer in a concise manner,
and if you don't know the answer, just say "I don't know."
"""
    )


    # Format retrieved documents
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    # RAG chain
    # 1. pass the first object - question and context
    # 2. prompt , 3. llm, 4.stroutputparser - so we get the return item as String


    rag_chain = (
        { "context" : retriever | format_docs, "question":RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # TEST the RAG

    questions = [
        "What is LangChain?",
        "Who created LangChain ?",
        "What is LangGraph used for?"
    ]

    print("Basic RAG Demo: \n")
    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"Q : {q}")
        print(f"A : {answer}")



if __name__ == "__main__":
    demo_basic_rag()