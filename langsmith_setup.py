"""
LangSmith Setup and Observablity
Production monitoring for LangChain/LangGraph
"""

import os
# from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from langsmith.run_trees import RunTree
from dotenv import load_dotenv

load_dotenv()

"""
Trace - Agent Flow, Input/Outputs, Tool Calls, Decision made
Metrics - Token count, Latency per node, Cost per run, Error rates
Evals - Correctness, Relevance, Human Feedback, Regression detection
"""

os.environ["LANGSMITH_TRACING"]="true"
os.environ["LANGSMITH_PROJECT"]="multi-agent-research-system"

@traceable(name="named_run_demos", tags=["production","summarization"])
def demo_basic_tracing():
    """Basic LangSmith tracing."""

    llm = ChatGroq(model="llama-3.3-70b-versatile",
                   temperature=0,
                   api_key=os.getenv("GROQ_API_KEY"))
    
    prompt = ChatPromptTemplate.from_template(
        "Explain {topic} in one sentence."
    )

    chain = prompt | llm | StrOutputParser()

    print("Basic Tracing Demo: \n")
    print("Running Chain with LangSmith tracing enabled...")

    result = chain.invoke({"topic":"machine learning"})

    print(f"Result: {result}")
    print("\n Check LangSmith dashboard for trace details.")


def demo_trace_with_metadata(user_id: str, request_type: str):
    """Add metadata to traces for filtering."""

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=os.getenv("GROQ_API_KEY"))

    #Metadata is automatically captured
    result = llm.invoke(f"Hello from user {user_id}")

    print(f"Metadata Result: {result}")
    return result.content

if __name__ == "__main__":
    demo_basic_tracing()
    demo_trace_with_metadata(user_id="user_123", request_type="greeting")


# Check lang dashboard. Go tracings