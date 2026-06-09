import os
from dotenv import load_dotenv
load_dotenv()
from importlib.metadata import version


from langchain_core import __version__ as core_version
#from langgraph import __version__ as lg_version   --> this line is deprecated use importlib.metadata and below line to find version
lg_version = version("langgraph")
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq


print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")
print("Groq version:", version("groq"))

def main():
   
    print("Python sees:", os.getenv("GROQ_API_KEY"))
    print("From getenv:", os.getenv("ANTHROPIC_API_KEY"))
    print("From environ:", os.environ.get("GROQ_API_KEY"))
    print("Is .env loaded:", ".env" in os.listdir())

    # test groq - llama
    llma_groq = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
    ) 
    response = llma_groq.invoke("Say 'setup complete!' in one word") 
    print(f"Response from ChatOpenAI: {response}")

    # test anthropic
    print(f"Response from Anthropic:")
    llm_anthropic = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0, api_key=os.getenv("ANTHROPIC_API_KEY"))
    response_anthropic = llm_anthropic.invoke("Say 'setup complete!' in one word")
    print(f"Response from ChatAnthropic: {response_anthropic}")

    print("Set up complete!")


if __name__ == "__main__":
    main()
