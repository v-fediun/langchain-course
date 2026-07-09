print("Hello, World!")

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate

from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.tools import tool
from langchain_core.messages import HumanMessage

@tool
def search(query: str) -> str:
    """Searches the web for the given query and returns the results.
    Args:
        query (str): The search query.
    Returns:
        str: The search results."""
    
    print(f"Searching for: {query}")
    return("THIS IS A MOCK RESULT FOR THE QUERY: " + query)


# llm = ChatOllama(model="gpt-oss:20b", temperature=0.2)
llm = ChatGoogleGenerativeAI( model="gemini-2.5-flash", temperature=0.2)

tools = [search]
agent = create_agent(model=llm, tools=tools)

result = agent.invoke({"messages": [HumanMessage(content="What is the capital of France?")]})
print(result)