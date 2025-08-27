from dotenv import load_dotenv

load_dotenv()

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
# from langchain_openai import ChatOpenAI
# from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

tools = [TavilySearch()]
# llm = ChatOpenAI(model="gpt-4")
# llm = ChatOllama(model="gemma3:270m")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
react_promt = hub.pull("hwchase17/react")
agent = create_react_agent(
    tools=tools,
    llm=llm,
    prompt=react_promt
)
agent_excuter=AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
chain=agent_excuter

def main():
    result = chain.invoke(
        input={
            "input": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details",
        }
    )
    print(result)


if __name__ == "__main__":
    main()
