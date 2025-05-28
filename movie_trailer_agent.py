import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import StrOutputParser
from langchain_neo4j import Neo4jChatMessageHistory, Neo4jGraph
from langchain_community.tools import YouTubeSearchTool
from uuid import uuid4

load_dotenv()

SESSION_ID = str(uuid4())
print(f"Session ID: {SESSION_ID}")

youtube = YouTubeSearchTool()

llm = ChatGoogleGenerativeAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-2.5-flash-preview-05-20",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URL"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie expert. You find movies from a genre or plot."),
        ("human", "{inut}"),
    ]
)

movie_chat = prompt | llm | StrOutputParser()


def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)


def call_trailer_search(input):
    input = input.replace(",", " ")
    return youtube.run(input)


tools = [
    Tool.from_function(
        name="Movie Chat",
        description="For when you need to chat about movies. The question will be a string. Return a string.",
        func=movie_chat.invoke,
    ),
    Tool.from_function(
        name="Movie Trailer Search",
        description="Use when needing to find a movie trailer. The question will include the word trailer. Return a link to a YouTube video.",
        func=call_trailer_search,
    ),
]

agent_prompt = hub.pull("hwchase17/react-chat")
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

while (q := input("> ")) != "exit":
    response = chat_agent.invoke(
        {"input": q},
        {"configurable": {"session_id": SESSION_ID}},
    )

    print(response["output"])
