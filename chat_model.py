import os
from dotenv import load_dotenv
from uuid import uuid4

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_neo4j import Neo4jChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_neo4j import Neo4jGraph

load_dotenv()

SESSION_ID = str(uuid4())
print(f"Session ID: {SESSION_ID}")

memory = ChatMessageHistory()


def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)


graph = Neo4jGraph(
    url=os.getenv("NEO4J_URL"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)


chat_llm = ChatGoogleGenerativeAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-2.0-flash-lite",
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=0,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a surfer dude, having a conversation about the surf conditions on the beach. Respond using surfer slang.",
        ),
        ("system", "{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

current_weather = """
    {
        "surf": [
            {"beach": "Fistral", "conditions": "6ft waves and offshore winds"},
            {"beach": "Polzeath", "conditions": "Flat and calm"},
            {"beach": "Watergate Bay", "conditions": "3ft waves and onshore winds"}
        ]
    }"""

chat_chain = prompt | chat_llm | StrOutputParser()

chat_with_message_history = RunnableWithMessageHistory(
    chat_chain,
    get_memory,
    input_messages_key="question",
    history_messages_key="chat_history",
)

while (question := input("> ")) != "exit":
    response = chat_with_message_history.invoke(
        {
            "context": current_weather,
            "question": question,
        },
        config={"configurable": {"session_id": SESSION_ID}},
    )

    print(response)
