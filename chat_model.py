import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

# from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

load_dotenv()

chat_llm = ChatGoogleGenerativeAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-2.0-flash-lite",
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=0,
)

# instructions = SystemMessage(
#     content="""
# You are a surfer dude, having a conversation about the surf conditions on the beach. Respond using surfer slang.
# """
# )

# question = HumanMessage(content="What are the surf conditions like today?")

# response = chat_llm.invoke([instructions, question])

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a surfer dude, having a conversation about the surf conditions on the beach. Respond using surfer slang.",
        ),
        ("human", "{question}"),
    ]
)

chat_chain = prompt | chat_llm | StrOutputParser()

response = chat_chain.invoke({"question": "What are the surf conditions like today?"})

print(response)
