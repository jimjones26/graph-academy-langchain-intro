import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from pydantic import SecretStr

load_dotenv()

embedding_provider = GoogleGenerativeAIEmbeddings(
    google_api_key=SecretStr(os.getenv("GOOGLE_API_KEY")),
    model="models/embedding-001",
)

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URL"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

movie_plot_vector = Neo4jVector(
    embedding_provider,
    graph=graph,
    index_name="moviePlots",
    embedding_node_property="plotEmbedding",
    text_node_property="plot",
)

result = movie_plot_vector.similarity_search(
    "A movie where aliens land and attack earth."
)
for doc in result:
    print(doc.metadata["title"], "-", doc.page_content)
