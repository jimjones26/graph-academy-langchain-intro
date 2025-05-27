import os
from dotenv import load_dotenv

from langchain_neo4j import Neo4jGraph

load_dotenv()

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URL"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

result = graph.query("""
MATCH (d:DocumentationCorpus{name: 'Google Agent Development Kit Documentation'}) 
RETURN d.sourceUrl
""")
print(result)

print(graph.schema)
