import os
import csv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def check_movie_schema_proper():
    """Use Neo4j's built-in schema inspection procedures"""
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )
    driver.verify_connectivity()

    # Method 1: Get node type properties (Neo4j 5.x)
    try:
        schema_query = """
        CALL db.schema.nodeTypeProperties()
        YIELD nodeType, nodeLabels, propertyName, propertyTypes, mandatory
        WHERE 'Movie' IN nodeLabels
        RETURN propertyName, propertyTypes
        ORDER BY propertyName
        """
        result = driver.execute_query(schema_query)

        print("Movie node properties from db.schema.nodeTypeProperties():")
        for record in result[0]:
            print(f"  - {record['propertyName']}: {record['propertyTypes']}")
    except:
        print("db.schema.nodeTypeProperties() not available (requires Neo4j 5.x)")

    # Method 2: Using APOC (if installed)
    try:
        apoc_query = """
        CALL apoc.meta.schema()
        YIELD value
        RETURN value
        """
        result = driver.execute_query(apoc_query)
        schema = result[0][0]["value"]

        if "Movie" in schema:
            print("\nMovie schema from APOC:")
            movie_schema = schema["Movie"]
            if "properties" in movie_schema:
                for prop, details in movie_schema["properties"].items():
                    print(f"  - {prop}: {details}")
    except:
        print("\nAPOC not available or apoc.meta.schema() failed")

    # Method 3: Get all property keys (simple approach)
    try:
        keys_query = """
        CALL db.propertyKeys()
        YIELD propertyKey
        RETURN propertyKey
        ORDER BY propertyKey
        """
        result = driver.execute_query(keys_query)
        all_keys = [record["propertyKey"] for record in result[0]]
        print(f"\nAll property keys in database: {all_keys}")
    except:
        pass

    # Method 4: Sample actual Movie nodes to see what properties they have
    sample_query = """
    MATCH (m:Movie)
    RETURN m
    LIMIT 5
    """
    result = driver.execute_query(sample_query)

    if result[0]:
        print("\nActual properties found in Movie nodes:")
        all_props = set()
        for record in result[0]:
            movie = dict(record["m"])
            all_props.update(movie.keys())
        print(f"  Properties: {sorted(all_props)}")

        # Show a sample movie
        sample_movie = dict(result[0][0]["m"])
        print("\nSample Movie:")
        for key, value in sample_movie.items():
            display_value = (
                str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
            )
            print(f"  - {key}: {display_value}")

    driver.close()


def get_movie_plots_smart(limit=None):
    """Get movies with smart property detection"""
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )
    driver.verify_connectivity()

    # First get actual properties from a sample
    sample_query = """
    MATCH (m:Movie)
    RETURN m
    LIMIT 10
    """
    result = driver.execute_query(sample_query)

    if not result[0]:
        print("No Movie nodes found in database!")
        driver.close()
        return []

    # Collect all properties from sample
    all_props = set()
    for record in result[0]:
        movie = dict(record["m"])
        all_props.update(movie.keys())

    print(f"Found properties in Movie nodes: {sorted(all_props)}")

    # Common description property names
    description_properties = [
        "plot",
        "description",
        "overview",
        "summary",
        "synopsis",
        "storyline",
        "tagline",
        "Plot",
        "Description",
    ]
    id_properties = ["movieId", "id", "ID", "tmdbId", "imdbId", "movie_id"]

    # Find which properties exist
    desc_prop = next((p for p in description_properties if p in all_props), None)
    id_prop = next((p for p in id_properties if p in all_props), None)

    # Check if title exists
    title_prop = (
        "title" if "title" in all_props else "name" if "name" in all_props else None
    )

    if not desc_prop:
        print(f"\nNo description property found!")
        print(f"Available properties: {sorted(all_props)}")
        print("\nYou may need to:")
        print("1. Check if your movies have plot/description data")
        print("2. Use a different property")
        print("3. Import plot data into your database")
        driver.close()
        return []

    print(f"\nUsing properties:")
    print(f"  - ID: {id_prop or 'id(m) [Neo4j internal ID]'}")
    print(f"  - Title: {title_prop or '[No title property found]'}")
    print(f"  - Description: {desc_prop}")

    # Build query
    if id_prop and title_prop:
        query = f"""
        MATCH (m:Movie) 
        WHERE m.{desc_prop} IS NOT NULL AND trim(m.{desc_prop}) <> ''
        RETURN m.{id_prop} AS movieId, m.{title_prop} AS title, m.{desc_prop} AS plot
        """
    elif title_prop:
        query = f"""
        MATCH (m:Movie) 
        WHERE m.{desc_prop} IS NOT NULL AND trim(m.{desc_prop}) <> ''
        RETURN id(m) AS movieId, m.{title_prop} AS title, m.{desc_prop} AS plot
        """
    else:
        query = f"""
        MATCH (m:Movie) 
        WHERE m.{desc_prop} IS NOT NULL AND trim(m.{desc_prop}) <> ''
        RETURN id(m) AS movieId, id(m) AS title, m.{desc_prop} AS plot
        """

    if limit is not None:
        query += f" LIMIT {limit}"

    movies, _, _ = driver.execute_query(query)

    print(f"\nFound {len(movies)} movies with non-empty {desc_prop}")

    driver.close()
    return movies


def generate_embeddings(file_name, limit=None):
    # Fix path for cross-platform compatibility
    file_name = os.path.join(".", "data", "movie-plot-embeddings.csv")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    movies = get_movie_plots_smart(limit=limit)

    if len(movies) == 0:
        print("\nNo movies with descriptions found. Cannot generate embeddings.")
        return

    csvfile_out = open(file_name, "w", encoding="utf8", newline="")
    fieldnames = ["movieId", "embedding"]
    output_plot = csv.DictWriter(csvfile_out, fieldnames=fieldnames)
    output_plot.writeheader()

    # Initialize Google Generative AI Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=GOOGLE_API_KEY
    )

    for i, movie in enumerate(movies):
        print(f"Processing {i + 1}/{len(movies)}: {movie['title']}")
        plot = f"{movie['title']}: {movie['plot']}"

        # Generate embedding using Google's model
        embedding = embeddings.embed_query(plot)

        output_plot.writerow({"movieId": movie["movieId"], "embedding": embedding})

    csvfile_out.close()
    print(f"\nEmbeddings saved to {file_name}")


if __name__ == "__main__":
    # First check the schema properly
    print("Checking database schema using Neo4j procedures...")
    check_movie_schema_proper()

    print("\n" + "=" * 50 + "\n")

    # Then generate embeddings
    generate_embeddings(
        "movie-plot-embeddings.csv", limit=5
    )  # Start with just 5 for testing
