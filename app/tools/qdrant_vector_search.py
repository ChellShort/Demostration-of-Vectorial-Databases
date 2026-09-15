from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from langchain.tools import tool
import json

client = QdrantClient(
    host="localhost",
    port=6333
)
model = SentenceTransformer("all-MiniLM-L6-v2")

@tool 
def vector_search(
    query: str,
    limit: int = 5,
):
    """
    Search documents using semantic similarity.
    """

    query_vector = model.encode(query)

    results = client.query_points(
        collection_name="prueba",
        query=query_vector.tolist(),
        limit=limit,
    )

    formatted_results = []
    for result in results.points:
        formatted_results.append(
            {
                "score": result.score,
                "text": result.payload["text"],
                "source": result.payload["source"]
            }
        )
    return json.dumps({
        "results": formatted_results
    })