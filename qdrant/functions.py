from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from fastembed import TextEmbedding
import asyncio

client = QdrantClient(url="http://localhost:6333")

async def create_collection(collection_name):
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.DOT),
        )
        return "Success creating Collection"
    except Exception:
        return "Failed at creating Collection"
    
async def delete_collection(collection_name):
    try:
        client.delete_collection(collection_name=collection_name)
        return "Success creating Collection"
    except Exception:
        return "Failed at creating Collection"
    
async def embed_basic():
    # model.embed() takes a list of strings and returns one vector per string
    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    query_vec = list(model.embed(["car repair"]))[0]
    doc_vec   = list(model.embed(["automobile maintenance"]))[0]
    # check both vectors are the same length (384 dimensions each), then peek at the first 5 floats of each
    print(len(query_vec), len(doc_vec))
    print(query_vec[:5])
    print(doc_vec[:5])
    
if __name__ == "__main__":
    asyncio.run(create_collection("prueba"))