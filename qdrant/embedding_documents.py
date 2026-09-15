from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import PointStruct

client = QdrantClient(
    host="localhost",
    port=6333
)

reader = PdfReader(r"C:\Users\U16_RSanchez#OGOB\Desktop\Clase 2\Demostration-of-Vectorial-Databases\REGLAMENTO INTERIOR DE TRABAJO 0426 2.pdf")

texto = ""
for page in reader.pages:
    texto += page.extract_text() + "\n"
    
print(texto[:1000])

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(texto)

print(f"Chunks: {len(chunks)}")

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

points = []

for i, (document, embedding) in enumerate(zip(chunks, embeddings)):
    points.append(
        PointStruct(
            id=i,
            vector=embedding.tolist(),
            payload={
                "text": document,
                "source": "REGLAMENTO INTERIOR DE TRABAJO 0426 2.pdf"
            }
        )
    )
    
client.upsert(
    collection_name="prueba",
    points=points
)