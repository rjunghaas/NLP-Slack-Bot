from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

COLLECTION_NAME = "rfp_questions"
MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"   # 384-dim vectors, ~80MB, downloads once from HuggingFace
VECTOR_SIZE = 384

def load_model():
    # Downloads on first run, cached locally after that
    return SentenceTransformer(MODEL_NAME)

def embed(model, text: str) -> list[float]:
    # Converts a sentence to a 384-number vector
    return model.encode(text).tolist()

def embed_batch(model, texts: list[str]) -> list[list[float]]:
    # More efficient for loading many questions at once
    return model.encode(texts).tolist()

def search(client: QdrantClient, model, query: str, top_k: int = 3):
    # Embed the query, ask Qdrant for nearest neighbors
    # Returns list of (question, answer, score) tuples
    vector = embed(model, query)
    results = client.query_points(collection_name=COLLECTION_NAME, query=vector, with_payload=True, limit=top_k).points
    return [{
        "question": r.payload["question"],
        "answer": r.payload["answer"],
        "score": round(r.score, 3)
    } for r in results]