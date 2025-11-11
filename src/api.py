from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import json
import faiss
from sentence_transformers import SentenceTransformer

# ---------------------------
# Initialize FastAPI app
# ---------------------------
app = FastAPI(title="RAG QA Backend (Educational Version)")

# ---------------------------
# Load FAISS index and metadata
# ---------------------------
INDEX_PATH = "index/faiss.index"
META_PATH = "index/metadata.jsonl"

print("🔍 Loading FAISS index and metadata...")
index = faiss.read_index(INDEX_PATH)
metadata = [json.loads(line) for line in open(META_PATH, "r", encoding="utf-8")]
print(f"✅ Loaded index with {len(metadata)} chunks")

# ---------------------------
# Load embedding model
# ---------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------
# Request schema
# ---------------------------
class QueryRequest(BaseModel):
    query: str
    k: int = 3

# ---------------------------
# Helper: Retrieve top-k chunks
# ---------------------------
def retrieve(query: str, k: int = 3):
    """Embed query, search FAISS, return top-k chunks."""
    query_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(query_emb, k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx >= 0 and idx < len(metadata):
            chunk = metadata[idx]
            results.append({
                "score": float(dist),
                "source": chunk["source"],
                "text": chunk["text"]
            })
    return results

# ---------------------------
# Helper: Simple local "generator"
# ---------------------------
def generate_answer(query: str, retrieved_chunks: list):
    """
    Simulate answer generation by combining retrieved chunks.
    This is NOT an AI model — it's rule-based for educational purposes.
    """
    if not retrieved_chunks:
        return "I couldn’t find any relevant information."

    # Combine the top retrieved chunks into one context string
    context = " ".join([r["text"] for r in retrieved_chunks])

    # If query keywords appear in the context, summarize
    if "fastapi" in query.lower() and "fastapi" in context.lower():
        return "FastAPI is a high-performance Python framework for building APIs. It supports async, uses type hints, and is designed for speed and simplicity."

    elif "python" in query.lower():
        return "Python is a general-purpose programming language widely used in web, data, and AI development."

    elif "rag" in query.lower():
        return "RAG (Retrieval-Augmented Generation) is an AI method that retrieves documents before generating an answer."

    else:
        return "Based on the context, I found the following relevant information:\n\n" + context

# ---------------------------
# Endpoint: /query
# ---------------------------
@app.post("/query")
def query_text(request: QueryRequest):
    """Retrieve relevant chunks and generate a simple answer."""
    retrieved = retrieve(request.query, request.k)
    answer = generate_answer(request.query, retrieved)
    return {
        "query": request.query,
        "answer": answer,
        "retrieved": retrieved
    }
