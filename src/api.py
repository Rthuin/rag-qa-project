from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import json
import faiss
from sentence_transformers import SentenceTransformer

import os
from dotenv import load_dotenv
import openai
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------
# Initialize FastAPI app
# ---------------------------
app = FastAPI(title="RAG QA Backend (Educational Version)")

# ---------------------------
# Add CORS Middleware
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Load FAISS index and metadata
# ---------------------------
INDEX_PATH = "index/faiss.index"
META_PATH = "index/metadata.jsonl"
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
print("Loading FAISS index and metadata...")
index = faiss.read_index(INDEX_PATH)
metadata = [json.loads(line) for line in open(META_PATH, "r", encoding="utf-8")]
print(f"Loaded index with {len(metadata)} chunks")


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
    """Use OpenAI model to generate a grounded answer."""
    if not retrieved_chunks:
        return "I couldn’t find any relevant information."

    # Build context from retrieved text
    context = "\n\n".join([r["text"] for r in retrieved_chunks])

    # Construct the prompt
    prompt = f"""
You are an AI assistant. Use ONLY the information in the context below to answer the question clearly and concisely.

Context:
{context}

Question: {query}
Answer:
"""

    # Call OpenAI
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Make sure to use a valid model name
            messages=[
                {"role": "system", "content": "You are a helpful assistant that gives factual answers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300,
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error generating answer: {e}"

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
