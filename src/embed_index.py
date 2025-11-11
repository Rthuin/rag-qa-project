import os
import json
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def read_chunks(path):
    """Read chunked text file (JSONL)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def embed_texts(texts):
    """Embed text using a local SentenceTransformer model."""
    print("Loading model... (this may take a few seconds)")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.astype("float32")

def build_faiss_index(embeddings):
    """Build and return a FAISS index from embeddings."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # L2 = cosine-like distance
    index.add(embeddings)
    return index

def main(chunks_path="data/processed/chunks.jsonl", index_dir="index"):
    os.makedirs(index_dir, exist_ok=True)

    chunks = list(read_chunks(chunks_path))
    texts = [c["text"] for c in chunks]
    print(f"Loaded {len(texts)} text chunks from {chunks_path}")

    embeddings = embed_texts(texts)

    index = build_faiss_index(embeddings)

    index_path = os.path.join(index_dir, "faiss.index")
    faiss.write_index(index, index_path)

    metadata_path = os.path.join(index_dir, "metadata.jsonl")
    with open(metadata_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Index and metadata saved in {index_dir}")

if __name__ == "__main__":
    main()
