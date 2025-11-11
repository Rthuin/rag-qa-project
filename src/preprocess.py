import os
import json
from tqdm import tqdm

def chunk_text(text, chunk_size=50, overlap=10):
    """Split text into overlapping chunks (by words)."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def preprocess(input_dir="data/raw", output_file="data/processed/chunks.jsonl"):
    os.makedirs("data/processed", exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    
    with open(output_file, "w", encoding="utf-8") as out:
        for fname in tqdm(files, desc="Processing files"):
            path = os.path.join(input_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                record = {"source": fname, "chunk_id": i, "text": chunk}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f" Preprocessed {len(files)} files. Output saved to {output_file}")

if __name__ == "__main__":
    preprocess()
