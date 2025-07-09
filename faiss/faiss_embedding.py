import faiss
import json
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import time

# Model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# Force CPU usage
device = torch.device("cpu")
model = model.to(device)

# Directory with processed JSONL files
DATA_DIR = "data/processed"
FAISS_OUTPUT_DIR = os.environ.get("SCRATCH", "/tmp") + "/faiss_indexes"
os.makedirs(FAISS_OUTPUT_DIR, exist_ok=True)

# Vector dimension for BERT
DIM = 768

def get_embedding_batch(texts, device=device):
    try:
        inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    except RuntimeError as e:
        print(f"Memory error on batch of size {len(texts)}: {e}. Retrying with smaller batch...")
        if len(texts) == 1:
            raise e
        mid = len(texts) // 2
        return np.vstack([
            get_embedding_batch(texts[:mid], device),
            get_embedding_batch(texts[mid:], device)
        ])

def index_faiss_for_file(filepath, index_path, batch_size=4, max_lines=1000):
    index = faiss.IndexFlatL2(DIM)
    buffer = []
    total_processed = 0
    start_time = time.time()

    with open(filepath, "r") as f:
        for i, line in enumerate(tqdm(f, desc=f"Indexing {os.path.basename(filepath)}")):
            if max_lines and i >= max_lines:
                break
            try:
                data = json.loads(line)
                text = data.get("body") or data.get("selftext")
                if text:
                    buffer.append(text)
                    if len(buffer) == batch_size:
                        embeddings = get_embedding_batch(buffer)
                        index.add(embeddings.astype(np.float32))
                        total_processed += len(buffer)
                        print(f"→ Processed {total_processed} entries")
                        buffer = []
            except Exception as e:
                print(f"Skipping line due to error: {e}")

    if buffer:
        embeddings = get_embedding_batch(buffer)
        index.add(embeddings.astype(np.float32))
        total_processed += len(buffer)
        print(f"→ Processed {total_processed} entries (final batch)")

    faiss.write_index(index, index_path)
    duration = time.time() - start_time
    print(f"✅ Saved FAISS index to {index_path} in {duration:.2f} seconds")

if __name__ == "__main__":
    if not os.access(FAISS_OUTPUT_DIR, os.W_OK):
        raise PermissionError(f"Cannot write to {FAISS_OUTPUT_DIR}. Check permissions.")

    for filename in os.listdir(DATA_DIR):
        if filename.endswith("_cleaned.json") and ("submissions" in filename or "comments" in filename):
            filepath = os.path.join(DATA_DIR, filename)
            index_path = os.path.join(FAISS_OUTPUT_DIR, f"{filename.replace('.json', '.index')}")
            if os.path.exists(index_path):
                print(f"✅ Skipping {filename}, index already exists.")
                continue
            print(f"🔄 Indexing {filename} ...")
            index_faiss_for_file(filepath, index_path, batch_size=4, max_lines=None)  # optionally set max_lines to debug
