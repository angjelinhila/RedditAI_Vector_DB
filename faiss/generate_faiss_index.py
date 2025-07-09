import os
# Prevent tokenizer parallel thread error on TACC (must precede transformers)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["RAYON_NUM_THREADS"] = "1"

import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import faiss
from pathlib import Path

# Use DistilBERT or ColBERT-style model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

device = torch.device("cpu")
model.to(device)

# Relative to the root of your repo
DATA_DIR = Path("Data/Processed").resolve()
OUTPUT_DIR = Path("faiss/index").resolve()
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Embedding dimension
DIM = 768

def get_token_embeddings(text, device=device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.squeeze(0).cpu().numpy()  # shape: (seq_len, dim)

def index_colbert_style(filepath, index_path, max_lines=None):
    index = faiss.IndexFlatIP(DIM)
    ids = []

    with open(filepath, "r") as f:
        for i, line in enumerate(tqdm(f, desc=f"Indexing {os.path.basename(filepath)}")):
            if max_lines and i >= max_lines:
                break
            try:
                data = json.loads(line)
                main_text = data.get("body") or data.get("selftext") or ""
                title = data.get("title", "")
                subreddit = data.get("subreddit", "")
                combined_text = f"{title} {main_text} {subreddit}"

                if combined_text.strip():
                    token_embeddings = get_token_embeddings(combined_text)
                    for token_vector in token_embeddings:
                        index.add(np.expand_dims(token_vector.astype(np.float32), axis=0))
                        ids.append(i)
            except Exception as e:
                print(f"Skipping line {i}: {e}")

    faiss.write_index(index, index_path)
    print(f"✅ Saved ColBERT-style index to: {index_path}")

if __name__ == "__main__":
    for filename in os.listdir(DATA_DIR):
        if filename.endswith("_cleaned.json"):
            input_path = os.path.join(DATA_DIR, filename)
            index_name = filename.replace(".json", "_colbert.index")
            output_path = os.path.join(OUTPUT_DIR, index_name)

            if os.path.exists(output_path):
                print(f"⏭️ Skipping {filename}, index already exists.")
                continue

            print(f"🔄 Indexing {filename}...")
            index_colbert_style(input_path, output_path, max_lines=None)
