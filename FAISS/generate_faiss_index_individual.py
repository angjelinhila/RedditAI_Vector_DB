
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import faiss

# Use DistilBERT or ColBERT-style model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

device = torch.device("cpu")
model.to(device)

# TACC paths
DATA_DIR = "/work/10481/angjelinhila8932/Reddit_Project/Reddit Vector DB Temp/Data/Processed"
OUTPUT_DIR = "/work/10481/angjelinhila8932/Reddit_Project/Reddit Vector DB Temp/FAISS/index"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Embedding dimension
DIM = 768

def get_token_embeddings(text, device=device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.squeeze(0).cpu().numpy()  # shape: (seq_len, dim)

def index_colbert_style(filepath, index_path, max_lines=1000):
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
    test_file = os.path.join(DATA_DIR, "openai_submissions_cleaned.json")
    index_file = os.path.join(OUTPUT_DIR, "openai_submissions_colbert.index")
    print(f"🔄 ColBERT-style indexing for: {test_file}")
    index_colbert_style(test_file, index_file, max_lines=1000)
