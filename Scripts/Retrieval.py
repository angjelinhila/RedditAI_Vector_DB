import os
import json
import requests
import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
from elasticsearch import Elasticsearch

# --- Load Models ---
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval().to("cpu")

# --- Elasticsearch Client ---
es = Elasticsearch("http://localhost:9200")

# --- Directory paths ---
FAISS_INDEX_DIR = "/tmp/colbert_indexes"
CORPUS_DIR = "/tmp/processed"

# --- Box shared links (replace with your actual shared URLs) ---
BOX_URLS = {
    "chatgpt_comments": {
        "index": "https://utexas.box.com/s/0evyjzslxi0a2h9xqydasilfpb4qa08s",
        "corpus": "https://utexas.box.com/s/zveaho2l2fqyz7mtfou0y943w4g63x90"
    },
    "chatgpt_submissions": {
        "index": "https://utexas.box.com/s/uc5bflmmexez5bsxqlk93kktlk456089",
        "corpus": "https://utexas.box.com/s/ov9twjo0x5dhhh7gesm65ihwodw8c9v9"
    },
    "artificial_comments": {
        "index": "https://utexas.box.com/s/my4upp3trp7s32pd3f5b77e64uklq83e",
        "corpus": "https://utexas.box.com/s/bus5d8ygiq17dxht0tfotfim6kbv8r8f"
    }, 
    "artificial_submissions": {
        "index": "https://utexas.box.com/s/0f10zszrlnj1870f4nmia166s499nw6t",
        "corpus": "https://utexas.box.com/s/i3uir20ccapeo01dlbzfxghoxogrb6ew"
    },
    "artificialinteligence_comments": {
        "index": "https://utexas.box.com/s/jplsyiyai56tv1fxopwf8mdpbbgn599t",
        "corpus": "https://utexas.box.com/s/7sleaqdqfg8887qpdte0llgzwrftw8ht"
    },
    "artificialinteligence_submissions": {
        "index": "https://utexas.box.com/s/g10kj6feftgrgfd2c2s8cxrmfewz2nfa",
        "corpus": "https://utexas.box.com/s/2rmqbfama00w76bhgf3hm42cn0p23je3"
    },
    "singularity_comments": {
        "index": "https://utexas.box.com/s/m9ptzia2enu6pfipb9tlurkalx2ajlrj",
        "corpus": "https://utexas.box.com/s/es4lz3m268rciwd010vlrticpyja12np"
    },
    "singularity_submissions": {
        "index": "https://utexas.box.com/s/z2qy81h2jehpeeln84qkkpphfb4vi7ud",
        "corpus": "https://utexas.box.com/s/q9msng8ocsx932oooko36kylhrwkgatd"
    },
    "openai_comments": {
        "index": "https://utexas.box.com/s/cg4gzcmzgyeunb4szanqqv2mjvtzy4bx",
        "corpus": "https://utexas.box.com/s/opoke8h0m00zioagg8hkr5zlsq5rlhpy"
    },
    "openai_submissions": {
        "index": "https://utexas.box.com/s/987eh4b20u8yze69em08i8ey4x8rqcyy",
        "corpus": "https://utexas.box.com/s/dx6i5d4mn8sodhuned3dg19dadieoxji"
    },
    "transhumanism_comments": {
        "index": "https://utexas.box.com/s/j0infymbofbrfl9pu32b8v129jmsdcbv",
        "corpus": "https://utexas.box.com/s/l293b6djkbzfy6phmbrcxj10gnuvycxk"
    },
    "transhumanism_submissions": {
        "index": "https://utexas.box.com/s/bvzph44ge35hg89u08ze8n7ckge5zl86",
        "corpus": "https://utexas.box.com/s/lpxvlaxwxqbxemynqljlcgndd2ocs1b5"
    },
    
}

# --- Global Parameters ---
TOP_K = 5

# --- Helper: Download if missing ---
def ensure_local_file(filename, url, local_path):
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"📥 Downloading {filename} from Box...")
        r = requests.get(url)
        with open(local_path, "wb") as f:
            f.write(r.content)

# --- Helper to load index + corpus dynamically ---
def load_index_and_corpus(filename):
    index_url = BOX_URLS[filename]["index"]
    corpus_url = BOX_URLS[filename]["corpus"]

    index_path = os.path.join(FAISS_INDEX_DIR, f"{filename}_colbert.index")
    corpus_path = os.path.join(CORPUS_DIR, f"{filename}_cleaned.json")

    ensure_local_file(f"{filename}.index", index_url, index_path)
    ensure_local_file(f"{filename}.json", corpus_url, corpus_path)

    index = faiss.read_index(index_path)
    with open(corpus_path, "r") as f:
        documents = [json.loads(line) for line in f]
    return index, documents

# --- Embedding Helper ---
def get_query_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        output = model(**inputs)
    return output.last_hidden_state.squeeze(0).mean(dim=0).numpy()

# --- Dense Retrieval ---
def dense_search(query, filename, top_k=TOP_K):
    index, documents = load_index_and_corpus(filename)
    query_vec = get_query_embedding(query).astype(np.float32)
    scores, indices = index.search(np.expand_dims(query_vec, axis=0), top_k)
    return [documents[i] for i in indices[0]]

# --- Sparse Retrieval (BM25/Elasticsearch) ---
def sparse_search(query, top_k=TOP_K):
    res = es.search(index="reddit_index", query={"match": {"body": query}}, size=top_k)
    return [hit["_source"].get("body", "") for hit in res["hits"]["hits"]]

# --- RAG Synthesis (Simple) ---
def generate_rag_answer(query, docs):
    context = " ".join(doc.get("body", "") for doc in docs)
    return f"Answer based on retrieved context: {context[:500]}..."

# --- Hybrid Entry Point ---
def hybrid_query(query, filename):
    dense_docs = dense_search(query, filename)
    sparse_quotes = sparse_search(query)
    answer = generate_rag_answer(query, dense_docs)
    return {
        "query": query,
        "answer": answer,
        "quotes": sparse_quotes,
        "references": dense_docs
    }
