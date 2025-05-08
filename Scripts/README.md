# Hybrid Reddit Search API

This project provides a hybrid retrieval system using both sparse (BM25 via Elasticsearch) and dense (ColBERT-style FAISS) indexing, exposed via a FastAPI endpoint.

---

## 🚀 Features
- Dense retrieval using FAISS
- Sparse keyword-based search using Elasticsearch
- Simple RAG-style synthesis for contextual answers
- JSON-based API interface with FastAPI

---

## 📁 Project Structure
```
Reddit_Vector_DB/
├── FAISS/                      # FAISS indexing scripts and index files
│   └── retrieval.py            # Hybrid retrieval logic
├── app/
│   └── main.py                 # FastAPI application
├── data/processed/            # Cleaned JSON data files
├── colbert_indexes/           # FAISS index files
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 📦 Installation

### 1. Clone repository
```bash
git clone <repo-url>
cd Reddit_Vector_DB
```

### 2. Build Docker container
```bash
docker build -t hybrid-reddit-api .
```

### 3. Run the API
```bash
docker run -p 8000:8000 hybrid-reddit-api
```

---

## 📡 API Usage

### Endpoint
**POST** `/query`

### Body (JSON)
```json
{
  "query": "What are users saying about chatbot bias?",
  "filename": "chatgpt_comments"
}
```

### Response (JSON)
```json
{
  "query": "...",
  "answer": "Answer based on retrieved context: ...",
  "quotes": ["...", "..."],
  "references": [{...}, {...}]
}
```

---

## 🧠 Notes
- All cleaned JSON files should live in `data/processed/`
- FAISS indexes must be named `[filename]_colbert.index`
- Elasticsearch must be running locally at `http://localhost:9200`

---

## 📬 To Do
- Swap RAG synthesis for an LLM
- Add support for query logs / analytics
- Deploy on Render or HuggingFace Spaces

---

Made with 🧠 by [Your Name]
