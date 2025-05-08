from elasticsearch import Elasticsearch
import json

es = Elasticsearch("http://localhost:9200")

INDEX_NAME = "reddit_index"
INPUT_FILE = "data/processed/reddit_cleaned.jsonl"

def create_index():
    """Define the BM25 Elasticsearch index."""
    settings = {
        "settings": {"analysis": {"analyzer": {"default": {"type": "standard"}}}},
        "mappings": {
            "properties": {
                "submission_id": {"type": "keyword"},
                "subreddit": {"type": "keyword"},
                "user": {"type": "keyword"},
                "body": {"type": "text", "analyzer": "standard"},
                "timestamp": {"type": "date"},
            }
        },
    }
    es.indices.create(index=INDEX_NAME, body=settings, ignore=400)
    print(f"Index {INDEX_NAME} created.")

def index_data():
    """Ingest preprocessed Reddit data into Elasticsearch."""
    with open(INPUT_FILE, "r") as infile:
        for line in infile:
            doc = json.loads(line)
            es.index(index=INDEX_NAME, id=doc["submission_id"], body=doc)
    print("Data indexed successfully!")

if __name__ == "__main__":
    create_index()
    index_data()
