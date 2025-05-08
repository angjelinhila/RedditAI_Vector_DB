from elasticsearch import Elasticsearch, helpers
import json
import os
from tqdm import tqdm

# Constants
ES_HOST = "http://localhost:9200"
DATA_DIR = "/Users/angjelin/Library/CloudStorage/Box-Box/Reddit Vector DB/Data/Processed"
INDEX_PREFIX = "reddit"
BULK_SIZE = 1000  # Number of docs per bulk request

# Connect to Elasticsearch
es = Elasticsearch(ES_HOST)

def create_index(index_name):
    settings = {
        "settings": {
            "analysis": {
                "analyzer": {
                    "default": {"type": "standard"}
                }
            }
        },
        "mappings": {
            "properties": {
                "submission_id": {"type": "keyword"},
                "id": {"type": "keyword"},
                "parent_id": {"type": "keyword"},
                "subreddit": {"type": "keyword"},
                "user": {"type": "keyword"},
                "title": {"type": "text"},
                "body": {"type": "text"},
                "selftext": {"type": "text"},
                "tokenized_body": {"type": "text"},
                "score": {"type": "integer"},
                "timestamp": {"type": "date"}
            }
        },
    }
    es.indices.create(index=index_name, body=settings, ignore=400)
    print(f"Index {index_name} created or already exists.")

def index_file_bulk(filepath, index_name):
    with open(filepath, "r") as f:
        actions = []
        for i, line in enumerate(tqdm(f, desc=f"Indexing {os.path.basename(filepath)}")):
            doc = json.loads(line)
            doc_id = doc.get("submission_id") or doc.get("id")
            actions.append({
                "_index": index_name,
                "_id": doc_id,
                "_source": doc
            })

            if len(actions) >= BULK_SIZE:
                helpers.bulk(es, actions)
                actions = []

        # Index any remaining documents
        if actions:
            helpers.bulk(es, actions)

if __name__ == "__main__":
    # Process each of the 12 files
    for filename in os.listdir(DATA_DIR):
        if filename.endswith("_cleaned.json") and ("submissions" in filename or "comments" in filename):
            filepath = os.path.join(DATA_DIR, filename)
            index_name = f"{INDEX_PREFIX}_{filename.replace('_cleaned.json', '').lower()}"
            create_index(index_name)
            index_file_bulk(filepath, index_name)
