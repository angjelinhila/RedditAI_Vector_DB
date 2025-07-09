import faiss
import numpy as np
import os
import time
from flask import Flask, request, jsonify

FAISS_INDEX_PATH = "/mnt/faiss_index/colbert_faiss.index"

# Wait for FAISS index to be available (Box Drive may stream it on demand)
while not os.path.exists(FAISS_INDEX_PATH):
    print("Waiting for FAISS index to become available from Box...")
    time.sleep(5)  # Give Box time to sync

# Load FAISS index
index = faiss.read_index(FAISS_INDEX_PATH)
print("FAISS index successfully loaded from Box cloud storage.")

# Flask API for FAISS search
app = Flask(__name__)

@app.route("/search", methods=["POST"])
def search():
    query_vector = np.array(request.json["vector"]).astype(np.float32).reshape(1, -1)
    distances, indices = index.search(query_vector, k=5)
    return jsonify({"results": indices.tolist(), "distances": distances.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
