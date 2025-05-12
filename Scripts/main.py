from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
from retrieval import hybrid_query

app = FastAPI(title="Hybrid Reddit Search API")

class QueryRequest(BaseModel):
    query: str
    filename: str  # e.g., "chatgpt_comments"

class HybridResponse(BaseModel):
    query: str
    answer: str
    quotes: List[str]
    references: List[dict]

@app.post("/query", response_model=HybridResponse)
def query_endpoint(request: QueryRequest):
    result = hybrid_query(request.query, request.filename)
    return result
