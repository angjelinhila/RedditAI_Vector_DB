from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
from Retrieval import hybrid_query  # Case-sensitive

app = FastAPI(title="Hybrid Reddit Search API")
templates = Jinja2Templates(directory="Templates")  # Capital T to match your folder

class QueryRequest(BaseModel):
    query: str
    filename: str  # e.g., "chatgpt_comments"

class HybridResponse(BaseModel):
    query: str
    answer: str
    quotes: List[str]
    references: List[dict]

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search", response_class=HTMLResponse)
def search(request: Request, query: str = Form(...), filename: str = Form(...)):
    try:
        result = hybrid_query(query, filename)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "query": result["query"],
            "answer": result["answer"],
            "quotes": result["quotes"],
            "filename": filename
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "query": query,
            "answer": "⚠️ An error occurred while retrieving your results.",
            "quotes": [],
            "filename": filename,
            "error": str(e)
        })
