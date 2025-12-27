from fastapi import FastAPI
from pydantic import BaseModel

from indass.assignment import rag_answer_api, get_stats

app = FastAPI()


class PromptRequest(BaseModel):
    question: str


@app.get("/api/stats")
def stats():
    return {
        "chunk_size": 1024,
        "overlap_ratio": 0.2,
        "top_k": 30
    }


@app.get("/api/stats")
def stats():
    return get_stats()
