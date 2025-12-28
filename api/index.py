from fastapi import FastAPI
from pydantic import BaseModel

# IMPORTANT: import the API-level helpers
from indass.assignment import rag_answer_api, get_stats

app = FastAPI()


class PromptRequest(BaseModel):
    question: str


@app.post("/api/prompt")
def prompt(req: PromptRequest):
    """
    Required by assignment:
    POST /api/prompt
    """
    return rag_answer_api(req.question)


@app.get("/api/stats")
def stats():
    """
    Required by assignment:
    GET /api/stats
    """
    return get_stats()
