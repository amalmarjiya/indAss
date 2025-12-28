from fastapi import FastAPI
from pydantic import BaseModel

# IMPORTANT: absolute import
from indass.assignment import rag_answer, index as pinecone_index

app = FastAPI()


class PromptRequest(BaseModel):
    question: str


@app.post("/api/prompt")
def prompt(req: PromptRequest):
    return {"response": rag_answer(req.question)}


@app.get("/api/stats")
def stats():
    return pinecone_index.describe_index_stats()
