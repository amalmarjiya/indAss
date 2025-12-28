from fastapi import FastAPI
from pydantic import BaseModel

# IMPORTANT: absolute import

from indass.assignment import rag_answer_api


app = FastAPI()


class PromptRequest(BaseModel):
    question: str


@app.post("/api/prompt")
def prompt(req: PromptRequest):
    return rag_answer_api(req.question)



