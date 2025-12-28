import os
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
import tiktoken
import pandas as pd

load_dotenv()

# ENV VARS
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "amalai")
PINECONE_HOST = os.getenv("PINECONE_HOST")

# DEBUG PRINTS (AFTER variables exist)
print("ENV CHECK:",
      "PINECONE_INDEX=", os.getenv("PINECONE_INDEX"),
      "PINECONE_HOST=", os.getenv("PINECONE_HOST"))

print("USING:",
      "PINECONE_INDEX=", PINECONE_INDEX,
      "PINECONE_HOST=", PINECONE_HOST)

print("ASSIGNMENT FILE LOADED:", __file__)

# =========================================================
# ENV VARS
# =========================================================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "amalai")
PINECONE_HOST = os.getenv("PINECONE_HOST")

LLMOD_API_KEY = os.getenv("OPENAI_API_KEY")  # LLMod key stored here
LLMOD_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.llmod.ai/v1")

assert PINECONE_API_KEY, "Missing PINECONE_API_KEY"
assert PINECONE_HOST, "Missing PINECONE_HOST"
assert LLMOD_API_KEY, "Missing OPENAI_API_KEY (LLMod key)"
assert LLMOD_BASE_URL, "Missing OPENAI_BASE_URL"

# =========================================================
# CONFIG (assignment required)
# =========================================================
EMBED_MODEL = "RPRTHPB-text-embedding-3-small"
CHAT_MODEL = "RPRTHPB-gpt-5-mini"

CHUNK_SIZE = 1024
OVERLAP_RATIO = 0.2

# Retrieval depth (increase if you need more candidates)
TOP_K = 30

# For summary/recommendation: how many chunks to show per chosen talk
MAX_CHUNKS_PER_TALK = 4

# =========================================================
# CLIENTS
# =========================================================
client = OpenAI(api_key=LLMOD_API_KEY, base_url=LLMOD_BASE_URL)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(name=PINECONE_INDEX, host=PINECONE_HOST)

# =========================================================
# CHUNKING (used only during ingest)
# ======st===================================================
enc = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return []
    tokens = enc.encode(text)
    overlap = int(CHUNK_SIZE * OVERLAP_RATIO)

    chunks, start = [], 0
    while start < len(tokens):
        end = min(len(tokens), start + CHUNK_SIZE)
        chunks.append(enc.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start = max(0, end - overlap)
    return chunks


# =========================================================
# EMBEDDINGS
# =========================================================
def embed_texts(texts: List[str]) -> List[List[float]]:
    res = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in res.data]


# =========================================================
# OPTIONAL: INGEST CSV INTO PINECONE (RUN ONLY WHEN NEEDED)
# Budget-friendly: you control limit.
# =========================================================
def ingest_csv_to_pinecone(csv_path: str, limit: int = None, batch_size: int = 50):
    df = pd.read_csv(csv_path)
    df = df[["talk_id", "title", "speaker_1", "transcript"]].dropna()

    records = []
    for _, row in df.iterrows():
        chunks = chunk_text(row["transcript"])
        for i, ch in enumerate(chunks):
            records.append(
                {
                    "id": f'{row["talk_id"]}_{i}',
                    "values": None,
                    "metadata": {
                        "talk_id": str(row["talk_id"]),
                        "title": str(row["title"]),
                        "speaker_1": str(row["speaker_1"]),
                        "chunk": ch,
                    },
                }
            )

    print("Total chunks in CSV:", len(records))
    if limit is not None:
        limit = min(limit, len(records))
    else:
        limit = len(records)

    for i in range(0, limit, batch_size):
        batch = records[i : i + batch_size]
        embs = embed_texts([x["metadata"]["chunk"] for x in batch])
        for v, e in zip(batch, embs):
            v["values"] = e

        index.upsert(vectors=batch, namespace="")
        time.sleep(0.1)

    print("Ingest complete")
    print(index.describe_index_stats())


# =========================================================
# RETRIEVAL HELPERS
# =========================================================
def retrieve_matches(question: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """Returns raw Pinecone matches."""
    q_emb = embed_texts([question])[0]
    res = index.query(
        vector=q_emb,
        top_k=top_k,
        include_metadata=True,
        namespace=""
    )

    return res.get("matches", []) or []


def normalize_matches(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten matches into a stable structure."""
    out = []
    for m in matches:
        md = m.get("metadata") or {}
        out.append(
            {
                "talk_id": str(md.get("talk_id", "")),
                "title": str(md.get("title", "")),
                "speaker_1": str(md.get("speaker_1", "")),
                "chunk": str(md.get("chunk", "")),
                "score": float(m.get("score", 0.0)),
            }
        )
    # Filter junk
    out = [x for x in out if x["talk_id"] and x["title"] and x["speaker_1"] and x["chunk"]]
    return out


def top_unique_talks(matches_norm: List[Dict[str, Any]], n: int = 3) -> List[Dict[str, Any]]:
    """
    Choose top-N unique talks by best score (max score among their chunks).
    Returns one representative record per talk, with a 'best_score'.
    """
    best_per_talk: Dict[str, Dict[str, Any]] = {}
    for r in matches_norm:
        tid = r["talk_id"]
        if tid not in best_per_talk or r["score"] > best_per_talk[tid]["best_score"]:
            best_per_talk[tid] = {
                "talk_id": tid,
                "title": r["title"],
                "speaker_1": r["speaker_1"],
                "best_score": r["score"],
            }
    talks = list(best_per_talk.values())
    talks.sort(key=lambda x: x["best_score"], reverse=True)
    return talks[:n]


def chunks_for_talk(matches_norm: List[Dict[str, Any]], talk_id: str, k: int = MAX_CHUNKS_PER_TALK) -> List[Dict[str, Any]]:
    """Get top-k chunks for the selected talk, sorted by score."""
    chunks = [r for r in matches_norm if r["talk_id"] == talk_id]
    chunks.sort(key=lambda x: x["score"], reverse=True)
    return chunks[:k]


# =========================================================
# ANSWERS FOR THE 4 REQUIRED TASK TYPES
# =========================================================
def answer_precise_fact(question: str) -> str:
    """
    Task 1: Find ONE talk + return title/speaker (fact retrieval).
    IMPORTANT: We return metadata directly (no LLM guessing).
    """
    matches = normalize_matches(retrieve_matches(question))
    if not matches:
        return "I don’t know based on the provided TED data."

    talks = top_unique_talks(matches, n=1)
    if not talks:
        return "I don’t know based on the provided TED data."

    t = talks[0]
    evidence_chunks = chunks_for_talk(matches, t["talk_id"], k=2)
    evidence = "\n".join([f'- "{c["chunk"][:220]}..."' for c in evidence_chunks])

    return (
        f'Title: {t["title"]}\n'
        f'Speaker: {t["speaker_1"]}\n'
        f"Evidence:\n{evidence}"
    )


def answer_multi_result_titles(question: str, n: int = 3) -> str:
    """
    Task 2: Return exactly 3 DISTINCT talk titles (not 3 chunks).
    """
    matches = normalize_matches(retrieve_matches(question))
    if not matches:
        return "I don’t know based on the provided TED data."

    talks = top_unique_talks(matches, n=n)
    if len(talks) < n:
        return "I don’t know based on the provided TED data."

    lines = []
    for i, t in enumerate(talks, start=1):
        lines.append(f'{i}. {t["title"]}')
    return "\n".join(lines)


SYSTEM_PROMPT_SUMMARY = """You are a TED Talk assistant.
Use ONLY the provided TED metadata + transcript excerpts.
Do NOT use external knowledge.
If you cannot answer from the context, say: "I don’t know based on the provided TED data."
Keep output concise.
"""

def answer_key_idea_summary(question: str) -> str:
    """
    Task 3: Find a relevant talk and summarize key idea.
    We still ensure title/speaker comes from metadata.
    """
    matches = normalize_matches(retrieve_matches(question))
    if not matches:
        return "I don’t know based on the provided TED data."

    talks = top_unique_talks(matches, n=1)
    if not talks:
        return "I don’t know based on the provided TED data."

    t = talks[0]
    ctx_chunks = chunks_for_talk(matches, t["talk_id"], k=MAX_CHUNKS_PER_TALK)

    context_text = "\n\n".join(
        [f'Title: {t["title"]}\nSpeaker: {t["speaker_1"]}\nExcerpt: {c["chunk"]}' for c in ctx_chunks]
    )

    user_prompt = (
        f"Question: {question}\n\n"
        f"Return exactly:\n"
        f"Title: <title>\n"
        f"Summary: <2-4 sentences>\n\n"
        f"Context:\n{context_text}"
    )

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_SUMMARY},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content


SYSTEM_PROMPT_RECO = """You are a TED Talk assistant.
Use ONLY the provided TED metadata + transcript excerpts.
Do NOT use external knowledge.
Pick ONE best talk and justify using evidence from excerpts (quote short phrases).
If you cannot answer from the context, say: "I don’t know based on the provided TED data."
"""

def answer_recommendation(question: str) -> str:
    """
    Task 4: Recommend ONE talk + evidence-based justification.
    We'll give the model 2-3 candidate talks (distinct) with their top chunks.
    """
    matches = normalize_matches(retrieve_matches(question))
    if not matches:
        return "I don’t know based on the provided TED data."

    candidates = top_unique_talks(matches, n=3)
    if not candidates:
        return "I don’t know based on the provided TED data."

    blocks = []
    for t in candidates:
        ctx_chunks = chunks_for_talk(matches, t["talk_id"], k=MAX_CHUNKS_PER_TALK)
        excerpts = "\n".join([f'- {c["chunk"]}' for c in ctx_chunks])
        blocks.append(
            f"Candidate Talk\nTitle: {t['title']}\nSpeaker: {t['speaker_1']}\nExcerpts:\n{excerpts}"
        )

    context_text = "\n\n---\n\n".join(blocks)

    user_prompt = (
        f"User request: {question}\n\n"
        f"Choose ONE talk from the candidates.\n"
        f"Return exactly:\n"
        f"Title: <title>\n"
        f"Speaker: <speaker>\n"
        f"Why this talk: <2-5 sentences with quotes/phrases from excerpts>\n\n"
        f"Context:\n{context_text}"
    )

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_RECO},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content
# =========================================================
# REQUIRED BY ASSIGNMENT API
# =========================================================

REQUIRED_SYSTEM_PROMPT = """You are a TED Talk assistant that answers questions strictly and
only based on the TED dataset context provided to you (metadata
and transcript passages). You must not use any external
knowledge. If the answer cannot be determined from the provided
context, respond: "I don’t know based on the provided TED data."
"""

def get_stats() -> Dict[str, Any]:
    """
    Required GET /api/stats
    """
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K
    }


def rag_answer_api(question: str) -> Dict[str, Any]:
    """
    Required POST /api/prompt
    Returns EXACT format required by the PDF
    """
    matches = normalize_matches(retrieve_matches(question))

    context = [
        {
            "talk_id": m["talk_id"],
            "title": m["title"],
            "chunk": m["chunk"],
            "score": m["score"]
        }
        for m in matches[:TOP_K]
    ]

    final_answer = rag_answer(question)

    return {
        "response": final_answer,
        "context": context,
        "Augmented_prompt": {
            "System": REQUIRED_SYSTEM_PROMPT,
            "User": question
        }
    }


# =========================================================
# SIMPLE ROUTER (one endpoint can handle all 4)
# =========================================================
def rag_answer(question: str) -> str:
    print("RAG ANSWER CALLED FROM:", __file__)
    print("RAG QUESTION:", question)
    q = question.lower()

    # Task 2 indicators
    if "exactly 3" in q or ("list" in q and "3" in q and "title" in q):
        return answer_multi_result_titles(question, n=3)

    # Task 4 indicators
    if "recommend" in q or "which talk would you recommend" in q:
        return answer_recommendation(question)

    # Task 3 indicators
    if "summary" in q or "key idea" in q:
        return answer_key_idea_summary(question)

    # Default Task 1
    return answer_precise_fact(question)


