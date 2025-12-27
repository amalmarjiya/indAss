import os
import re
import time
import pandas as pd
from tqdm import tqdm
import tiktoken
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
load_dotenv()

# =========================================================
# ENV VARS (set in your OS / PyCharm run configuration / .env)
# =========================================================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "amalai")
PINECONE_HOST = os.getenv("PINECONE_HOST")

LLMOD_API_KEY = os.getenv("OPENAI_API_KEY")  # LLMod key stored here
LLMOD_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.llmod.ai/v1")

assert PINECONE_API_KEY, "Missing PINECONE_API_KEY"
assert PINECONE_HOST, "Missing PINECONE_HOST"
assert LLMOD_API_KEY, "Missing OPENAI_API_KEY (your LLMod key)"
assert LLMOD_BASE_URL, "Missing OPENAI_BASE_URL"

# =========================================================
# CONFIG (assignment required)
# =========================================================
EMBED_MODEL = "RPRTHPB-text-embedding-3-small"
CHAT_MODEL = "RPRTHPB-gpt-5-mini"

CHUNK_SIZE = 1024
OVERLAP_RATIO = 0.2
TOP_K = 15

# =========================================================
# CLIENTS
# =========================================================
client = OpenAI(api_key=LLMOD_API_KEY, base_url=LLMOD_BASE_URL)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(name=PINECONE_INDEX, host=PINECONE_HOST)

print(index.describe_index_stats())

# =========================================================
# LOAD DATASET
# =========================================================
df = pd.read_csv("ted_talks_en.csv")
df = df[["talk_id", "title", "speaker_1", "transcript"]].dropna()
print("Talks loaded:", len(df))

# =========================================================
# CHUNKING
# =========================================================
enc = tiktoken.get_encoding("cl100k_base")

def chunk_text(text: str):
    tokens = enc.encode(re.sub(r"\s+", " ", str(text)).strip())
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
# BUILD RECORDS
# =========================================================
records = []
for _, row in df.iterrows():
    for i, ch in enumerate(chunk_text(row["transcript"])):
        records.append({
            "id": f'{row["talk_id"]}_{i}',
            "values": None,
            "metadata": {
                "talk_id": str(row["talk_id"]),
                "title": row["title"],
                "speaker_1": row["speaker_1"],
                "chunk": ch
            }
        })

print("Total chunks:", len(records))

# =========================================================
# EMBEDDINGS
# =========================================================
def embed_texts(texts):
    res = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in res.data]

print("Embedding dim:", len(embed_texts(["test"])[0]))

# =========================================================
# UPSERT (LIMITED)
# =========================================================
BATCH = 50
LIMIT = 2000

for i in tqdm(range(0, min(LIMIT, len(records)), BATCH)):
    batch = records[i:i+BATCH]
    embs = embed_texts([x["metadata"]["chunk"] for x in batch])
    for v, e in zip(batch, embs):
        v["values"] = e
    index.upsert(vectors=batch)
    time.sleep(0.1)

print("Upsert complete")
print(index.describe_index_stats())

# =========================================================
# RAG
# =========================================================
SYSTEM_PROMPT = """You are a TED Talk assistant that answers questions strictly and
only based on the TED dataset context provided to you (metadata
and transcript passages). You must not use any external
knowledge, the open internet, or information that is not explicitly
contained in the retrieved context. If the answer cannot be
determined from the provided context, respond: “I don’t know
based on the provided TED data.” Always explain your answer
using the given context, quoting or paraphrasing the relevant
transcript or metadata when helpful.
"""

def rag_answer(question: str):
    q_emb = embed_texts([question])[0]
    res = index.query(vector=q_emb, top_k=TOP_K, include_metadata=True)

    matches = res.get("matches", [])
    if not matches:
        return "I don’t know based on the provided TED data."

    context_text = "\n\n---\n\n".join(m["metadata"]["chunk"] for m in matches)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context_text}"}
        ]
    )
    return response.choices[0].message.content

print(rag_answer("Find a TED talk about overcoming fear"))
