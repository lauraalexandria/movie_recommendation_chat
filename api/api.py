import os

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from src.rag_engine import RAGEngine

load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "movies")
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "BAAI/bge-small-en")
TOP_K = os.getenv("TOP_K")
GPT_MODEL_NAME = os.getenv("GPT_MODEL_NAME", "gpt-4o-mini")

app = FastAPI()
engine = RAGEngine(
    collection_name=COLLECTION_NAME, emb_model_name=EMB_MODEL_NAME
)


@app.get("/")
async def root():
    return {"message": "API RAG Engine is working!", "status": "online"}


class QueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K


@app.post("/rag/query")
async def rag_query(request: QueryRequest):
    context = engine.create_context(request.query, request.top_k)
    retrieved_docs = engine.retrieved_documents
    response = engine.generate_response(
        request.query, context, gpt_model_name=GPT_MODEL_NAME
    )
    return {"response": response, "retrieved_docs": retrieved_docs}
