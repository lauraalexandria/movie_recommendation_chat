from fastapi import FastAPI
from pydantic import BaseModel

from .utils.rag_engine import RAGEngine

app = FastAPI()
engine = RAGEngine(
    collection_name="movies", emb_model_name="BAAI/bge-small-en"
)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 10


@app.post("/rag/query")
async def rag_query(request: QueryRequest):
    context = engine.create_context(request.query, request.top_k)
    response = engine.generate_response(
        request.query, context, gpt_model_name="gpt-4o-mini"
    )
    return {"response": response}
