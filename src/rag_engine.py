import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

from .vector_search import VectorSearch

LOG_DIR = os.getenv("LOG_DIR", "logs/system")

# pylint: disable=duplicate-code, broad-exception-caught)
logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(LOG_DIR, "app.log"),
    format="%(asctime)s - %(levelname)s - %(message)s",
)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST")
client_openai = OpenAI()


class RAGEngine:
    def __init__(
        self,
        collection_name: str = "movies",
        emb_model_name: str = "BAAI/bge-small-en",
    ):
        self.vector_search = VectorSearch(
            host=QDRANT_HOST,
            port=6333,
            collection_name=collection_name,
            emb_model_name=emb_model_name,
        )
        self.retrieved_documents = []

    def create_context(self, query: str, top_k: int = 10) -> str:
        """Create context from vector search"""

        self.retrieved_documents = self.vector_search.search(
            query, top_k=top_k
        )

        context = ""
        for movie in self.retrieved_documents:
            context += f"Title: {movie["title"].title()}:\n"
            context += f"Director: {movie["director"]}\n"
            context += f"Plot: {movie["plot"]}\n\n"

        return context

    def generate_response(
        self, query: str, context: str, gpt_model_name: str = "gpt-4o-mini"
    ) -> str:
        """Generate answer using LLM/OpenAI"""

        messages = [
            {
                "role": "system",
                "content": "You are a helpful cinephile"
                "Answer the questions using only the provided context."
                "Do not use any outside knowledge.",
            },
            {
                "role": "user",
                "content": f"Context:{context}\n\nQuestion:{query}\n\nAnswer:",
            },
        ]

        response = client_openai.chat.completions.create(
            model=gpt_model_name, messages=messages
        )

        return response.choices[0].message.content.strip()
