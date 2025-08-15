import logging
import os

import click
from dotenv import load_dotenv
from fastembed import TextEmbedding
from openai import OpenAI
from qdrant_client import QdrantClient

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client_qdrant = QdrantClient("http://localhost:6333")
client_openai = OpenAI()

COLLECTION_NAME = "chosen_movies"
EMBEDDING_DIMENSIONALITY = 384


@click.command()
@click.option(
    "--model-name",
    default="BAAI/bge-small-en",
    help="Embedding model name",
)
@click.option(
    "--collection-name",
    default="movies",
    help="Name for qdrant collection",
)
@click.option(
    "--top-k",
    default=5,
    help="Number of movies to recommend",
)
@click.option(
    "--query",
    default="a non-american romantic movie",
    help="Query to search movies",
)
def recommend_movies(
    model_name: str, collection_name: str, top_k: int, query: str
):

    embedding_model = TextEmbedding(model_name=model_name)

    logging.info("Entering query and search for answer")
    query_embedding = list(embedding_model.embed([query]))[0].tolist()
    results = client_qdrant.query_points(
        collection_name=collection_name,
        query=query_embedding,
        with_payload=True,
        limit=top_k,
    )

    recommendations = [hit.payload for hit in results.points]

    context = ""
    for movie in recommendations:
        context += (
            f"{movie["title"]}\n {movie["director"]}\n {movie["plot"]}\n\n"
        )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful cinephile"
            "Answer the questions using only the provided context."
            "Do not use any outside knowledge.",
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:",
        },
    ]

    response = client_openai.chat.completions.create(
        model="gpt-4o-mini", messages=messages
    )

    answer = response.choices[0].message.content.strip()
    print(answer)

    return answer


if __name__ == "__main__":
    recommend_movies()
