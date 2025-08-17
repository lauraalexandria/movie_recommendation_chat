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
    vector_results = client_qdrant.query_points(
        collection_name=collection_name,
        query=query_embedding,
        with_payload=True,
        limit=top_k,
    )

    vector_recommendations = [hit.payload for hit in vector_results.points]

    context = ""
    for movie in vector_recommendations:
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

    # limit = 1
    # hybrid_results = client_qdrant.query_points(
    #     collection_name=collection_name,
    #     prefetch=[
    #         models.Prefetch(
    #             query=models.Document(
    #                 text=query,
    #                 model=model_name, #"jinaai/jina-embeddings-v2-small-en",
    #             ),
    #             using="BAAI-small", #"jina-small",
    #             limit=(5 * limit),
    #         ),
    #         models.Prefetch(
    #             query=models.Document(
    #                 text=query,
    #                 model="Qdrant/bm25",
    #             ),
    #             using="bm25",
    #             limit=(5 * limit),
    #         ),
    #     ],
    #     # Fusion query enables fusion on the prefetched results
    #     query=models.FusionQuery(fusion=models.Fusion.RRF),
    #     with_payload=True,
    # )

    return answer


if __name__ == "__main__":
    recommend_movies()
