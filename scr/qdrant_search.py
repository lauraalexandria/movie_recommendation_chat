import logging

import click
from fastembed import TextEmbedding
from qdrant_client import QdrantClient

client = QdrantClient("http://localhost:6333")

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
    default="a romantic movie",
    help="Query to search movies",
)
def recommend_movies(model_name, collection_name, top_k, query):

    embedding_model = TextEmbedding(model_name=model_name)

    logging.info("Entering query and search for answer")
    query_embedding = list(embedding_model.embed([query]))[0].tolist()
    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        with_payload=True,
        limit=top_k,
    )

    recommendations = [hit.payload for hit in results.points]

    for movie in recommendations:
        print(
            "*",
            movie["title"],
            "-",
            movie["director"],
            "-",
            movie["plot"][:100] + "...",
        )


if __name__ == "__main__":
    recommend_movies()
