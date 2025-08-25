import logging

import click
import pandas as pd
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

logging.basicConfig(
    level=logging.INFO,
    filename="app.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

client_qdrant = QdrantClient("http://localhost:6333")


@click.command()
@click.option(
    "--collection-name",
    default="movies",
    help="Name for qdrant collection",
)
@click.option(
    "--embedding-dimensionality",
    default=384,
    help="Vector dimensionality for embeddings",
)
@click.option(
    "--model-name",
    default="BAAI/bge-small-en",
    help="Embedding model name",
)
@click.option(
    "--path-source",
    default=".data/raw",
    help="Path for datasets",
)
def create_collection(
    collection_name: str,
    embedding_dimensionality: int,
    model_name: str,
    path_source: str,
):

    logging.info("Creating collection")
    client_qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=embedding_dimensionality, distance=models.Distance.COSINE
        ),
    )

    logging.info("Reading datasets")
    wiki = pd.read_csv(f"{path_source}/wiki_movie_plots_deduped.csv")
    boxd = pd.read_csv(f"{path_source}/ratings.csv")
    boxd = boxd[boxd["Rating"] >= 4]
    boxd = boxd.drop(["Date", "Letterboxd URI", "Rating"], axis=1)
    boxd = boxd.rename({"Name": "Title", "Year": "Release Year"}, axis=1)

    df = wiki.merge(boxd, on=["Title", "Release Year"])

    embedding_model = TextEmbedding(model_name=model_name)

    logging.info("Embedding select movies information")
    for _, row in df.iterrows():
        text = (
            f"{row['Title']} {row['Genre']} {row['Director']} {row['Plot']}"
        )
        embedding = list(embedding_model.embed([text]))[0].tolist()
        client_qdrant.upsert(
            collection_name=collection_name,
            points=[
                {
                    "id": _,
                    "vector": embedding,
                    "payload": {
                        "title": row["Title"],
                        "year": row["Release Year"],
                        "origin": row["Origin/Ethnicity"],
                        "director": row["Director"],
                        "cast": row["Cast"],
                        "genres": row["Genre"],
                        "plot": row["Plot"],
                    },
                }
            ],
        )


if __name__ == "__main__":
    create_collection()
