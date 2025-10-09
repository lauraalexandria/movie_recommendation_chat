import logging
import os

import click
import pandas as pd
from qdrant_client import QdrantClient, models

logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join("logs/system", "app.log"),
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

    logging.info("Creating collections")
    client_qdrant.create_collection(
        collection_name=f"{collection_name}_semantic",
        vectors_config={
            f"{model_name}": models.VectorParams(
                size=embedding_dimensionality, distance=models.Distance.COSINE
            )
        },
    )

    client_qdrant.create_collection(
        collection_name=f"{collection_name}_sparse",
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        },
    )

    client_qdrant.create_collection(
        collection_name=f"{collection_name}_hybrid",
        vectors_config={
            f"{model_name}": models.VectorParams(
                size=embedding_dimensionality, distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        },
    )

    logging.info("Reading datasets")
    df = pd.read_csv(f"{path_source}/movie_plots.csv")

    points_semantic = []
    points_sparse = []
    points_hybrid = []
    for _, row in df.iterrows():

        text = f"""
        {row['Title']} {row['Genre_0']} {row['Genre_1']} {row['Genre_2']}
        {row['Director']} {row['Plot']}
        """
        payload = {
            "content": text,
            "title": row["Title"],
            "year": row["Year"],
            "origin": row["Country"],
            "director": row["Director"],
            "cast": row["Actors"],
            "genres": row["Genre_0"],
            "genres_sec": row["Genre_1"],
            "genres_ter": row["Genre_2"],
            "plot": row["Plot"],
            "id": _,
        }

        point_semantic = models.PointStruct(
            id=_,
            vector={
                f"{model_name}": models.Document(
                    text=text,
                    model=model_name,
                )
            },
            payload=payload,
        )
        points_semantic.append(point_semantic)

        point_sparse = models.PointStruct(
            id=_,
            vector={
                "bm25": models.Document(
                    text=text,
                    model="Qdrant/bm25",
                ),
            },
            payload=payload,
        )
        points_sparse.append(point_sparse)

        point_hybrid = models.PointStruct(
            id=_,
            vector={
                f"{model_name}": models.Document(
                    text=text,
                    model=model_name,
                ),
                "bm25": models.Document(
                    text=text,
                    model="Qdrant/bm25",
                ),
            },
            payload=payload,
        )
        points_hybrid.append(point_hybrid)

    client_qdrant.upsert(
        collection_name=f"{collection_name}_semantic", points=points_semantic
    )
    client_qdrant.upsert(
        collection_name=f"{collection_name}_sparse", points=points_sparse
    )
    client_qdrant.upsert(
        collection_name=f"{collection_name}_hybrid", points=points_hybrid
    )


if __name__ == "__main__":
    create_collection()
