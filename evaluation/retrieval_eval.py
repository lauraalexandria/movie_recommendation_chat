import logging
import os

import click
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from tqdm import tqdm
from utils.vector_search import VectorSearch

# pylint: disable=duplicate-code
logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join("logs/system", "app.log"),
    format="%(asctime)s - %(levelname)s - %(message)s",
)

load_dotenv()

client_qdrant = QdrantClient("http://localhost:6333")


def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)


def mrr(relevance_total):
    total_score = 0.0

    for line in relevance_total:
        # pylint: disable=consider-using-enumerate
        for rank in range(len(line)):
            if line[rank] is True:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relevance_total)


@click.command()
@click.option(
    "--ground-truth-path",
    default="./data/processed/ground-truth-retrieval.csv",
    help="Path to ground truth data",
)
@click.option(
    "--emb-model-name",
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
    default=10,
    help="Number of movies to recommend",
)
def evaluate(ground_truth_path, collection_name, emb_model_name, top_k):

    logging.info("Load evaluation data")
    ground_truth_data = pd.read_csv(ground_truth_path)
    ground_truth_data = ground_truth_data.to_dict(orient="records")

    vector_search = VectorSearch(
        collection_name=collection_name, emb_model_name=emb_model_name
    )
    relevance_test_total = []

    logging.info("Evaluating questions")
    for q in tqdm(ground_truth_data):
        doc_id = q["movie"]

        vector_recommendations = vector_search.search(
            q["solicitations"], top_k=top_k
        )

        answer = []
        for movie in vector_recommendations:
            answer = answer + [f"{movie["title"]} - {movie["year"]}"]

        relevance_test = [m == doc_id for m in answer]
        relevance_test_total.append(relevance_test)

    logging.info("HitRate: %s", hit_rate(relevance_test_total))
    logging.info("MRR: %s", mrr(relevance_test_total))

    return {
        "hit_rate": hit_rate(relevance_test_total),
        "MRR": mrr(relevance_test_total),
    }


if __name__ == "__main__":
    evaluate()
