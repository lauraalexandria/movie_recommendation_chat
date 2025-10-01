import logging
import os

import click
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from src.vector_search import VectorSearch

logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join("logs/system", "retrieval_eval.log"),
    format="%(asctime)s - %(levelname)s - %(message)s",
)

load_dotenv()
QDRANT_HOST = os.getenv("QDRANT_HOST")


# pylint: disable=too-many-locals, broad-exception-caught
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
        host=QDRANT_HOST,
        port=6333,
        collection_name=collection_name,
        emb_model_name=emb_model_name,
    )
    search_relevance = []
    hybrid_relevance = []

    logging.info("Evaluating questions")
    for q in tqdm(ground_truth_data):
        doc_id = q["movie"]

        search_recommendations = vector_search.search(
            q["solicitations"], top_k=top_k
        )

        search_answer = []
        for movie in search_recommendations:
            search_answer = search_answer + [
                f"{movie["title"]} - {movie["year"]}"
            ]

        relevance_aux = [m == doc_id for m in search_answer]
        search_relevance.append(relevance_aux)

        hybrid_search_fail = 0
        try:
            hybrid_recommendations = vector_search.hybrid_search(
                q["solicitations"], top_k=top_k
            )

            hybrid_answer = []
            for movie in hybrid_recommendations:
                hybrid_answer = hybrid_answer + [
                    f"{movie["title"]} - {movie["year"]}"
                ]

            relevance_aux = [m == doc_id for m in hybrid_answer]
            hybrid_relevance.append(relevance_aux)
        except Exception:
            hybrid_search_fail += 1

    logging.info("Embedding Model: %s K: %s", emb_model_name, top_k)
    logging.info("Vector Search HitRate: %s", hit_rate(search_relevance))
    logging.info("Vector Search MRR: %s", mrr(search_relevance))

    logging.info("Hybrid Search HitRate: %s", hit_rate(hybrid_relevance))
    logging.info("Hybrid Search MRR: %s", mrr(hybrid_relevance))
    logging.info("Hybrid Search Fails: %s", hybrid_search_fail)

    print(
        "search_hit_rate",
        hit_rate(search_relevance),
        "search_MRR",
        mrr(search_relevance),
        "hybrid_hit_rate",
        hit_rate(hybrid_relevance),
        "hybrid_MRR",
        mrr(hybrid_relevance),
    )

    return {
        "search_hit_rate": hit_rate(search_relevance),
        "search_MRR": mrr(search_relevance),
        "hybrid_hit_rate": hit_rate(hybrid_relevance),
        "hybrid_MRR": mrr(hybrid_relevance),
    }


if __name__ == "__main__":
    evaluate()
