import json
import logging
import os

import click
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from src.vector_search import VectorSearch

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
    semantic_relevance = []
    sparse_relevance = []
    hybrid_relevance = []

    logging.info("Evaluating questions")
    for q in tqdm(ground_truth_data):
        doc_id = q["movie"]

        semantic_recommendations = vector_search.semantic_search(
            q["solicitations"], top_k=top_k
        )

        semantic_answer = []
        for movie in semantic_recommendations:
            semantic_answer = semantic_answer + [
                f"{movie["title"]} - {movie["year"]}"
            ]

        relevance_aux = [m == doc_id for m in semantic_answer]
        semantic_relevance.append(relevance_aux)

        sparse_recommendations = vector_search.sparse_search(
            q["solicitations"], top_k=top_k
        )

        sparse_answer = []
        for movie in sparse_recommendations:
            sparse_answer = sparse_answer + [
                f"{movie["title"]} - {movie["year"]}"
            ]

        relevance_aux = [m == doc_id for m in sparse_answer]
        sparse_relevance.append(relevance_aux)

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

    log_entry = {
        "Type": "Retrieval Evaluation",
        "Embedding Model": emb_model_name,
        "K": top_k,
        "Semantic Search HitRate": hit_rate(semantic_relevance),
        "Semantic Search MRR": mrr(semantic_relevance),
        "Sparse Search HitRate": hit_rate(sparse_relevance),
        "Sparse Search MRR": mrr(sparse_relevance),
        "Hybrid Search HitRate": hit_rate(hybrid_relevance),
        "Hybrid Search MRR": mrr(hybrid_relevance),
    }

    print(log_entry)

    with open("logs/system/eval.log", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    evaluate()
