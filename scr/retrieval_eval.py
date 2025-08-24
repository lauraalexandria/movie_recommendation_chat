import logging

import click
import pandas as pd
from dotenv import load_dotenv
from fastembed import TextEmbedding
from openai import OpenAI
from qdrant_client import QdrantClient
from tqdm import tqdm

# pylint: disable=duplicate-code
logging.basicConfig(
    level=logging.INFO,
    filename="app.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

load_dotenv()

client_qdrant = QdrantClient("http://localhost:6333")
client_openai = OpenAI()


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
    default=10,
    help="Number of movies to recommend",
)
# @click.option(
#     "--query",
#     default="a non-american romantic movie",
#     help="Query to search movies",
# )
def vector_search(
    model_name: str, collection_name: str, top_k: int, query: str
):

    embedding_model = TextEmbedding(model_name=model_name)

    # pylint: disable=duplicate-code, assignment-from-no-return
    # MOTIVO DE ERRO?
    logging.info("Entering query and search for answer")
    query_embedding = list(embedding_model.embed([query]))[0].tolist()
    vector_results = client_qdrant.query_points(
        collection_name=collection_name,
        query=query_embedding,
        with_payload=True,
        limit=top_k,
    )

    vector_recommendations = [hit.payload for hit in vector_results.points]

    answer = []
    for movie in vector_recommendations:
        answer = answer.append(f"{movie["title"]} - {movie["year"]}")

    return answer


def evaluate(ground_truth, search_function):
    relevance_total = []

    logging.info("Evaluating questions")
    for q in tqdm(ground_truth):
        doc_id = q["movie"]
        results = search_function(query=q["solicitations"])
        relevance = [m == doc_id for m in results]
        relevance_total.append(relevance)

    return {
        "hit_rate": hit_rate(relevance_total),
        "MRR": mrr(relevance_total),
    }


if __name__ == "__main__":

    logging.info("Load evaluation data")
    ground_truth_data = pd.read_csv(
        "./data/processed/ground-truth-retrieval.csv"
    )
    ground_truth_data = ground_truth_data.to_dict(orient="records")
    # metrics = evaluate(
    #     ground_truth_data,
    #     vector_search #lambda q: vector_search(query = q['solicitations'])
    # )

    relevance_test_total = []

    logging.info("Evaluating questions")
    for q_ in tqdm(ground_truth_data):
        doc_id_ = q_["movie"]

        ####################################################################
        MODEL_NAME = "BAAI/bge-small-en"
        COLLECTION_NAME = "movies"
        TOP_K = 10
        movie_query = q_["solicitations"]
        EMBEDDING_MODEL = TextEmbedding(model_name=MODEL_NAME)

        logging.info("Entering query and search for answer")
        query_embedding_ = list(EMBEDDING_MODEL.embed([movie_query]))[
            0
        ].tolist()
        vector_results_ = client_qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding_,
            with_payload=True,
            limit=TOP_K,
        )

        vector_recommendations_ = [
            hit.payload for hit in vector_results_.points
        ]

        answer_ = []
        for movie_ in vector_recommendations_:
            answer_ = answer_ + [f"{movie_["title"]} - {movie_["year"]}"]

        #######################################################################
        relevance_test = [m == doc_id_ for m in answer_]
        relevance_test_total.append(relevance_test)

    print(hit_rate(relevance_test_total))
    print(mrr(relevance_test_total))

    # A função tem que ficar mais bonita...
    # E ainda dá para eu fazer um boosting também?
    # 0.6511627906976745
    # 0.3821274763135227
    # tem vantagem em usar qdrant ao inves das bibliotecas do chat de receitas?
