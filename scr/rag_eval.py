import json
import logging
import os

import pandas as pd
from dotenv import load_dotenv
from fastembed import TextEmbedding
from openai import OpenAI
from qdrant_client import QdrantClient

# from rag import recommend_movies
from tqdm import tqdm

# pylint: disable=duplicate-code
logging.basicConfig(
    level=logging.INFO,
    filename="app.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client_openai = OpenAI()
client_qdrant = QdrantClient("http://localhost:6333")


# @click.command()
# @click.option(
#     "--model-name",
#     default="BAAI/bge-small-en",
#     help="Embedding model name",
# )
def llm(prompt: str, gpt_model_name: str):

    answer_message = [
        # {
        #    "role": "system",
        #     "content": "You are a helpful cinephile"
        #     "Answer the questions using only the provided context."
        #     "Do not use any outside knowledge.",
        # },
        {"role": "user", "content": prompt},
    ]

    answer = client_openai.chat.completions.create(
        model=gpt_model_name, messages=answer_message
    )
    # CONSIGUERIA COLOCAR TODAS OS PROMPTS DE UMA VEZ?

    answer = answer.choices[0].message.content.strip()

    return answer


if __name__ == "__main__":

    EVALUATE_TEMPLATE = """
    You are an expert evaluator for a RAG system.
    Analyze the relevance of the generated answer to the given solicitation.
    Based on the relevance of the generated answer, you will classify it
    as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

    Here is the data for evaluation:

    solicitation: {solicitation}
    Generated Answer: {answer_llm}

    Analyze content and context of the answer in relation to the solicitation
    and provide your evaluation in parsable JSON without using code blocks:

    {{
    "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
    "Explanation": "[Provide a brief explanation for your evaluation]"
    }}
    """.strip()

    df_solicitation = pd.read_csv(
        "./data/processed/ground-truth-retrieval.csv"
    )
    ground_truth = df_solicitation.to_dict(orient="records")
    df_sample = df_solicitation.sample(n=200, random_state=1)
    sample = df_sample.to_dict(orient="records")

    evaluations = []

    for q in tqdm(sample):
        solicitation = q["solicitations"]
        # answer_llm = recommend_movies(
        # model_name="BAAI/bge-small-en",
        # collection_name="movies", top_k=10, query=solicitation)

        ####################################################################
        MODEL_NAME = "BAAI/bge-small-en"
        COLLECTION_NAME = "movies"
        TOP_K = 10
        query = q["solicitations"]
        embedding_model = TextEmbedding(model_name=MODEL_NAME)

        # pylint: disable=duplicate-code
        logging.info("Entering query and search for answer")
        query_embedding = list(embedding_model.embed([query]))[0].tolist()
        vector_results = client_qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            with_payload=True,
            limit=TOP_K,
        )

        vector_recommendations = [
            hit.payload for hit in vector_results.points
        ]

        CONTEXT = ""
        for movie in vector_recommendations:
            title = movie["title"]
            director = movie["director"]
            plot = movie["plot"]
            CONTEXT += f"{title}\n {director}\n {plot}\n\n"

        messages = [
            {
                "role": "system",
                "content": "You are a helpful cinephile"
                "Answer the questions using only the provided context."
                "Do not use any outside knowledge.",
            },
            {
                "role": "user",
                "content": f"Context: {CONTEXT}\nQuestion: {query}\nAnswer:",
            },
        ]

        response = client_openai.chat.completions.create(
            model="gpt-4o-mini", messages=messages
        )

        answer_llm = response.choices[0].message.content.strip()

        #######################################################################

        awswer_prompt = EVALUATE_TEMPLATE.format(
            solicitation=solicitation, answer_llm=answer_llm
        )

        evaluation = llm(prompt=awswer_prompt, gpt_model_name="gpt-4o-mini")
        evaluation = json.loads(evaluation)

        evaluations.append((q, answer_llm, evaluation))

    df_eval_mini = pd.DataFrame(
        evaluations, columns=["record", "answer", "evaluation"]
    )

    df_eval_mini["movie"] = df_eval_mini.record.apply(lambda d: d["movie"])
    df_eval_mini["solicitation"] = df_eval_mini.record.apply(
        lambda d: d["solicitations"]
    )

    df_eval_mini["relevance"] = df_eval_mini.evaluation.apply(
        lambda d: d["Relevance"]
    )
    df_eval_mini["explanation"] = df_eval_mini.evaluation.apply(
        lambda d: d["Explanation"]
    )

    del df_eval_mini["record"]
    del df_eval_mini["evaluation"]

    print(df_eval_mini.relevance.value_counts(normalize=True))

    # gpt-4o-mini
    # RELEVANT           0.735
    # NON_RELEVANT       0.145
    # PARTLY_RELEVANT    0.120

    # "gpt-5-nano": {"in": 0.05, "out": 0.40}
