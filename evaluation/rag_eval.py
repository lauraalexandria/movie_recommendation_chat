import json
import logging
import os

import click
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from tqdm import tqdm

from src.rag_engine import RAGEngine

from .simulate_cost import CostSimulation

# pylint: disable=too-many-arguments, too-many-positional-arguments,
# pylint: disable=too-many-locals
# logging.basicConfig(
#     level=logging.INFO,
#     filename=os.path.join("logs/system", "eval.log"),
#     format="%(asctime)s - %(levelname)s - %(message)s",
# )

logger_eval = logging.getLogger("eval")
logger_eval.addHandler(logging.FileHandler("logs/system/eval.log"))

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST")
client_openai = OpenAI()
client_qdrant = QdrantClient(f"http://{QDRANT_HOST}:6333")

EVALUATE_TEMPLATE = """
You are an expert evaluator for a RAG system.
Analyze the relevance of the generated answer to the given solicitation.
Based on the relevance of the generated answer, you will classify it
as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

solicitation: {solicitation}
Generated Answer: {answer}

Analyze content and context of the answer in relation to the solicitation
and provide your evaluation in parsable JSON without using code blocks:

{{
"Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
"Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()


def evaluation_category(
    evaluate_template: str,
    solicitation: str,
    answer: str,
    gpt_model_name: str,
):

    prompt = evaluate_template.format(
        solicitation=solicitation, answer=answer
    )

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
@click.option(
    "--gpt-model-name",
    default="gpt-4o-mini",
    help="ChatGPT model name",
)
@click.option(
    "--flag-simulate-cost",
    default=False,
    help="Flag to print simulated cost",
)
def evaluate_rag(
    ground_truth_path,
    emb_model_name,
    collection_name,
    top_k,
    gpt_model_name,
    flag_simulate_cost: bool,
):

    df_solicitation = pd.read_csv(ground_truth_path)
    ground_truth = df_solicitation.to_dict(orient="records")

    engine = RAGEngine(
        collection_name=collection_name, emb_model_name=emb_model_name
    )

    if flag_simulate_cost:

        simulate_cost = CostSimulation(gpt_model_name=gpt_model_name)

        prompt_list = []
        for q in tqdm(ground_truth):
            solicitation = q["solicitations"]

            prompt_list = prompt_list + [
                EVALUATE_TEMPLATE.format(
                    solicitation=solicitation, answer="8"
                )
            ]
        print(
            "This evaluation data generation will cost approximately $",
            simulate_cost.simulate_cost(prompt_list) * 2,
        )

    else:

        evaluations = []
        for q in tqdm(ground_truth):
            solicitation = q["solicitations"]
            # print(solicitation)
            context = engine.create_context(solicitation, top_k)
            rag_response = engine.generate_response(
                solicitation, context, gpt_model_name=gpt_model_name
            )

            messages = [
                {"role": "system", "content": "You are a helpful cinephile"}
            ]
            chat_response = client_openai.chat.completions.create(
                model=gpt_model_name, messages=messages
            )

            rag_evaluation = evaluation_category(
                evaluate_template=EVALUATE_TEMPLATE,
                solicitation=solicitation,
                answer=rag_response,
                gpt_model_name=gpt_model_name,
            )
            evaluation = json.loads(rag_evaluation)
            evaluations.append((q, rag_response, evaluation, "rag"))

            chat_evaluation = evaluation_category(
                evaluate_template=EVALUATE_TEMPLATE,
                solicitation=solicitation,
                answer=chat_response,
                gpt_model_name=gpt_model_name,
            )
            evaluation = json.loads(chat_evaluation)
            evaluations.append((q, chat_response, evaluation, "chat"))

        df_eval = pd.DataFrame(
            evaluations, columns=["record", "answer", "evaluation", "origin"]
        )

        df_eval["movie"] = df_eval.record.apply(lambda d: d["movie"])
        df_eval["solicitation"] = df_eval.record.apply(
            lambda d: d["solicitations"]
        )

        df_eval["relevance"] = df_eval.evaluation.apply(
            lambda d: d["Relevance"]
        )
        df_eval["explanation"] = df_eval.evaluation.apply(
            lambda d: d["Explanation"]
        )

        del df_eval["record"]
        del df_eval["evaluation"]
        df_eval.to_csv(
            "data/processed/df_eval.csv",
            index=False,
        )

        log_entry = (
            pd.crosstab(df_eval["relevance"], df_eval["origin"], dropna=False)
            / df_solicitation.shape[0]
        ).to_dict()
        print(log_entry)

        with open("logs/system/eval.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":

    evaluate_rag()
