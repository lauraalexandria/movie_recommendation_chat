import json
import logging
import os

import click
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from .simulate_cost import CostSimulation

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client_openai = OpenAI()


# pylint: disable=too-many-locals
@click.command()
@click.option(
    "--path-source",
    default=".data/raw",
    help="Path for datasets",
)
@click.option(
    "--n-tests",
    default=3,
    help="Number of texts to generate",
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
def generate_evaluation_data(
    path_source: str,
    n_tests: int,
    gpt_model_name: str,
    flag_simulate_cost: bool,
):

    logging.info("Reading datasets")
    documents = pd.read_csv(f"{path_source}/raw/movie_plots.csv")

    json_format = (
        '{{"solicitations": ["solicitation", "...", "solicitation"]}}'
    )
    prompt_template = f"""
    You emulate a user that wants to find new movies.
    Formulate {n_tests} short solicitations this user might ask that the answer
    would be only the movie name in the context below.
    Make the movie solicitations in order to explore curiosity about genres,
    the plot and the director.
    Include factual solicitations and others that demand inference.
    Do not include the title of the movie in the solicitation.

    Context:

    Title: {{Title}}
    Director: {{Director}}
    Country: {{Country}}
    Plot: {{Plot}}

    Return in the exact JSON format, with double quotes:
    {json_format}
    """

    documents = documents.to_dict(orient="records")
    results = {}

    if flag_simulate_cost:

        simulate_cost = CostSimulation(gpt_model_name=gpt_model_name)

        prompt_list = []
        for doc in tqdm(documents):

            prompt_list = prompt_list + [prompt_template.format(**doc)]
        print(
            "This evaluation data generation will cost approximately $",
            simulate_cost.simulate_cost(prompt_list),
        )

    else:
        for doc in tqdm(documents):

            doc_id = f"{doc["Title"]} - {doc["Year"]}"
            prompt = prompt_template.format(**doc)
            question = client_openai.chat.completions.create(
                model=gpt_model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            question = question.choices[0].message.content
            print(question)
            if "```" in question:
                question = question.replace("```", "").replace("json", "")
            questions = json.loads(question)
            results[doc_id] = questions["solicitations"]

        df_results = (
            pd.DataFrame.from_dict(results, orient="index")
            .stack()
            .reset_index()
            .drop(columns="level_1")
            .rename(columns={"level_0": "movie", 0: "solicitations"})
        )

        path_source = f"{path_source}/processed"
        df_results.to_csv(
            f"{path_source}/ground-truth-retrieval-{gpt_model_name}.csv",
            index=False,
        )


if __name__ == "__main__":
    generate_evaluation_data()
