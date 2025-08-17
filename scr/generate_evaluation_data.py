import json
import logging
import os

import click
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client_openai = OpenAI()


@click.command()
@click.option(
    "--path-source",
    default=".data",
    help="Path for datasets",
)
@click.option(
    "--n-tests",
    default=3,
    help="Number of texts to generate",
)
def generate_evaluation_data(
    path_source: str,
    n_tests: int,
):

    logging.info("Reading datasets")
    wiki = pd.read_csv(f"{path_source}/raw/wiki_movie_plots_deduped.csv")
    boxd = pd.read_csv(f"{path_source}/raw/ratings.csv")
    boxd = boxd[boxd["Rating"] >= 4]
    boxd = boxd.drop(["Date", "Letterboxd URI", "Rating"], axis=1)
    boxd = boxd.rename({"Name": "Title", "Year": "Release Year"}, axis=1)

    df = wiki.merge(boxd, on=["Title", "Release Year"])

    # ACHAVA QUE PRECISAVA, MAS NÃO!
    # The record should contain the answer to the solicitations.
    # Create the awswers based in the context below.

    prompt_template = f"""
    You emulate a user that wants to find new movies.
    Formulate {n_tests} short solicitations this user might ask that the answer
    would be the movie in the context below.
    Make the solicitations in order to explore curiosity about genres, the
    plot and the director.
    Include factual solicitations and others that demand inference.
    Do not include the title of the movie in the solicitation.

    Context:

    Title: {{Title}}
    Director: {{Director}}
    Plot: {{Plot}}

    Retorn in JSON format:
    {{"solicitations": [{{"solicitation", "...", "solicitation"}}]}}
    """

    documents = df.to_dict(orient="records")
    results = {}
    for doc in tqdm(documents):

        doc_id = f"{doc["Title"]} - {doc["Release Year"]}"
        prompt = prompt_template.format(**doc)
        questions_raw = client_openai.chat.completions.create(
            model="gpt-4", messages=[{"role": "user", "content": prompt}]
        )
        questions_raw = questions_raw.choices[0].message.content
        questions = json.loads(questions_raw)
        results[doc_id] = questions["solicitations"]

    df_results = (
        pd.DataFrame.from_dict(results, orient="index")
        .stack()
        .reset_index()
        .drop(columns="level_1")
        .rename(columns={"level_0": "movie", 0: "solicitations"})
    )

    df_results.to_csv(
        f"{path_source}/processed/ground-truth-retrieval.csv", index=False
    )


if __name__ == "__main__":
    generate_evaluation_data()

    # ESSA BOSTA PAREOU DE FUNCIONAR!
    # talvez seja o qdrant que tava desconectado kkk
    # depois de gerar os dados melhor eu confirmar se algum registro
    # da coluna solicitations possui a string solicitation
