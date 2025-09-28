import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

OMDB_API_KEY = os.getenv("OMDB_API_KEY")

PATH_SOURCE = "data/raw"
boxd = pd.read_csv(f"{PATH_SOURCE}/ratings.csv")

# Filter ratings and Exclude tv shows

boxd = boxd[boxd["Rating"] >= 4]
boxd = boxd[
    ~boxd["Name"].isin(
        [
            "O Auto da Compadecida",
            "Cosmos: A Personal Voyage",
            "Bo Burnham: The Inside Outtakes",
        ]
    )
]
boxd["Name"] = (
    boxd["Name"]
    .str.lower()
    .str.replace(" ", "+")
    .str.replace("’", "%E2%80%99")
    .str.replace(".", "")
    .str.replace("?", "%3F")
    # Translate some brazilian names
    .str.replace("brainstorm", "bicho+de+sete+cabe%C3%A7as")
    .str.replace("letter+beyond+the+walls", "carta+para+al%C3%A9m+dos+muros")
    .str.replace("the+wrung-out+man", "o+homem+que+virou+suco")
    .str.replace("the+quartet", "o+quatrilho")
    .str.replace("redeemer", "redentor")
    .str.replace("the+nutty+boy", "menino+maluquinho")
    .str.replace("the+blue+trial", "o+%C3%BAltimo+azul")
    .str.replace("my+uncle+killed+a+guy", "meu+tio+matou+um+cara")
    .str.replace("braindead", "dead+alive")
)
boxd["Name"] = boxd["Name"].str.split(":", expand=True)[0]


def get_movie_by_imdb_id(imdb_id, year, api_key):
    url_aux = f"?apikey={api_key}&t={imdb_id}&y={year}&plot=full"
    url = f"http://www.omdbapi.com/{url_aux}"
    response = requests.get(url)
    return response.json()


movies_data = {}
for index, row in boxd.iterrows():
    movie = get_movie_by_imdb_id(row["Name"], row["Year"], OMDB_API_KEY)
    if "Title" in movie:
        movies_data[index] = movie
    else:
        movie = get_movie_by_imdb_id(
            row["Name"], row["Year"] - 1, OMDB_API_KEY
        )
        if "Title" in movie:
            movies_data[index] = movie
        else:
            print(row["Name"])

movies_data = pd.DataFrame(movies_data).transpose()
movies_data = movies_data[
    [
        "Title",
        "Year",
        "Rated",
        "Released",
        "Runtime",
        "Genre",
        "Director",
        "Writer",
        "Actors",
        "Plot",
        "Language",
        "Country",
        "Awards",
    ]
]
movies_data[movies_data.select_dtypes(include=["object"]).columns] = (
    movies_data.select_dtypes(include=["object"]).apply(
        lambda x: x.str.lower()
    )
)
movies_data = pd.concat(
    [
        movies_data,
        movies_data["Genre"]
        .str.split(", ", expand=True)
        .add_prefix("Genre_"),
    ],
    axis=1,
)
movies_data = movies_data.drop("Genre", axis=1)

print(movies_data.shape)
print(boxd.shape)

if Path(f"{PATH_SOURCE}/movie_plots.csv").exists():

    previous_df = pd.read_csv(f"{PATH_SOURCE}/movie_plots.csv")
    if previous_df.shape[0] <= movies_data.shape[0]:
        movies_data.to_csv(f"{PATH_SOURCE}/movie_plots.csv", index=False)
    else:
        print(
            "The current dataframe is smaller than the previous saved version."
        )
else:
    movies_data.to_csv(f"{PATH_SOURCE}/movie_plots.csv", index=False)
