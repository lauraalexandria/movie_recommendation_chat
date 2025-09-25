import os

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

OMDB_API_KEY = os.getenv("OMDB_API_KEY")
print(OMDB_API_KEY)

PATH_SOURCE = "data/raw"
boxd = pd.read_csv(f"{PATH_SOURCE}/ratings.csv")
boxd = boxd[boxd["Rating"] >= 4]
boxd["Name"] = boxd["Name"].str.lower().str.replace(" ", "_")


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

print(movies_data.shape)
print(boxd.shape)
movies_data.to_csv("data/raw/movie_plots.csv", index=False)
