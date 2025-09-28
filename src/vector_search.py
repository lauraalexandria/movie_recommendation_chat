import logging
import os
import re
from typing import Dict, List

from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

# pylint: disable=duplicate-code
logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join("logs/system", "app.log"),
    format="%(asctime)s - %(levelname)s - %(message)s",
)

load_dotenv()


class VectorSearch:
    def __init__(
        self,
        collection_name: str = "movies",
        host: str = "localhost",
        port: int = 6333,
        emb_model_name: str = "BAAI/bge-small-en",
    ):
        self.client = QdrantClient(host=host, port=port)
        self.embedding_model = TextEmbedding(model_name=emb_model_name)
        self.collection_name = collection_name

    def get_embedding(self, text: str) -> List[float]:
        """Generate text embedding"""
        logging.info("Entering query and embedding")
        embeddings = list(self.embedding_model.embed([text]))
        return embeddings[0].tolist()

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Vector search"""
        query_embedding = self.get_embedding(query)

        logging.info("Searching for answers")
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            with_payload=True,
            limit=top_k,
        )

        final_dicts = []
        for hit in search_results.points:
            aux_dict = hit.payload
            aux_dict["score"] = hit.score
            final_dicts.append(aux_dict)

        return final_dicts

    def extract_genres(self, query):
        genres_list = [
            "action",
            "adventure",
            "animation",
            "comedy",
            "crime",
            "documentary",
            "drama",
            "fantasy",
            "horror",
            "mystery",
            "romance",
            "science fiction",
            "thriller",
            "western",
            "biography",
            "family",
            "history",
            "music",
            "musical",
            "war",
            "sport",
            "superhero",
            "noir",
            "rom-com",
            "romantic",
        ]

        found_genres = []
        query_lower = query.lower()

        for genre in genres_list:
            if re.search(r"\b" + re.escape(genre) + r"\b", query_lower):
                found_genres.append(genre)

        return found_genres

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Hybrid search"""
        query_embedding = self.get_embedding(query)
        genres = self.extract_genres(query)

        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="genres",
                        match=models.MatchValue(value=genres[0]),  # .title)
                    )
                ]
            ),
        )

        return [hit.payload for hit in search_results.points]
