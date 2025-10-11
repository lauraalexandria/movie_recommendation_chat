import logging
import os
from typing import Dict, List

from dotenv import load_dotenv
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
        self.emb_model_name = emb_model_name
        self.collection_name = collection_name

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Vector search"""

        logging.info("Searching for answers")
        search_results = self.client.query_points(
            collection_name=f"{self.collection_name}_semantic",
            query=models.Document(
                text=query,
                model=self.emb_model_name,
            ),
            using=self.emb_model_name,
            with_payload=True,
            limit=top_k,
        )

        final_dicts = []
        for hit in search_results.points:
            aux_dict = hit.payload
            aux_dict["score"] = hit.score
            final_dicts.append(aux_dict)

        return final_dicts

    def sparse_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Vector search"""

        logging.info("Searching for answers")
        search_results = self.client.query_points(
            collection_name=f"{self.collection_name}_sparse",
            query=models.Document(
                text=query,
                model="Qdrant/bm25",
            ),
            using="bm25",
            limit=top_k,
            with_payload=True,
        )

        final_dicts = []
        for hit in search_results.points:
            aux_dict = hit.payload
            aux_dict["score"] = hit.score
            final_dicts.append(aux_dict)

        return final_dicts

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Hybrid search"""

        search_results = self.client.query_points(
            collection_name=f"{self.collection_name}_hybrid",
            query=models.Document(
                text=query,
                model=self.emb_model_name,
            ),
            using=self.emb_model_name,
            limit=top_k,
            with_payload=True,
        )

        return [hit.payload for hit in search_results.points]
