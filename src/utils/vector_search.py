import logging
from typing import Dict, List

from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient

# pylint: disable=duplicate-code
logging.basicConfig(
    level=logging.INFO,
    filename="app.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

load_dotenv()

client_qdrant = QdrantClient("http://localhost:6333")


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

        # print([hit.score for hit in search_results.points])
        # score = getattr(search_results.points, 'score', 0.0)
        # final_dict = [hit.payload for hit in search_results.points]
        # final_dict["score"] = score

        final_dicts = []
        for hit in search_results.points:
            aux_dict = hit.payload
            aux_dict["score"] = hit.score
            final_dicts.append(aux_dict)

        return final_dicts

    # def hybrid_search(self, query: str, top_k: int = 5,
    #  filters: Dict = None) -> List[Dict]:
    #     """Hybrid search"""
    #     query_embedding = self.get_embedding(query)

    #     search_results = self.client.query_points(
    #         collection_name=self.collection_name,
    #         query=query_embedding,
    #         limit=top_k,
    #         query_filter=filters
    #     )

    #     return [hit.payload for hit in search_results.points]
