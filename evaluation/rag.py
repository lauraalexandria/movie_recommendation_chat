import click

from src.rag_engine import RAGEngine


@click.command()
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
    "--query",
    default="a non-american romantic movie",
    help="Query to search movies",
)
def rag_query(
    emb_model_name: str, collection_name: str, query: str, top_k: int
):
    """RAG requests CLI"""
    engine = RAGEngine(
        collection_name=collection_name, emb_model_name=emb_model_name
    )
    context = engine.create_context(query, top_k)
    response = engine.generate_response(query, context)

    click.echo(f"Solicitation: {query}")
    click.echo(f"\nAnswer:\n{response}")


if __name__ == "__main__":
    rag_query()
