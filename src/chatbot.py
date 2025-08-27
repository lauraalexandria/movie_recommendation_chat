import click
import streamlit as st

from utils.rag_engine import RAGEngine


@click.command()
@click.option(
    "--collection-name",
    default="movies",
    help="Name for qdrant collection",
)
@click.option(
    "--emb-model-name",
    default="BAAI/bge-small-en",
    help="Embedding model name",
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
def chatbot(collection_name, emb_model_name, top_k, gpt_model_name):
    # Page configuration
    st.set_page_config(page_title="Chat RAG", page_icon="🤖")

    # Engine RAG initialization
    @st.cache_resource
    def get_rag_engine():
        return RAGEngine(
            collection_name=collection_name, emb_model_name=emb_model_name
        )

    rag_engine = get_rag_engine()

    # Title and description
    st.title("📽️🍿🎞️ Movie Recommendation Chat with RAG")
    st.write(
        """
        Make solicitations and questions about movies!
        (but mainly recommendations)
        """
    )

    # Iniciate chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # To show chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if prompt := st.chat_input("Enter your solicitation..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate RAG answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                context = rag_engine.create_context(prompt, top_k=top_k)
                response = rag_engine.generate_response(
                    query=prompt,
                    context=context,
                    gpt_model_name=gpt_model_name,
                )

            # Show answer
            st.markdown(response)

        # Add chat answer to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )


if __name__ == "__main__":

    chatbot()
