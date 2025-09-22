import datetime
import json
import os
from typing import Any, Dict, List

import click
import pandas as pd
import requests
import streamlit as st
import uuid6
from dotenv import load_dotenv

# current_dir = Path(__file__).parent
# root_dir = current_dir.parent
# sys.path.append(str(root_dir))

load_dotenv()

DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# pylint: disable=too-many-arguments, too-many-positional-arguments
# pylint: disable=too-many-locals, too-many-statements, broad-exception-caught


class EnhancedChatMonitor:
    def __init__(self, log_file: str = "logs/chat/chat_logs.jsonl"):
        self.log_file = log_file
        self.session_id = str(uuid6.uuid7())

    def ask_rag(self, query: str):
        response = requests.post(
            "http://rag-api:8000/rag/query", json={"query": query}
        )
        return response.json()["response"], response.json()["retrieved_docs"]

    def log_interaction(
        self,
        user_query: str,
        retrieved_docs: List[Dict],
        llm_response: str,
        response_time: float,
    ) -> Dict[str, Any]:
        """Register complete interaction metadata"""

        # Extract metadata from retrieved documents
        doc_metadata = []
        for doc in retrieved_docs:
            doc_metadata.append(
                {
                    "title": doc.get("title", "No title"),
                    "year": doc.get("year", "unknown"),
                    "origin": doc.get("origin", "unknown"),
                    "director": doc.get("director", "unknown"),
                    "cast": doc.get("cast", "unknown"),
                    "genres": doc.get("genres", "unknown"),
                    "similarity_score": doc.get("score", None),
                    "content_preview": (
                        str(doc.get("content", ""))[:100] + "..."
                        if doc.get("content")
                        else None
                    ),
                }
            )

        # Create log entry
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "session_id": self.session_id,
            "user_query": user_query,
            "retrieved_documents": doc_metadata,
            "retrieved_count": len(retrieved_docs),
            "llm_response": llm_response,
            "response_time_seconds": round(response_time, 2),
            "context_used": [doc["title"] for doc in doc_metadata],
            "average_similarity": (
                round(
                    sum(doc.get("score", 0) for doc in retrieved_docs)
                    / len(retrieved_docs),
                    4,
                )
                if retrieved_docs
                else 0
            ),
        }

        # Saving log
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        return log_entry

    def log_feedback(
        self,
        user_query: str,
        llm_response: str,
        feedback: str,
        comment: str = None,
    ):
        """Log user feedback when entered"""
        feedback_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "user_query": user_query,
            "llm_response": llm_response,
            "feedback": feedback,
            "comment": comment,
        }

        with open(
            "logs/chat/feedback_logs.jsonl", "a", encoding="utf-8"
        ) as f:
            f.write(json.dumps(feedback_entry, ensure_ascii=False) + "\n")


def initialize_session_state():
    """Initialize session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "feedback_data" not in st.session_state:
        st.session_state.feedback_data = {}


def display_chat_metrics(response_time: float, retrieved_docs: List[Dict]):
    """Show metrics chat"""
    with st.expander("📊 Metrics related to the answer:"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("⏱️ Response time", f"{response_time:.2f}s")

        with col2:
            st.metric("📄 Documents used", len(retrieved_docs))

        with col3:
            avg_score = (
                sum(doc.get("score", 0) for doc in retrieved_docs)
                / len(retrieved_docs)
                if retrieved_docs
                else 0
            )
            st.metric("⭐ Similarity", f"{avg_score:.3f}")

        # Show retrieved documents
        st.write("**Used Documents:**")
        for i, doc in enumerate(retrieved_docs, 1):
            with st.expander(f"Title {i}: {doc.get('title', 'No title')}"):
                st.write(f"**Source:** {doc.get('source', 'Unknown')}")
                st.write(f"**Similarity:** {doc.get('score', 0):.3f}")
                if doc.get("content"):
                    st.write(f"**Content:** {doc['content'][:200]}...")


def get_user_feedback():
    """Feedback Interface for user - ONLY for the lanst message"""

    bottom = None
    feedback_comment = None

    if (
        len(st.session_state.messages) > 1
        and st.session_state.messages[-1]["role"] == "assistant"
        and "feedback" not in st.session_state.messages[-1]
    ):

        st.write("---")
        st.write("💭 Esta resposta foi útil?")

        col1, col2, col3, _ = st.columns([1, 1, 2, 4])

        with col1:
            if st.button("👍", key="feedback_positive"):
                bottom = "positive"

        with col2:
            if st.button("👎", key="feedback_negative"):
                bottom = "negative"

        with col3:
            feedback_comment = st.text_input(
                "Comentary (opcional):",
                key="feedback_comment",
                placeholder="Any comment?",
            )

    return bottom, feedback_comment


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
def main():  # collection_name, emb_model_name, top_k, gpt_model_name
    # Page configuration
    st.set_page_config(page_title="Chat RAG", page_icon="🤖", layout="wide")

    # Engine RAG initialization
    initialize_session_state()
    monitor = EnhancedChatMonitor()

    # Title and description
    st.title("📽️🍿🎞️ Movie Recommendation Chat with RAG")
    st.caption(
        """
        Make solicitations and questions about movies!
        (but mainly recommendations)
        """
    )

    # Sidebar informations
    with st.sidebar:
        st.header("ℹ️ About this chat")
        st.write(
            """
        This chat uses RAG (Retrieval-Augmented Generation) to answer
        your movie questions based on a knowledge base.

        **Colected Metadata:**
        - Your solicitation
        - Retrieved documents
        - Response time
        - Feedback (optional)
        """
        )

        # Usage logs button (development only)
        if st.button("📊 See usage logs ", disabled=not DEBUG):
            try:
                logs_df = pd.read_json(
                    "logs/chat/chat_logs.jsonl", lines=True
                )
                st.dataframe(logs_df)
            except FileNotFoundError:
                st.warning("No logs available yet. The file doesn't exist.")
            except ValueError as e:
                st.warning(f"Log file is empty or invalid: {e}")
            except Exception as e:
                st.error(f"Error loading logs: {e}")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show feedback if exists
            if message.get("feedback"):
                emoji = "👍" if message["feedback"] == "positive" else "👎"
                st.caption(f"*Your feedback: {emoji}*")

    # User Input
    if prompt := st.chat_input("Enter your solicitation..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Processar com RAG
        with st.spinner("🔍 Thinking..."):
            start_time = datetime.datetime.now()

            # Generate response step
            response, retrieved_docs = monitor.ask_rag(query=prompt)

            response_time = (
                datetime.datetime.now() - start_time
            ).total_seconds()

        # Add assistant response WITHOUT feedback initially
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

        # Log interaction
        monitor.log_interaction(
            user_query=prompt,
            retrieved_docs=retrieved_docs,
            llm_response=response,
            response_time=response_time,
        )

        # Force rerun to show new message and feedback interface
        st.rerun()

    # Show feedback interface ONLY for the last message
    if (
        len(st.session_state.messages) > 0
        and st.session_state.messages[-1]["role"] == "assistant"
        and "feedback" not in st.session_state.messages[-1]
    ):

        feedback, comment = get_user_feedback()
        if feedback:
            # Add feedback to the last message
            st.session_state.messages[-1]["feedback"] = feedback

            # Log feedback
            monitor.log_feedback(
                user_query=st.session_state.messages[-2][
                    "content"
                ],  # Last user query
                llm_response=st.session_state.messages[-1][
                    "content"
                ],  # Last assistant response
                feedback=feedback,
                comment=comment,
            )

            # Rerun to update UI
            st.rerun()


if __name__ == "__main__":
    main()
