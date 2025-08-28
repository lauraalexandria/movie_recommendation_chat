import datetime
import json
import os
import uuid
from typing import Any, Dict, List

import click
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from utils.rag_engine import RAGEngine

load_dotenv()

DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# ERROS
# Se eu dou um feedback vai para o registro errado
# Não consigo comentar E dar a nota
# O cálculo da similaridade sempre aparece igual a zero, eita...
# Curioso como Coherence sempre aparece como sugestão... ou sou eu?
# refatoração das pastas

# pylint: disable=too-many-arguments, too-many-positional-arguments
# pylint: disable=too-many-locals, too-many-statements, broad-exception-caught


class EnhancedChatMonitor:
    def __init__(self, log_file: str = "chat_logs.jsonl"):
        self.log_file = log_file
        self.session_id = str(uuid.uuid4())

    def log_interaction(
        self,
        user_query: str,
        retrieved_docs: List[Dict],
        llm_response: str,
        response_time: float,
        user_feedback: str = None,
        feedback_comment: str = None,
    ) -> Dict[str, Any]:
        """Register complete interaction metadata"""

        # Extract metadata from retrieved documents
        doc_metadata = []
        for doc in retrieved_docs:
            doc_metadata.append(
                {
                    "doc_id": doc.get("id", "unknown"),
                    "title": doc.get("title", "No title"),
                    "source": doc.get("source", "unknown"),
                    "similarity_score": doc.get("score", 0),
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
            "user_feedback": user_feedback,
            "feedback_comment": feedback_comment,
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


def initialize_session_state():
    """Initialize session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = {}
    if "current_query" not in st.session_state:
        st.session_state.current_query = None


def display_chat_metrics(response_time: float, retrieved_docs: List[Dict]):
    """Exibe métricas da conversa"""
    with st.expander("📊 Métricas desta resposta"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("⏱️ Tempo", f"{response_time:.2f}s")

        with col2:
            st.metric("📄 Documentos", len(retrieved_docs))

        with col3:
            avg_score = (
                sum(doc.get("score", 0) for doc in retrieved_docs)
                / len(retrieved_docs)
                if retrieved_docs
                else 0
            )
            st.metric("⭐ Similaridade", f"{avg_score:.3f}")

        # Mostrar documentos recuperados
        st.write("**Documentos utilizados:**")
        for i, doc in enumerate(retrieved_docs, 1):
            with st.expander(
                f"Documento {i}: {doc.get('title', 'Sem título')}"
            ):
                st.write(f"**Fonte:** {doc.get('source', 'Desconhecida')}")
                st.write(f"**Similaridade:** {doc.get('score', 0):.3f}")
                if doc.get("content"):
                    st.write(f"**Conteúdo:** {doc['content'][:200]}...")


def get_user_feedback():
    """Interface para feedback do usuário"""
    st.write("---")
    st.write("💭 Esta resposta foi útil?")

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("👍", use_container_width=True, key="feedback_positive"):
            return "positive", None

    with col2:
        if st.button("👎", use_container_width=True, key="feedback_negative"):
            return "negative", None

    with col3:
        feedback_comment = st.text_input(
            "Comentário (opcional):",
            key="feedback_comment",
            placeholder="O que poderia ser melhorado?",
        )
        if feedback_comment:
            return "with_comment", feedback_comment

    return None, None


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
def main(collection_name, emb_model_name, top_k, gpt_model_name):
    # Page configuration
    st.set_page_config(page_title="Chat RAG", page_icon="🤖", layout="wide")

    # Engine RAG initialization
    # (?) @st.cache_resource
    initialize_session_state()
    monitor = EnhancedChatMonitor()
    rag_engine = RAGEngine(
        collection_name=collection_name, emb_model_name=emb_model_name
    )

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
                logs_df = pd.read_json("chat_logs.jsonl", lines=True)
                st.dataframe(logs_df)
            except FileNotFoundError:
                st.warning("No logs available yet. The file doesn't exist.")
            except ValueError as e:
                st.warning(f"Log file is empty or invalid: {e}")
            except Exception as e:
                st.error(f"Error loading logs: {e}")

    # To show chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # To show feedback
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
        with st.spinner("🔍 Buscando informações..."):
            start_time = datetime.datetime.now()

            # Fase de retrieval
            # retrieved_docs = rag_engine.retrieve_documents(prompt, top_k=3)

            # Fase de generation
            context = rag_engine.create_context(prompt, top_k=top_k)
            retrieved_docs = rag_engine.retrieved_documents
            response = rag_engine.generate_response(
                query=prompt,
                context=context,
                gpt_model_name=gpt_model_name,
            )

            response_time = (
                datetime.datetime.now() - start_time
            ).total_seconds()

        # Resposta do assistente
        with st.chat_message("assistant"):
            st.markdown(response)
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )

            # Mostrar métricas
            display_chat_metrics(response_time, retrieved_docs)

            # Registrar interação (sem feedback ainda)
            log_entry = monitor.log_interaction(
                user_query=prompt,
                retrieved_docs=retrieved_docs,
                llm_response=response,
                response_time=response_time,
            )

            # Store interaction metadata
            st.session_state.last_interaction = log_entry

        # Solicitar feedback
        feedback, comment = get_user_feedback()
        if feedback:
            # Atualizar log com feedback
            updated_log = monitor.log_interaction(
                user_query=prompt,
                retrieved_docs=retrieved_docs,
                llm_response=response,
                response_time=response_time,
                user_feedback=(
                    feedback if feedback != "with_comment" else "commented"
                ),
                feedback_comment=comment,
            )

            # Marcar feedback como dado para esta mensagem
            st.session_state.feedback_given[
                len(st.session_state.messages) - 1
            ] = feedback
            # Store log entry for feedback
            st.session_state.last_interaction = updated_log

            # Feedback visual
            if feedback == "positive":
                st.success("✅ Obrigado pelo feedback positivo!")
            elif feedback == "negative":
                st.warning("📝 Obrigado pelo feedback. Vamos melhorar!")
            else:
                st.info("💡 Obrigado pelo comentário detalhado!")

            # Atualizar mensagem com feedback
            st.session_state.messages[-1]["feedback"] = feedback
            st.rerun()


if __name__ == "__main__":
    main()
