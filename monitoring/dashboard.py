import json
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="RAG Chat Analytics", layout="wide")
st.title("📊 RAG Chat Performance Dashboard")


@st.cache_data
def load_data():
    chat_data = []
    feedback_data = []

    try:
        with open("logs/chat/chat_logs.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                chat_data.append(json.loads(line))
    except FileNotFoundError:
        st.warning("chat_logs.jsonl not found")

    try:
        with open(
            "logs/chat/feedback_logs.jsonl", "r", encoding="utf-8"
        ) as f:
            for line in f:
                feedback_data.append(json.loads(line))
    except FileNotFoundError:
        st.warning("feedbacks.jsonl not found")

    return pd.DataFrame(chat_data), pd.DataFrame(feedback_data)


chat_df, feedback_df = load_data()

if chat_df.empty:
    st.info("Waiting for feedback data...")
else:
    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Queries", len(chat_df))

    with col2:
        avg_sim = chat_df["average_similarity"].mean()
        st.metric("Average Similarity", f"{avg_sim:.2f}")

    with col3:
        avg_docs = chat_df["retrieved_count"].mean()
        st.metric("Docs by Query", f"{avg_docs:.1f}")

    with col4:
        avg_time = chat_df["response_time_seconds"].mean()
        st.metric("Average Response Time", f"{avg_time:.2f}")

    # Plots
    col1, col2 = st.columns(2)

    with col1:
        chat_df["solicitation"] = chat_df.index
        fig_sim = px.line(
            chat_df,
            x="solicitation",
            y="average_similarity",
            title="Avarage similarity over time/requests",
        )
        st.plotly_chart(fig_sim, use_container_width=True)

        all_genres = [
            genre
            for doc_list in chat_df["retrieved_documents"]
            for doc in doc_list
            for genre in doc.get("genres", "").split(", ")
        ]
        genre_counts = pd.Series(all_genres).value_counts().head(8)
        fig_genres = px.bar(
            x=genre_counts.values,
            y=genre_counts.index,
            title="Most Recovered Genres",
        )
        st.plotly_chart(fig_genres, use_container_width=True)

    with col2:
        fig_dist = px.histogram(
            chat_df,
            x="average_similarity",
            title="Average Similarity Distribution",
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        titles = []
        genres = []

        for doc_list in chat_df["retrieved_documents"]:
            for doc in doc_list:
                titles.append(doc.get("title", "Unknown"))
                genres.append(
                    doc.get("genres", "Unknown").split(", ")[0]
                    if doc.get("genres")
                    else "Unknown"
                )

        df_treemap = pd.DataFrame({"title": titles, "genre": genres})
        df_counts = (
            df_treemap.groupby(["genre", "title"])
            .size()
            .reset_index(name="count")
        )

        fig_titles = px.treemap(
            df_counts,
            path=["genre", "title"],
            values="count",
            title="Most Recovered Films by Genre",
        )
        st.plotly_chart(fig_titles, use_container_width=True)

    # Seção de feedbacks
    if not feedback_df.empty:
        st.subheader("📈 Users Feedbacks")

        col1, col2 = st.columns(2)

        with col1:
            rating_dist = feedback_df["feedback"].value_counts()
            fig_ratings = px.bar(
                x=rating_dist.index,
                y=rating_dist.values,
                title="Feedbacks Distribuition",
            )
            st.plotly_chart(fig_ratings, use_container_width=True)

        with col2:
            feedback_comments = feedback_df[feedback_df["comment"] != ""]
            st.dataframe(
                feedback_comments[["user_query", "feedback", "comment"]],
                use_container_width=True,
            )

    st.subheader("Recent Queries")
    recent_chats = chat_df[
        [
            "timestamp",
            "user_query",
            "average_similarity",
            "response_time_seconds",
        ]
    ].tail(10)
    st.dataframe(recent_chats, use_container_width=True)

if st.button("Export Report"):
    report = {
        "total_queries": len(chat_df),
        "avg_similarity": chat_df["average_similarity"].mean(),
        "avg_response_time": chat_df["response_time_seconds"].mean(),
        "generated_at": datetime.now().isoformat(),
    }
    st.code(report)
    st.download_button(
        "Download Report", json.dumps(report, indent=2), "rag_report.json"
    )
