# Movie Recommendation Chat

The goal of this project is to develop and build an end-to-end Retrieval-Augmented Generation (RAG) chat movie recommendation expert, delivering a friendly chat interface and a monitoring dashboard with performance and feedback.

[photos from the chat and de monitoting dash?]

The project includes:
* **Data Extraction:** Downloding selected movie plots and data from OMDb API
* **Data preparation/Embedding:** Data transformation to unstructured data
* **Modeling/RAG:** Brings base knowledge to ChatGPT API
* **User Interface:** Deploys the chat in a friendly interface
* **Monitoring:** Dashboard about performance and feedback
* **Deployment:** Model and applications deployment using Docker

## Tools used

* Conda - Virtual enviroment
* Pylint - Static code analyser
* pre-commit - pre-commit hooks
* OMDb API
* Qdrant - embedding?
* OpenAI API
* FastAPI
* Streamlit - Chat UI and monitoring dashboard

## Projet Struture

```
.
├── Makefile
├── README.md
├── api
│   ├── Dockerfile
│   └── api.py
├── data
│   ├── processed
│   └── raw
├── docker-compose.yaml
├── evaluation
│   ├── __init__.py
│   ├── generate_evaluation_data.py
│   ├── rag.py
│   ├── rag_eval.py
│   ├── retrieval_eval.py
│   └── simulate_cost.py
├── logs
│   ├── chat
│   └── system
├── monitoring
│   ├── Dockerfile
│   ├── dashboard.py
├── requirements.txt
├── src
│   ├── Dockerfile
│   ├── __init__.py
│   ├── create_qdrant_collection.py
│   ├── data_extractor.py
│   ├── prev_data_extractor.py
│   ├── rag_engine.py
│   └── vector_search.py
└── ui
    ├── Dockerfile
    └── chatbot.py
```


## Model

This single make command includes: (teria como o docker-compose já criar a coleção?)

1. (Optional) Generate the database that will be used with the knowledge base on which the chat should be based;
1. Creates the Qdrant collection, in this case, it will be generated for both semantic, vector, and hybrid search. This means that all films present in the knowledge base will be projected into the three types of defined vector spaces, a process known as embedding.
    * Consequently, the api that is used by the chat uses the rag_engine/vector_search, in a way that the user enter a query, this query is embbended using the same model as the collection this query is compared to the movies plots the K biggest similarity are selected, and then used as the context in the ChatGPT API and then the answer is generated and appears in the chat!
1. The API with the select RAG is hosted
1. The chatbot is created and hosted
1. After a user interacts with the chatbot,the person can evaluate each of the answers of a positive or negative vote and also add a comment;


```
make complete-chat-flow (uai preciso?)
```

### Evaluation and Selection

I tested the Vector Search, that compares only the similarity, and a Hybrid Search the filter applied where based in genres. The k vary between 5, 10 and 15, but the increase between 10 and 15 weren't so significant.

In order to test the results i created a ground truth data

#### Retrieval Evaluation

| Method  | HitRate@5 | MRR@5 | HitRate@10 | MRR@10 | HitRate@15 | MRR@15 | Selected |  
|-|-|-|-|-|-|-|-|
| Semantic | 0.7143 | 0.5889 |  0.7862 | 0.5985 | 0.8198 | 0.6011| |
| **BM25** | **0.8469** | **0.7808** | **0.8833** | **0.7857** |  **0.8973** | **0.7868** |✅ |
| Hybrid  | 0.7143 | 0.5889 | 0.7862 | 0.5985 | 0.8198 | 0.6011 | |

#### LLM Evaluation

| Prompt Strategy       | Factuality | Relevance | Fluency | Selected |
|-----------------------|------------|-----------|---------|----------|
| Raw chat | 0.80       | 0.78      | 0.87    | ✅ |
| RAG      | **0.84**   | **0.82**  | 0.85    | ✅ |

the bare chat can make recommendations based on directors history, but my rag can make recommendations of new movies and non-american movies

## How to execute locally

### Requirements

* Makefile
* Conda
* Docker
* Qdrant

### Setup project with Makefile

1. Create and activate enviroment
```
make setup
```

2. Activate enviroment
```
source ~/.bashrc && conda activate chat-env
```

3. Install dependencies and pre-commit
```
make install
```

### Add your Credentials

Create your own .env based on .env.example
```
cp .env.example .env
```

And change the default values to your needs:
1. (Optional) The OMDB credentials can be view in ....
1. The OpenAI credentials can be view in https://platform.openai.com/, after create an account, search for "API Keys";
1. Note that the parameters selected for the model are defined in the .env (e.g EMB_MODEL_NAME, TOP_K, GPT_MODEL_NAME), but in personal tests you can customize both the .env and the Makefile

### Build Dockers

Build the dockers, so the Qdrant will be available so the API and the Chat and de monitoring chat

```
docker-compose up --build -d
```

#### Create Qdrant Collection

```
make create-qdrant-collections
```

#### Access Points

| Service         | URL                                  |
|-----------------|--------------------------------------|
| Qdrant          | http://localhost:6333/dashboard      |
| FastAPI         | http://localhost:8000/docs           |
| Chat            | http://localhost:8501                |
| Monitor Dash    | http://localhost:8502                |


### FastAPI

Open an API that's possible to include the requests and receive the model answers.

With the API open, you can also add the data with bash:
``` bash
curl -X POST "http://localhost:8000/rag/query"   -H "Content-Type: application/json"  -d '{"query": "i wanna a few comedy movies for quiet people", "top_k": 10}'
```

### Chat with Streamlit

Creates the UI interface where the user can make your solicitations and keeps the historic.

```
make chatbot
```

E tem outros logs e tal

The chatbot interface includes a feedback interface that also register those feedbacks in order to "feed" the monitoring dashboard. The feedback is based in a positive/negative button and a optional comment.

Those data follows those formats

{
    "title": doc.get("title", "No title"),
    "year": doc.get("year", "unknown"),
    "origin": doc.get("origin", "unknown"),
    "director": doc.get("director", "unknown"),
    "cast": doc.get("cast", "unknown"),
    "genres": doc.get("genres", "unknown"),
    "similarity_score": doc.get("score", None),
    "content_preview": "Content used"
}

{
    "timestamp": datetime.datetime.now().isoformat(),
    "session_id": self.session_id,
    "user_query": user_query,
    "retrieved_documents": doc_metadata,
    "retrieved_count": len(retrieved_docs),
    "llm_response": llm_response,
    "response_time_seconds": round(response_time, 2),
    "context_used": [doc["title"] for doc in doc_metadata],
    "average_similarity": Average similarity
}

For feedback:

{
    "timestamp": datetime.datetime.now().isoformat(),
    "user_query": user_query,
    "llm_response": llm_response,
    "feedback": feedback,
    "comment": comment,
}

### Monitoring model

The Streamlit monitoring dashboard provides:

Time-response
Similarity between the query and the context
Movies/Genres retrieved
User feedback (thumbs up/down)

```
make monitor
```

### Deativate enviroment

```
docker-compose down -v
```

```
conda deactivate
```

## How to run your own version of the project

### Data Extration

As recomendações foram baseadas em minhas avaliação no letterboxd, filmes cuja a avaliação foi maior que 4, você quiser também replicar com suas avaliações e baixar os dados no site da letterboxd e trocar o arquivo data/raw/ratings.csv pelo seu próprio e então extrair as respectivas informações de seus logs

```
make extract-data
```

### Create Qdrant Collection

```
make create-qdrant-collections
```

### Run evaluations

With a new dataset will be necessary to recreate the ground truth data

```
make generate-evaluation-data
```

Actually, it is possible to simulate how much will cost to will to run this step with

```
make generate-evaluation-data-simulation
```

First, we can evaluate the retrieval

```
make generate-evaluation-data
```

After selected the retrieval, we can also run the evaluation for the the rag system. Detail for make sure the chat will correct even the answers about movies it does not know, for example, recent movies.

```
make rag-eval-simulation
make rag-eval
```

#### FastAPI

Open an API that's possible to include the requests and receive the model answers.

With the API open, you can also add the data with bash:
``` bash
curl -X POST "http://localhost:8000/rag/query"   -H "Content-Type: application/json"  -d '{"query": "i wanna a few comedy movies for quiet people", "top_k": 10}'
```

### Chat with Streamlit

Creates the UI interface where the user can make your solicitations and keeps the historic.

```
make chatbot
```

The chatbot interface includes a feedback interface that also register those feedbacks in order to "feed" the monitoring dashboard. The feedback is based in a positive/negative button and a optional comment.

Those data follows those formats

{
    "title": doc.get("title", "No title"),
    "year": doc.get("year", "unknown"),
    "origin": doc.get("origin", "unknown"),
    "director": doc.get("director", "unknown"),
    "cast": doc.get("cast", "unknown"),
    "genres": doc.get("genres", "unknown"),
    "similarity_score": doc.get("score", None),
    "content_preview": "Content used"
}

{
    "timestamp": datetime.datetime.now().isoformat(),
    "session_id": self.session_id,
    "user_query": user_query,
    "retrieved_documents": doc_metadata,
    "retrieved_count": len(retrieved_docs),
    "llm_response": llm_response,
    "response_time_seconds": round(response_time, 2),
    "context_used": [doc["title"] for doc in doc_metadata],
    "average_similarity": Average similarity
}

For feedback:

{
    "timestamp": datetime.datetime.now().isoformat(),
    "user_query": user_query,
    "llm_response": llm_response,
    "feedback": feedback,
    "comment": comment,
}

### Monitoring model

The Streamlit monitoring dashboard provides:

Time-response
Similarity between the query and the context
Movies/Genres retrieved
User feedback (thumbs up/down)

```
make monitor
```

### Deativate enviroment

```
docker-compose down -v
```

```
conda deactivate
```


## To-do list (next improvements)

* [ ] Bring more movies from Wiki - e os plots da wikipedia são bem grandes, né?
* [ ] Add ratings.csv and generate data to repo? and my logged tests?

## Acknowledgments

## Author

