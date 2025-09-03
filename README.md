# Movie Recommendation Chat

## Objective

The project includes:
*

## Tools used

* Conda - Virtual enviroment
* Pylint - Static code analyser
* pre-commit - pre-commit hooks
* Qdrant - embedding?

## Projet Struture

## How to execute locally

### Requirements

* Makefile
* Conda
* Docker
* Qdrant
* OpenAI API
* FastAPI
* Streamlit

### Setup project with Makefile

1. Create and activate enviroment
```
make -f Makefile setup
```

2. Activate enviroment
```
source ~/.bashrc && conda activate chat-env
```

3. Install dependencies and pre-commit
```
make -f Makefile install
```

### Add your Credentials

Create your own .env based on .env.example
```
cp .env.example .env
```

And change the default values to your needs:
1. The Kaggle credentials can be view in "Settings" > Down until "API" > "Create New API Token" and the file kaggle.json will be downloaded with the credentials;
1. The OpenAI credentials can be view in https://platform.openai.com/, after create an account, search for "API Keys";

### Build Dockers



Ou seja, tenho que adicionar no docker
```
docker run -p 6333:6333 qdrant/qdrant
```

```
docker-compose down -v; docker-compose build; docker-compose up -d
```

#### Access Points

| Service         | URL                        | Credentials                                          |
|-----------------|----------------------------|------------------------------------------------------|
| MLflow UI       | http://localhost:5000      | -                                                    |
| Qdrant          | http://localhost:6333      | -                                                    |
| Adminer         | http://localhost:8080      |PostgreSQL/db/postgre/${POSTGRES_PASSWORD}/${DB_NAME} |
| Grafana         | http://localhost:3000      | admin/admin                                          |
| FastAPI         | http://localhost:8000/docs | -                                                    |
| Chat            | http://localhost:8501      | -                                                    |

### Model

This single make command includes:

1. explain model

(?) EXPLAIN METRICS? Example in validation!

```
make -f Makefile all-model-steps
```


### Open MLFlow

In order to analyze models runs and Evidently report, it is possible to open the MLFlow interface. Experiment ´<SELECT NAME>´ contains model runs and ´<SELECT NAME>_reports´ contains evidently reports.
```
mlflow server --backend-store-uri sqlite:///mlflow.db
```

### FastAPI

Open an API that's possible to include the requests and receive the model answers.

``` bash
uvicorn src.api:app --reload
```

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


```
make -f Makefile monitor
```

### Deploy in Docker

1. Build image
```
docker build -t <SELECT NAME>:latest .
```

2. Run model in a container
```
docker run -p 8080:8080 <SELECT NAME>:latest
```


### Deativate enviroment
```
conda deactivate
```

## To-do list (next improvements)

* no Retrieval evaluation eu deveria comparar diferentes modelos?
* Como gerar os dados de base de comparação???
* se eu passar uma solicitação do tipo quero um filme igual a esse, acho que não vai dar certo...
* [ ] Bring more movies from Wiki - e os plots da wikipedia são bem grandes, né?
* [ ] Add ratings.csv and generate data to repo? and my logged tests?
