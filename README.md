# Movie Recommendation Chat

## Objective

The project includes:
*

## Tools used

* Conda - Virtual enviroment
* Pylint - Static code analyser
* pre-commit - pre-commit hooks

## Projet Struture

## How to execute locally

### Requirements

* Makefile
* Conda
* Docker

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
1. The OpenAI credentials can be view in https://platform.openai.com/, after create an account, search for "API Keys";

### Build Dockers

```
docker-compose down -v; docker-compose build; docker-compose up -d
```

#### Acess Points

| Service         | URL                        | Credentials                                          |
|-----------------|----------------------------|------------------------------------------------------|
| MLflow UI       | http://localhost:5000      | -                                                    |
| Adminer         | http://localhost:8080      |PostgreSQL/db/postgre/${POSTGRES_PASSWORD}/${DB_NAME} |
| Grafana         | http://localhost:3000      | admin/admin                                          |
| FastAPI         | http://localhost:8000/docs | -                                                    |

### Model

This single make command includes:

1. explain model

(?) EXPLAIN METRICS? Example in validation!

```
make -f Makefile all-model-steps
```

### Monitoring model

Creates a evidently html report about the current and previous predictions. It is also logged in MLFlow.
```
make -f Makefile monitor
```

### Open MLFlow

In order to analyze models runs and Evidently report, it is possible to open the MLFlow interface. Experiment ´<SELECT NAME>´ contains model runs and ´<SELECT NAME>_reports´ contains evidently reports.
```
mlflow server --backend-store-uri sqlite:///mlflow.db
```

### FastAPI

Open an API that's possible to include a csv file with the data PROCESSED and returns the predictions values. In `predict-csv` and `Try it out`

``` bash
uvicorn scr.api_csv:app --reload
```

With the API open, you can also add the data with bash:
``` bash
curl -X POST "http://localhost:8000/predict-csv"   -H "Content-Type: multipart/form-data"   -F "file=@data/processed/x_val.csv"
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
