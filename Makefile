# === CONFIGS ===
VENV_NAME = chat-env
PYTHON = python
DATA_DIR = data
RAW_DATA = $(DATA_DIR)/raw#/dataset.csv # será que é tão necessário assim?
EVAL_DATA = $(DATA_DIR)/processed/ground-truth-retrieval.csv

COLLECTION_NAME = movies
EMBEDDING_DIMENSIONALITY = 384
MODEL_NAME = BAAI/bge-small-en
TOP_K = 10
QUERY = "why is submarine a good movie?" # "tell me a non-american comedy movie" # "i want to watch a movie similar to submarine"
N_TESTS = 3
GPT_MODEL_NAME = "gpt-4o-mini"

# === COMANDS ===
## Create virtual environment
setup:
	conda create -n $(VENV_NAME) python=3.13 -y && conda init
	@echo "Active environment: conda activate $(VENV_NAME)"

## Install dependencies
install:
	pip install -r requirements.txt
	pre-commit install

## Extract Data
extract-data:
	$(PYTHON) src/data_extractor.py

## Create Qdrant Collection
create-qdrant-collection:
	$(PYTHON) src/create_qdrant_collection.py --collection-name $(COLLECTION_NAME) --embedding-dimensionality $(EMBEDDING_DIMENSIONALITY) --model-name $(MODEL_NAME) --path-source $(RAW_DATA)

## Data Preparation
generate-evaluation-data: # $(RAW_DATA)
	$(PYTHON) -m evaluation.generate_evaluation_data --path-source $(DATA_DIR) --n-tests $(N_TESTS) --gpt-model-name $(GPT_MODEL_NAME)


## Retrieval Evaluation
retrieval_eval:
	$(PYTHON) -m evaluation.retrieval_eval --ground-truth-path $(EVAL_DATA) --emb-model-name $(MODEL_NAME) --collection-name $(COLLECTION_NAME) --top-k $(TOP_K)

## RAG Evaluation
rag_eval:
	$(PYTHON) -m evaluation.rag_eval --ground-truth-path $(EVAL_DATA) --emb-model-name $(MODEL_NAME) --collection-name $(COLLECTION_NAME) --top-k $(TOP_K) --gpt-model-name $(GPT_MODEL_NAME)

## Rag
rag:
	$(PYTHON) -m evaluation.rag --emb-model-name $(MODEL_NAME) --collection-name $(COLLECTION_NAME) --top-k $(TOP_K) --query $(QUERY)

## ChatBot
chatbot:
	streamlit run ui/chatbot.py -- --emb-model-name $(MODEL_NAME) --collection-name $(COLLECTION_NAME) --top-k $(TOP_K) --gpt-model-name $(GPT_MODEL_NAME)

## Create target ans split data
target-split: # $(RAW_DATA)
	$(PYTHON) src/temporal_target_and_split.py --input-path $(PROCESSED_DATA) --target-col-source $(TARGET_COL_SOURCE) --horizon $(HORIZON) --split-data $(SPLIT_DATE)

## Model tuning
tune: # $(PROCESSED_DATA)
	$(PYTHON) src/catboost_optimization.py --split-data $(SPLIT_DATE)

## Final model
#eda:
#	$(PYTHON) src/eda.py --data $(PROCESSED_DATA)
#	jupyter nbconvert notebooks/eda.ipynb --to html

## Final model
final-model:
	$(PYTHON) src/select_and_register_model.py

## Monitor
monitor:
	$(PYTHON) src/monitoring/monitor.py --current-date $(SPLIT_DATE)

first-all-model-steps: extract-data prepare-data feat-eng target-split tune final-model
all-model-steps: extract-data prepare-data feat-eng target-split tune final-model monitor
