# === CONFIGS ===
VENV_NAME = chat-env
PYTHON = python
DATA_DIR = data
RAW_DATA = $(DATA_DIR)/raw
EVAL_DATA = $(DATA_DIR)/processed/ground-truth-retrieval.csv

COLLECTION_NAME = movies
EMBEDDING_DIMENSIONALITY = 512
EMB_MODEL_NAME = jinaai/jina-embeddings-v2-small-en
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
create-qdrant-collections:
	$(PYTHON) src/create_qdrant_collections.py --collection-name $(COLLECTION_NAME) --embedding-dimensionality $(EMBEDDING_DIMENSIONALITY) --model-name $(EMB_MODEL_NAME) --path-source $(RAW_DATA)

## Data Preparation
generate-evaluation-data-simulation:
	$(PYTHON) -m evaluation.generate_evaluation_data --path-source $(DATA_DIR) --n-tests $(N_TESTS) --gpt-model-name $(GPT_MODEL_NAME) --flag-simulate-cost True

## Data Preparation
generate-evaluation-data:
	$(PYTHON) -m evaluation.generate_evaluation_data --path-source $(DATA_DIR) --n-tests $(N_TESTS) --gpt-model-name $(GPT_MODEL_NAME) --flag-simulate-cost False

## Retrieval Evaluation
retrieval-eval:
	$(PYTHON) -m evaluation.retrieval_eval --ground-truth-path $(EVAL_DATA) --emb-model-name $(EMB_MODEL_NAME) --collection-name $(COLLECTION_NAME) --top-k $(TOP_K)

## Simulation RAG Evaluation
rag-eval-simulation:
	$(PYTHON) -m evaluation.rag_eval --ground-truth-path $(EVAL_DATA) --emb-model-name $(EMB_MODEL_NAME) --collection-name $(COLLECTION_NAME) --top-k $(TOP_K) --gpt-model-name $(GPT_MODEL_NAME) --flag-simulate-cost True

## RAG Evaluation
rag-eval:
	$(PYTHON) -m evaluation.rag_eval --ground-truth-path $(EVAL_DATA) --emb-model-name $(EMB_MODEL_NAME) --collection-name $(COLLECTION_NAME) --top-k $(TOP_K) --gpt-model-name $(GPT_MODEL_NAME) --flag-simulate-cost False

## Rag
rag:
	$(PYTHON) -m evaluation.rag --emb-model-name $(EMB_MODEL_NAME) --collection-name $(COLLECTION_NAME) --top-k $(TOP_K) --query $(QUERY)

## ChatBot
chatbot:
	streamlit run ui/chatbot.py --emb-model-name $(EMB_MODEL_NAME) --collection-name $(COLLECTION_NAME) --top-k $(TOP_K) --gpt-model-name $(GPT_MODEL_NAME)

## Monitor
monitor:
	streamlit run monitoring/dashboard.py

## Complete Chat Flow
complete-chat-flow: extract-data create-qdrant-collections chatbot monitor
