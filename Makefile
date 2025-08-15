# === CONFIGS ===
VENV_NAME = chat-env
PYTHON = python
DATA_DIR = data
RAW_DATA = $(DATA_DIR)/raw#/dataset.csv
PROCESSED_DATA = $(DATA_DIR)/processed/model_data.csv

COLLECTION_NAME = chosen_movies2
EMBEDDING_DIMENSIONALITY = 384
MODEL_NAME = BAAI/bge-small-en
TOP_K = 10
QUERY = "a non-american romantic movie"

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
	$(PYTHON) scr/data_extractor.py

## Create Qdrant Collection
create-qdrant-collection:
	$(PYTHON) scr/create_qdrant_collection.py --collection-name $(COLLECTION_NAME) --embedding-dimensionality $(EMBEDDING_DIMENSIONALITY) --model-name $(MODEL_NAME) --path-source $(RAW_DATA)

## Qdrant Search
qdrant-search:
	$(PYTHON) scr/qdrant_search.py --model-name $(MODEL_NAME) --collection-name $(COLLECTION_NAME) --top-k $(TOP_K) --query $(QUERY)

## Data Preparation
prepare-data: # $(RAW_DATA)
	$(PYTHON) scr/data_preparation.py

## Feature Engineering
feat-eng: # $(RAW_DATA)
	$(PYTHON) scr/feature_engineering.py

## Create target ans split data
target-split: # $(RAW_DATA)
	$(PYTHON) scr/temporal_target_and_split.py --input-path $(PROCESSED_DATA) --target-col-source $(TARGET_COL_SOURCE) --horizon $(HORIZON) --split-data $(SPLIT_DATE)

## Model tuning
tune: # $(PROCESSED_DATA)
	$(PYTHON) scr/catboost_optimization.py --split-data $(SPLIT_DATE)

## Final model
#eda:
#	$(PYTHON) scr/eda.py --data $(PROCESSED_DATA)
#	jupyter nbconvert notebooks/eda.ipynb --to html

## Final model
final-model:
	$(PYTHON) scr/select_and_register_model.py

## Monitor
monitor:
	$(PYTHON) scr/monitoring/monitor.py --current-date $(SPLIT_DATE)

first-all-model-steps: extract-data prepare-data feat-eng target-split tune final-model
all-model-steps: extract-data prepare-data feat-eng target-split tune final-model monitor
