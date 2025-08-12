# === CONFIGS ===
VENV_NAME = chat-env
PYTHON = python
DATA_DIR = data
RAW_DATA = $(DATA_DIR)/raw/dataset.csv
PROCESSED_DATA = $(DATA_DIR)/processed/model_data.csv

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
	$(PYTHON) scr/model_pipeline/data_extractor.py

## Data Preparation
prepare-data: # $(RAW_DATA)
	$(PYTHON) scr/model_pipeline/data_preparation.py

## Feature Engineering
feat-eng: # $(RAW_DATA)
	$(PYTHON) scr/model_pipeline/feature_engineering.py

## Create target ans split data
target-split: # $(RAW_DATA)
	$(PYTHON) scr/model_pipeline/temporal_target_and_split.py --input-path $(PROCESSED_DATA) --target-col-source $(TARGET_COL_SOURCE) --horizon $(HORIZON) --split-data $(SPLIT_DATE)

## Model tuning
tune: # $(PROCESSED_DATA)
	$(PYTHON) scr/model_pipeline/catboost_optimization.py --split-data $(SPLIT_DATE)

## Final model
#eda:
#	$(PYTHON) scr/eda.py --data $(PROCESSED_DATA)
#	jupyter nbconvert notebooks/eda.ipynb --to html

## Final model
final-model:
	$(PYTHON) scr/model_pipeline/select_and_register_model.py

## Monitor
monitor:
	$(PYTHON) scr/monitoring/monitor.py --current-date $(SPLIT_DATE)

first-all-model-steps: extract-data prepare-data feat-eng target-split tune final-model
all-model-steps: extract-data prepare-data feat-eng target-split tune final-model monitor
