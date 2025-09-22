FROM python:3.13-slim

WORKDIR /app


# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fonte
COPY src/ src/
COPY ui/ .
RUN mkdir -p /app/logs/system

# Expor porta se for uma API
EXPOSE 8501

# Comando para rodar
# streamlit run ui/chatbot.py -- --emb-model-name $(MODEL_NAME) --collection-name $(COLLECTION_NAME) --top-k $(TOP_K) --gpt-model-name $(GPT_MODEL_NAME)
CMD ["streamlit", "run", "chatbot.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
#, "--server.port", "8501", "--server.address", "0.0.0.0"]
