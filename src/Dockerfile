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
COPY src/ .
RUN mkdir -p /app/logs/system

# Expor porta se for uma API
EXPOSE 6000

# Comando para rodar
CMD ["python", "rag_engine.py"]
