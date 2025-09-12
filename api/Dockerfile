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
COPY api/ .
RUN mkdir -p /app/logs/system

# Expor porta se for uma API
EXPOSE 8000

# Comando para rodar
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
