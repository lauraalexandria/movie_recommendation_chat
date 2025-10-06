FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY monitoring/ .
RUN mkdir -p /app/logs/system

EXPOSE 8502

CMD ["streamlit", "run", "dashboard.py", "--server.port", "8502", "--server.address", "0.0.0.0"]
