FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn

COPY . /app

EXPOSE 8080
# Vertex AI sends predictions to port 8080 by default
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
