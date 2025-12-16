FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

# Environment for MLflow model resolution (can be overridden)
ENV MLFLOW_TRACKING_URI=file:./mlruns
ENV MODEL_NAME=CreditRiskProxyModel
ENV MODEL_STAGE=Production

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
