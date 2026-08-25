# Canonical runtime: Python 3.11, matching CI (.github/workflows/ci.yml) and the
# version the committed model artifacts were verified against.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install from the committed lock, not the loose ranges in requirements.txt, so
# the image gets scikit-learn 1.8.0 / xgboost 3.0.4 and loads model_artifacts/
# without serialization-version warnings. Dev and test dependencies are
# deliberately not installed into the runtime image.
COPY requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.lock.txt

COPY . .

# Default: API service. Override CMD for Streamlit/Admin via render.yaml.
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
