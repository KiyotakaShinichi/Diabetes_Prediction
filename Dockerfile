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
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

COPY . .

# Default: API service. Override CMD for Streamlit/Admin via render.yaml.
EXPOSE 8000

# Container-level liveness, using the API's own /health route.
#
# Python's stdlib rather than curl: the slim base image ships no curl, and
# adding one purely for a probe would enlarge the image and its vulnerability
# surface for no runtime benefit. urllib is already present and needs no
# network beyond loopback.
#
# start-period covers first-request bundle deserialization on a small instance,
# so a cold start is not mistaken for a failure. The probe deliberately targets
# /health (liveness: is the process answering) rather than /ready (can it serve
# predictions) - an unready instance should stop receiving traffic, not be
# killed and restarted into the same state. CI probes both.
#
# This adds no endpoint: /health is defined once, in app.py.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
