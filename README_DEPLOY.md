# Diabetes Risk Assessment System - Deployment Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     User-Facing Services                     │
├─────────────────────────────────────────────────────────────┤
│  Port 8501: Streamlit Frontend (Public)                     │
│  - Patient risk assessment form                              │
│  - Clinical decision support UI                              │
├─────────────────────────────────────────────────────────────┤
│  Port 8000: FastAPI Backend (API)                            │
│  - /predict endpoint with A/B testing                        │
│  - Health check and documentation                            │
├─────────────────────────────────────────────────────────────┤
│  Port 8502: Admin Dashboard (Internal Only)                  │
│  - Login-protected analytics                                 │
│  - Inference logs and A/B metrics                            │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Train Models

```powershell
# Train Logistic Regression (Variant A)
python logisticregression_only.py

# Train XGBoost (Variant B) - optional for A/B testing
python boostedtrees_ab.py
```

Expected outputs in `model_artifacts/`:
- `model_bundle.pkl` - Logistic regression model
- `boosted_model_bundle.pkl` - XGBoost model (if trained)
- `metrics.json` - Evaluation metrics

### 3. Start Services

```powershell
# Terminal 1: API Server
uvicorn app:app --host 0.0.0.0 --port 8000

# Terminal 2: Public Frontend
streamlit run streamlit_app.py --server.port 8501

# Terminal 3: Admin Dashboard (separate, internal only)
streamlit run admin_app.py --server.port 8502
```

## Service Details

### Public Frontend (Port 8501)

User-facing clinical assessment interface:
- **Risk Assessment Tab**: Input patient clinical data, get risk score
- **Clinical Information Tab**: Model methodology and factor descriptions

Access: `http://localhost:8501`

### API Backend (Port 8000)

RESTful inference API with A/B testing:
- `GET /health` - Service health check
- `POST /predict` - Risk prediction
- `GET /inference-logs` - Recent inference records
- `GET /analytics-summary` - Aggregated metrics

API Docs: `http://localhost:8000/docs`

### Admin Dashboard (Port 8502)

**Important**: This dashboard should NOT be exposed publicly. It contains sensitive analytics and requires authentication.

- Login-protected access
- Inference analytics and A/B test metrics
- CSV export of logs

Default credentials: `admin / admin12345`

Create new admin users:
```powershell
python create_admin_user.py
```

## Model Features

The model uses 10 clinical and lifestyle factors:

| Feature | Description | Values |
|---------|-------------|--------|
| GenHlth | General health status | 1=Excellent to 5=Poor |
| HighBP | High blood pressure | 0=No, 1=Yes |
| BMI | Body Mass Index | 10-80 |
| HighChol | High cholesterol | 0=No, 1=Yes |
| Age | Age category | 1=18-24 to 13=80+ |
| DiffWalk | Difficulty walking | 0=No, 1=Yes |
| HeartDiseaseorAttack | Heart disease/MI | 0=No, 1=Yes |
| PhysHlth | Poor physical health days | 0-30 |
| Education | Education level | 1-6 |
| PhysActivity | Physical activity | 0=No, 1=Yes |

## API Usage

### Predict Endpoint

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "GenHlth": 3,
    "HighBP": 1,
    "BMI": 30.0,
    "HighChol": 1,
    "Age": 9,
    "DiffWalk": 0,
    "HeartDiseaseorAttack": 0,
    "PhysHlth": 5,
    "Education": 4,
    "PhysActivity": 1
}'
```

Response:
```json
{
  "request_id": "uuid-here",
  "model_variant": "A",
  "model_name": "logistic_regression",
  "prediction": 1,
  "risk_category": "HIGH",
  "probability": 0.72,
  "threshold": 0.35
}
```

### A/B Testing

Include `user_id` for deterministic variant assignment:

```bash
curl -X POST "http://localhost:8000/predict?user_id=patient123&model_variant=auto" ...
```

Or force a specific variant:
```bash
curl -X POST "http://localhost:8000/predict?model_variant=B" ...
```

## Database Configuration

### Local Development (SQLite)

No configuration needed. Database created automatically at `data/inference_logs.db`.

### Production (PostgreSQL)

Set environment variable before starting services:

```powershell
$env:DATABASE_URL = "postgresql://user:password@host:5432/dbname"
```

## Deployment Options

### Streamlit Cloud (Frontend Only)

1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Set `streamlit_app.py` as main file
4. **Do not deploy `admin_app.py`** to public cloud

### Render / Railway (Full Stack)

1. Deploy API with `uvicorn app:app --host 0.0.0.0 --port $PORT`
2. Deploy Streamlit with `streamlit run streamlit_app.py --server.port $PORT`
3. Set `DATABASE_URL` environment variable for PostgreSQL

### Docker

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Security Notes

1. **Admin Dashboard**: Run on port 8502 internally only, never expose to public internet
2. **API Keys**: Consider adding authentication to `/inference-logs` and `/analytics-summary`
3. **Database**: Use PostgreSQL with SSL in production
4. **HTTPS**: Always use HTTPS in production deployments

## File Reference

| File | Purpose |
|------|---------|
| `logisticregression_only.py` | Logistic regression training with Optuna |
| `boostedtrees_ab.py` | XGBoost training for A/B testing |
| `app.py` | FastAPI inference API |
| `streamlit_app.py` | Public frontend |
| `admin_app.py` | Internal admin dashboard |
| `inference_db.py` | Database abstraction |
| `admin_auth.py` | Admin authentication |
| `create_admin_user.py` | Admin user management CLI |
