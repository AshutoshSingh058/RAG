# Kubernetes RAG Demo

A Streamlit and FastAPI RAG application for Kubernetes documentation.

## Demo

Example queries:

- How do Kubernetes CronJobs work?
- Create a Deployment with three replicas.
- How do Pods communicate through Services?
- What is the difference between a Job and a CronJob?
- How do Kubernetes liveness and readiness probes work?

## Architecture

- FastAPI: [app/main.py](app/main.py)
- Streamlit: [ui/app.py](ui/app.py)
- Embeddings: local `all-mpnet-base-v2`
- Vector database: Qdrant Cloud
- Models and fallbacks: Portkey
- Guardrails: NeMo Guardrails
- Reranking: local FlashRank

## Environment variables

Set these as secrets in Render and Streamlit Cloud:

```env
QDRANT_CLUSTER_ENDPOINT=
QDRANT_API_KEY=
PORTKEY_API_KEY=
PORTKEY_CONFIG=
PORTKEY_PRIMARY_SLUG=
PORTKEY_FALLBACK_SLUG=
LOGFIRE_TOKEN=
RESPONSE_MAX_TOKENS=2048
BACKEND_URL=
CORS_ORIGINS=
EMBEDDINGS_MODE=gemini
EMBEDDINGS_MODEL=all-mpnet-base-v2
EMBEDDINGS_DIM=768
```

Do not commit populated `.env` files.

For local development and ingestion, use `EMBEDDINGS_MODE=local`. On Render, use
`EMBEDDINGS_MODE=gemini` so the server does not load the local embedding model.

## Render dependency profile

The Dockerfile uses `requirements.txt`, a runtime-only dependency set.
Document parsing, local Sentence Transformers, FlashRank, and NeMo packages are
not installed on Render.

For local development, install the full set from `requirements-local.txt`:

```powershell
pip install -r requirements-local.txt
```

## Render backend

Render can deploy the included Dockerfile using [render.yaml](render.yaml).

Required environment variables:

- `QDRANT_CLUSTER_ENDPOINT`
- `QDRANT_API_KEY`
- `PORTKEY_API_KEY`
- `PORTKEY_CONFIG`
- `PORTKEY_PRIMARY_SLUG`
- `PORTKEY_FALLBACK_SLUG`
- `LOGFIRE_TOKEN`

Health endpoints:

- `GET /health`
- `GET /ready`

## Streamlit UI

Deploy [ui/app.py](ui/app.py) to Streamlit Community Cloud.

Set `BACKEND_URL` in Streamlit secrets:

```toml
BACKEND_URL = "https://your-render-service.onrender.com"
```

Set `CORS_ORIGINS` on Render if browser-based API calls are required:

```env
CORS_ORIGINS=https://your-app.streamlit.app
```