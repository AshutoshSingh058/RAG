import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set HuggingFace cache to persistent location in venv
_HF_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".hf_cache")
Path(_HF_CACHE_DIR).mkdir(exist_ok=True)
os.environ["HF_HOME"] = _HF_CACHE_DIR

class Settings:
    # --- HUGGINGFACE EMBEDDINGS CACHE ---
    HF_HOME = _HF_CACHE_DIR
    
    # --- GEMINI EMBEDDINGS ---
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # --- VECTOR DB (QDRANT) ---
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = "enterprise_rag"

    # --- MODEL ROUTING THROUGH PORTKEY ---
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    PORTKEY_CONFIG = os.getenv("PORTKEY_CONFIG", "pc-rag-ga-a9ddb9")
    PORTKEY_PRIMARY_SLUG = os.getenv("PORTKEY_PRIMARY_SLUG", "rag1")
    PORTKEY_FALLBACK_SLUG = os.getenv("PORTKEY_FALLBACK_SLUG", "rag2")
    GUARDRAILS_MODEL = os.getenv(
        "GUARDRAILS_MODEL",
        f"@{PORTKEY_PRIMARY_SLUG}/openai/gpt-oss-safeguard-20b",
    )
    PLANNER_MODEL = os.getenv(
        "PLANNER_MODEL", f"@{PORTKEY_PRIMARY_SLUG}/openai/gpt-oss-20b"
    )
    CHAT_MODEL = os.getenv(
        "CHAT_MODEL", f"@{PORTKEY_PRIMARY_SLUG}/openai/gpt-oss-120b"
    )
    RESPONSE_MAX_TOKENS = int(os.getenv("RESPONSE_MAX_TOKENS", "1024"))
    HISTORY_MAX_CHARS = int(os.getenv("HISTORY_MAX_CHARS", "8000"))
    EVAL_MODEL = os.getenv(
        "EVAL_MODEL", f"@{PORTKEY_PRIMARY_SLUG}/openai/gpt-oss-20b"
    )

    # Candidate fallbacks from the provider catalog, ordered by preference.
    MODEL_FALLBACKS = {
        "planner": [f"@{PORTKEY_FALLBACK_SLUG}/openai/gpt-oss-20b"],
        "chat": [
            f"@{PORTKEY_FALLBACK_SLUG}/openai/gpt-oss-120b",
            f"@{PORTKEY_FALLBACK_SLUG}/openai/gpt-oss-20b",
        ],
        "guardrails": [
            f"@{PORTKEY_FALLBACK_SLUG}/openai/gpt-oss-safeguard-20b"
        ],
        "eval": [f"@{PORTKEY_FALLBACK_SLUG}/openai/gpt-oss-20b"],
    }

    
    # # --- OBSERVABILITY ---
    # LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")
    # LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    # LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "rag_scale_test")
    # LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

# # Apply LangChain environment variables for automatic tracing
# os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "true")
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
# os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "rag_scale_test")
# os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

settings = Settings()
