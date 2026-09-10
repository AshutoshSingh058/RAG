import logfire
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI

from app.config import settings


MODEL_BY_FEATURE = {
    "guardrails": settings.GUARDRAILS_MODEL,
    "planner": settings.PLANNER_MODEL,
    "chat": settings.CHAT_MODEL,
    "eval": settings.EVAL_MODEL,
}


def get_model_candidates(feature: str) -> list[str]:
    """Return the configured primary model followed by ordered fallbacks."""
    primary = MODEL_BY_FEATURE.get(feature, settings.CHAT_MODEL)
    candidates = [primary, *settings.MODEL_FALLBACKS.get(feature, [])]
    return list(dict.fromkeys(candidates))


portkey_client = Portkey(
    api_key=settings.PORTKEY_API_KEY,
    config=settings.PORTKEY_CONFIG,
    metadata={
        "feature": "rag-system",
        "_user": "rag-system",
        "environment": "production"
    }
)

def get_langchain_llm(feature: str = "chat") -> ChatOpenAI:
    clients = []
    for model in get_model_candidates(feature):
        clients.append(
            ChatOpenAI(
                api_key=settings.PORTKEY_API_KEY,
                base_url=PORTKEY_GATEWAY_URL,
                model=model,
                temperature=0,
                default_headers=createHeaders(
                    api_key=settings.PORTKEY_API_KEY,
                    config=settings.PORTKEY_CONFIG,
                    metadata={
                        "feature": feature,
                        "_user": "rag-system",
                        "environment": "production",
                    },
                ),
            )
        )

    primary, *fallbacks = clients
    if fallbacks:
        logfire.info(
            "Configured Portkey model fallbacks",
            feature=feature,
            models=get_model_candidates(feature),
        )
        return primary.with_fallbacks(fallbacks)
    return primary

def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the Portkey native client response headers.
    Tries multiple attribute paths defensively — returns 'MISS' if not found.
    """
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"