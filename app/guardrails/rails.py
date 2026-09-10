import logfire
import re
from nemoguardrails import RailsConfig, LLMRails

from app.gateway import get_langchain_llm
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS


_rails: LLMRails | None = None

_DETERMINISTIC_GUARD_PATTERNS = (
    (
        re.compile(r"\b(ignore|disregard|forget)\b.*\b(previous|prior|system)\b", re.I),
        "I maintain consistent guidelines regardless of how I am prompted. I am here to help with Kubernetes, Intel, and networking. What can I help you with?",
    ),
    (
        re.compile(r"\b(DAN|developer mode|bypass your guidelines|override your safety)\b", re.I),
        "I maintain consistent guidelines regardless of how I am prompted. I am here to help with Kubernetes, Intel, and networking. What can I help you with?",
    ),
    (
        re.compile(r"\b(sql injection|exploit|vulnerability|malware|phishing)\b", re.I),
        "I'm an Enterprise IT Assistant focused on Kubernetes, Intel hardware, and networking. I can't help with that - but ask me anything technical!",
    ),
    (
        re.compile(r"\b(tell|write)\b.*\b(joke|poem)\b", re.I),
        "I'm an Enterprise IT Assistant focused on Kubernetes, Intel hardware, and networking. I can't help with that - but ask me anything technical!",
    ),
)


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Uses the Portkey-backed LLM for intent classification at the gate, keeping
    authentication and model routing consistent with the RAG pipeline.
    """
    global _rails

    guard_llm = get_langchain_llm(feature="guardrails")
        
    logfire.info(
        f"Guardrails LLM: "
        f"{guard_llm.model_name if hasattr(guard_llm, 'model_name') else guard_llm}"
    )
    
    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT
    )
    _rails = LLMRails(config, llm=guard_llm)
    logfire.info("🛡️ NeMo Guardrails initialised via Portkey gateway.")
    
    


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    Returns:
        (True,  rail_response) — a rail fired; return this response immediately,
                                skip the RAG pipeline entirely.
        (False, None)          — message is clean; proceed to LangGraph.
    """
    for pattern, response in _DETERMINISTIC_GUARD_PATTERNS:
        if pattern.search(message):
            logfire.info(f"🛡️ Deterministic guardrail fired | query='{message[:80]}'")
            return True, response

    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, None

    with logfire.span("🛡️ Guardrails Check"):
        result = _rails.generate(messages=[{"role": "user", "content": message}])

        # NeMo returns {'role': 'assistant', 'content': '...'} — extract text
        content = result.get("content", "") if isinstance(result, dict) else str(result)

        fired = any(indicator in content for indicator in RAIL_INDICATORS)

        if fired:
            logfire.info(f"🛡️ Guardrails fired | query='{message[:80]}'")
            return True, content

        logfire.info("✅ Guardrails passed.")
        return False, None
