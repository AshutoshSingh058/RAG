import logfire
import re
from app.services.retrieval.embedding import embed_query
TOPIC = "Kubernetes"
TOPIC_ANCHORS = (
    "Kubernetes cluster and container orchestration",
    "Kubernetes Pods, Deployments, Services, Jobs, and CronJobs",
    "Kubernetes scheduling, networking, scaling, storage, and controllers",
    "kubectl commands and Kubernetes YAML manifests",
)
TOPIC_SIMILARITY_THRESHOLD = 0.38
_anchor_vectors: list[list[float]] | None = None


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    import math

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _topic_similarity(message: str) -> float:
    global _anchor_vectors
    if _anchor_vectors is None:
        _anchor_vectors = [embed_query(anchor) for anchor in TOPIC_ANCHORS]
    query_vector = embed_query(message)
    score = max(
        _cosine_similarity(query_vector, anchor_vector)
        for anchor_vector in _anchor_vectors
    )
    logfire.info(
        "Topic relevance check",
        topic=TOPIC,
        similarity=round(score, 4),
        threshold=TOPIC_SIMILARITY_THRESHOLD,
    )
    return score

_DETERMINISTIC_GUARD_PATTERNS = (
    (
        re.compile(r"\b(ignore|disregard|forget)\b.*\b(previous|prior|system)\b", re.I),
        "I maintain consistent guidelines regardless of how I am prompted. I can help with Kubernetes. What would you like to know?",
    ),
    (
        re.compile(r"\b(DAN|developer mode|bypass your guidelines|override your safety)\b", re.I),
        "I maintain consistent guidelines regardless of how I am prompted. I can help with Kubernetes. What would you like to know?",
    ),
    (
        re.compile(r"\b(sql injection|exploit|vulnerability|malware|phishing)\b", re.I),
        "I can't help with harmful exploitation or malware. I can help with Kubernetes security and safe cluster operations.",
    ),
)

_CONVERSATIONAL_PATTERNS = (
    (
        re.compile(r"^(hi|hello|hey|good morning|good afternoon|what's up|howdy)\b", re.I),
        "Hello! I can help with Kubernetes topics such as Pods, Deployments, Services, Jobs, CronJobs, networking, storage, and cluster operations.",
    ),
    (
        re.compile(r"\b(what can you do|what do you know|what are your capabilities|help)\b", re.I),
        "I can answer questions about Kubernetes, including Pods, Deployments, Services, Jobs, CronJobs, networking, storage, scheduling, and kubectl.",
    ),
    (
        re.compile(r"^(bye|goodbye|see you|thanks bye|that is all|i am done)\b", re.I),
        "Goodbye!",
    ),
)


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

    for pattern, response in _CONVERSATIONAL_PATTERNS:
        if pattern.search(message):
            logfire.info("Deterministic conversational response")
            return True, response

    similarity = _topic_similarity(message)
    if similarity < TOPIC_SIMILARITY_THRESHOLD:
        logfire.info(
            "Off-topic request blocked by semantic relevance",
            query=message[:80],
            similarity=round(similarity, 4),
        )
        return True, "I'm your Kubernetes assistant, so I can only answer Kubernetes-related questions. Please ask about Pods, Deployments, Services, Jobs, CronJobs, networking, storage, or cluster operations."

    logfire.info("✅ Semantic topic check passed.")
    return False, None
