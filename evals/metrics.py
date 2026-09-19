"""
Phase 2 — RAGAS metrics.
All LLM-based metrics use the Portkey eval route and its configured model fallbacks.
Metrics run one sample at a time with cooldowns to reduce provider rate-limit pressure.
Contexts are truncated to 300 chars (2 chunks max) so no single request exceeds the limit.
"""


import os
import asyncio
import json
import logfire
import pandas as pd
from datetime import datetime, timezone
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders

from app.config import settings
from app.gateway import get_model_candidates

from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas import SingleTurnSample
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
)

# HuggingFace cache directory (set in app.config)
_HF_CACHE_DIR = os.environ.get("HF_HOME")

JUDGE_MODEL = settings.EVAL_MODEL
JUDGE_RETRY_STATUS_CODES = {404, 408, 429, 500, 502, 503, 504}
COOLDOWN_STANDARD = 62
COOLDOWN_MINI = 40       # between individual samples — lets sliding TPM window recover (~2,800 tok/sample)
GENERAL_BATCH_SIZE = 1  # one sample at a time: abatch_score fires calls concurrently per sample,
                         # so batch>1 stacks multiple samples' async calls inside the same second
JUDGE_ATTEMPTS = 3
JUDGE_RETRY_DELAY = 5
METRICS_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "metrics_results.json")
METRICS_VERSIONS_DIR = os.path.join(os.path.dirname(__file__), "metrics_results_versions")


class _CachedScore:
    def __init__(self, value: float):
        self.value = value


def load_metric_results(path: str = METRICS_RESULTS_PATH, sample_count: int | None = None) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as file:
            checkpoint = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}
    if sample_count is not None and checkpoint.get("sample_count") != sample_count:
        return {}
    results = checkpoint.get("results", {})
    results.pop("tool_correctness", None)
    return results


def _save_metric_results(results: dict, sample_count: int, path: str = METRICS_RESULTS_PATH) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump({"sample_count": sample_count, "results": results}, file, indent=2)


def _save_metric_version(results: dict, sample_count: int) -> str:
    os.makedirs(METRICS_VERSIONS_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version_path = os.path.join(
        METRICS_VERSIONS_DIR,
        f"metrics_results_{timestamp}.json",
    )
    with open(version_path, "w", encoding="utf-8") as file:
        json.dump({"sample_count": sample_count, "results": results}, file, indent=2)
    return version_path


class _FallbackCompletions:
    def __init__(
        self,
        primary: AsyncOpenAI,
        fallback: AsyncOpenAI | None,
        fallback_model: str | None,
    ):
        self.primary = primary
        self.fallback = fallback
        self.fallback_model = fallback_model

    async def create(self, **kwargs):
        try:
            return await self.primary.chat.completions.create(**kwargs)
        except Exception as error:
            status_code = getattr(error, "status_code", None)
            retryable = (
                status_code in JUDGE_RETRY_STATUS_CODES
                or isinstance(error, (APIConnectionError, APITimeoutError))
            )
            if self.fallback is None or not retryable:
                raise

            logfire.warning(
                "Eval judge primary failed; using Portkey fallback model",
                status_code=status_code,
                model=kwargs.get("model", JUDGE_MODEL),
            )
            fallback_kwargs = {**kwargs, "model": self.fallback_model}
            return await self.fallback.chat.completions.create(**fallback_kwargs)


class _FallbackChat:
    def __init__(
        self,
        primary: AsyncOpenAI,
        fallback: AsyncOpenAI | None,
        fallback_model: str | None,
    ):
        self.completions = _FallbackCompletions(primary, fallback, fallback_model)


class _FallbackPortkeyClient(AsyncOpenAI):
    def __init__(
        self,
        primary: AsyncOpenAI,
        fallback: AsyncOpenAI | None,
        fallback_model: str | None,
    ):
        super().__init__(
            api_key=settings.PORTKEY_API_KEY,
            base_url=PORTKEY_GATEWAY_URL,
            default_headers=createHeaders(
                api_key=settings.PORTKEY_API_KEY,
                config=settings.PORTKEY_CONFIG,
                metadata={
                    "feature": "eval",
                    "_user": "rag-system",
                    "environment": "production",
                },
            ),
        )
        self.chat = _FallbackChat(primary, fallback, fallback_model)


def _build_judge():
    models = get_model_candidates("eval")

    def make_client() -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=settings.PORTKEY_API_KEY,
            base_url=PORTKEY_GATEWAY_URL,
            default_headers=createHeaders(
                api_key=settings.PORTKEY_API_KEY,
                config=settings.PORTKEY_CONFIG,
                metadata={
                    "feature": "eval",
                    "_user": "rag-system",
                    "environment": "production",
                },
            ),
        )

    primary = make_client()
    fallback = make_client() if len(models) > 1 else None
    client = _FallbackPortkeyClient(
        primary,
        fallback,
        models[1] if len(models) > 1 else None,
    )
    llm = llm_factory(
        models[0],
        provider="openai",
        client=client,
        temperature=0,
        max_tokens=4096,
    )
    embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        use_api=False,
        cache_folder=_HF_CACHE_DIR,  # Persist to venv .hf_cache/ directory
    )
    return llm, embeddings

async def _cooldown(seconds: int, label: str, status_cb=None):
    msg = f"⏳ {seconds}s cooldown after {label} (provider rate-limit buffer)..."
    if status_cb:
        status_cb(msg)
    for _ in range(seconds // 10):
        await asyncio.sleep(10)
    if status_cb:
        status_cb(f"✅ Ready — starting next experiment.")
        
        
def _prep_samples(golden_dataset: dict) -> list:
    """
    Returns only samples with actual_response populated.
    Preserves the complete generated response and all retrieved contexts for scoring.
    """
    valid = []
    for s in golden_dataset["rag_samples"]:
        response = s.get("actual_response", "").strip()
        if not response:
            continue
        valid.append({**s, "actual_contexts": s.get("actual_contexts") or []})
    return valid


def _score_df(metric_key: str, samples: list, scores) -> pd.DataFrame:
    return pd.DataFrame([
        {"question": s["question"][:65], metric_key: round(float(r.value), 3)}
        for s, r in zip(samples, scores)
    ])


async def _batched_score(
    metric,
    inputs: list,
    samples: list,
    status_cb=None,
    label: str = "",
    checkpoint: dict | None = None,
    checkpoint_key: str | None = None,
) -> list:
    """
    Runs abatch_score in chunks of GENERAL_BATCH_SIZE with cooldowns between chunks.
    Keeps each burst small enough for the configured provider limits.
    """
    saved_key = checkpoint_key or label
    if checkpoint and saved_key in checkpoint:
        return [_CachedScore(value) for value in checkpoint[saved_key]]

    all_scores = []
    batches = [inputs[i : i + GENERAL_BATCH_SIZE] for i in range(0, len(inputs), GENERAL_BATCH_SIZE)]
    for b_idx, batch in enumerate(batches):
        if b_idx > 0:
            await _cooldown(COOLDOWN_MINI, f"{label} batch {b_idx}", status_cb)
        for attempt in range(1, JUDGE_ATTEMPTS + 1):
            try:
                scores = await metric.abatch_score(batch)
                break
            except Exception as error:
                logfire.warning(
                    "Eval judge attempt failed; retrying",
                    metric=label,
                    batch=b_idx + 1,
                    attempt=attempt,
                    error_type=type(error).__name__,
                    error=str(error)[:300],
                )
                if attempt == JUDGE_ATTEMPTS:
                    raise
                if status_cb:
                    status_cb(
                        f"⚠️ {label} batch {b_idx + 1} failed; retrying "
                        f"({attempt + 1}/{JUDGE_ATTEMPTS})..."
                    )
                await asyncio.sleep(JUDGE_RETRY_DELAY)
        all_scores.extend(scores)
    return all_scores

async def run_all_metrics(
    golden_dataset: dict,
    status_cb=None,
    sample_limit: int | None = None,
    checkpoint_path: str = METRICS_RESULTS_PATH,
) -> dict:
    """
    Runs all 5 RAGAS experiments. Returns dict keyed by metric name → DataFrame.
    status_cb(message: str) is called for live UI updates.
    """
    judge_llm, ragas_embeddings = _build_judge()
    samples = _prep_samples(golden_dataset)
    if sample_limit is not None:
        samples = samples[:sample_limit]

    if not samples:
        raise ValueError("No samples with actual_response found. Run Phase 1 first.")

    checkpoint = load_metric_results(checkpoint_path, len(samples))
    results = {}

    def save_metric(key: str, frame: pd.DataFrame) -> None:
        results[key] = frame
        checkpoint[key] = frame.iloc[:, 1].tolist()
        _save_metric_results(checkpoint, len(samples), checkpoint_path)

    with logfire.span("🧪 Eval Phase 2 — All Metrics", total_samples=len(samples)):

        # ── Exp 1: Faithfulness ───────────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 1/6 — Faithfulness ({len(samples)} samples)...")
        with logfire.span("🧪 Exp 1 — Faithfulness"):
            inputs = [
                {
                    "user_input": s["question"],
                    "response": s["actual_response"],
                    "retrieved_contexts": s["actual_contexts"],
                }
                for s in samples
            ]
            scores = await _batched_score(
                Faithfulness(llm=judge_llm),
                inputs,
                samples,
                status_cb,
                "Faithfulness",
                checkpoint,
                checkpoint_key="faithfulness",
            )
            df = _score_df("faithfulness", samples, scores)
            save_metric("faithfulness", df)
            logfire.info("🧪 Faithfulness done", avg=round(df["faithfulness"].mean(), 3))

        await _cooldown(COOLDOWN_STANDARD, "Faithfulness", status_cb)

        # ── Exp 2: Answer Relevancy ───────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 2/6 — Answer Relevancy ({len(samples)} samples)...")
        with logfire.span("🧪 Exp 2 — Answer Relevancy"):
            inputs = [
                {"user_input": s["question"], "response": s["actual_response"]}
                for s in samples
            ]
            scores = await _batched_score(
                AnswerRelevancy(llm=judge_llm, embeddings=ragas_embeddings),
                inputs,
                samples,
                status_cb,
                "Answer Relevancy",
                checkpoint,
                checkpoint_key="answer_relevancy",
            )
            df = _score_df("answer_relevancy", samples, scores)
            save_metric("answer_relevancy", df)
            logfire.info("🧪 Answer Relevancy done", avg=round(df["answer_relevancy"].mean(), 3))

        await _cooldown(COOLDOWN_STANDARD, "Answer Relevancy", status_cb)

        # ── Exp 3: Context Precision ──────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 3/6 — Context Precision ({len(samples)} samples)...")
        with logfire.span("🧪 Exp 3 — Context Precision"):
            inputs = [
                {
                    "user_input": s["question"],
                    "reference": s["reference"],
                    "retrieved_contexts": s["actual_contexts"],
                }
                for s in samples
            ]
            scores = await _batched_score(
                ContextPrecision(llm=judge_llm),
                inputs,
                samples,
                status_cb,
                "Context Precision",
                checkpoint,
                checkpoint_key="context_precision",
            )
            df = _score_df("context_precision", samples, scores)
            save_metric("context_precision", df)
            logfire.info("🧪 Context Precision done", avg=round(df["context_precision"].mean(), 3))

        await _cooldown(COOLDOWN_STANDARD, "Context Precision", status_cb)

        # ── Exp 4: Context Recall ─────────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 4/6 — Context Recall ({len(samples)} samples)...")
        with logfire.span("🧪 Exp 4 — Context Recall"):
            inputs = [
                {
                    "user_input": s["question"],
                    "reference": s["reference"],
                    "retrieved_contexts": s["actual_contexts"],
                }
                for s in samples
            ]
            scores = await _batched_score(
                ContextRecall(llm=judge_llm),
                inputs,
                samples,
                status_cb,
                "Context Recall",
                checkpoint,
                checkpoint_key="context_recall",
            )
            df = _score_df("context_recall", samples, scores)
            save_metric("context_recall", df)
            logfire.info("🧪 Context Recall done", avg=round(df["context_recall"].mean(), 3))

        await _cooldown(COOLDOWN_STANDARD, "Context Recall", status_cb)

        # ── Exp 5: Answer Correctness (split into batches) ────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 5/6 — Answer Correctness batch 1/2...")
        with logfire.span("🧪 Exp 5 — Answer Correctness"):
            inputs = [
                {
                    "user_input": s["question"],
                    "response": s["actual_response"],
                    "reference": s["reference"],
                }
                for s in samples
            ]
            all_scores = await _batched_score(
                AnswerCorrectness(llm=judge_llm, embeddings=ragas_embeddings),
                inputs,
                samples,
                status_cb,
                "Answer Correctness",
                checkpoint,
                checkpoint_key="answer_correctness",
            )
            df = _score_df("answer_correctness", samples, all_scores)
            save_metric("answer_correctness", df)
            logfire.info("🧪 Answer Correctness done", avg=round(df["answer_correctness"].mean(), 3))

        if status_cb:
            status_cb("✅ All 5 experiments complete!")

        _save_metric_version(checkpoint, len(samples))

    return results
