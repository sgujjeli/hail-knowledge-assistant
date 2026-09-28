"""
RAGAS evaluation pipeline — faithfulness, answer_relevancy, context_precision,
context_recall.

Why RAGAS?
  - Faithfulness: does the answer contain ONLY claims supported by retrieved context?
    Score < 0.80 → REVIEW_REQUIRED flag returned to frontend.
  - Answer Relevancy: does the answer actually address the question?
  - Context Precision: are retrieved chunks actually relevant to the question?
  - Context Recall: did retrieval surface the right chunks?

We run RAGAS asynchronously AFTER the streaming response completes, then
persist the scores to ChatSession in Postgres for analysis.

Endpoint used internally:
  Azure OpenAI is used as the LLM judge inside RAGAS via LangchainLLM adapter.

Docs: https://docs.ragas.io/en/stable/
"""
from datasets import Dataset

FAITHFULNESS_THRESHOLD = 0.80   # below this → REVIEW_REQUIRED

# ── RAGAS v0.1 vs v0.2 compatibility shim ───────────────────────────────────
# v0.2 renamed evaluate() → aevaluate() and restructured metric imports.
# We detect which version is installed and import accordingly.
try:
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    try:
        # v0.2+ async API
        from ragas import aevaluate as _ragas_evaluate
        _RAGAS_ASYNC = True
    except ImportError:
        # v0.1 sync API
        from ragas import evaluate as _ragas_evaluate   # type: ignore[assignment]
        _RAGAS_ASYNC = False
    _RAGAS_AVAILABLE = True
except Exception:
    _RAGAS_AVAILABLE = False
    _RAGAS_ASYNC = False


async def evaluate_response(
    query: str,
    response: str,
    chunks: list[dict],
) -> dict:
    """
    Run RAGAS evaluation on a single query-response pair.

    Args:
        query:    user question
        response: LLM-generated answer
        chunks:   retrieved context chunks from hybrid_search

    Returns:
        {
            "faithfulness": float,
            "answer_relevancy": float,
            "context_precision": float,
            "context_recall": float,
            "review_required": bool,  # True if faithfulness < 0.80
        }
    """
    if not _RAGAS_AVAILABLE:
        return _zero_scores(eval_error="ragas not available")

    contexts = [chunk["text"] for chunk in chunks]

    # RAGAS expects a HuggingFace Dataset with these exact column names
    data = {
        "question":     [query],
        "answer":       [response],
        "contexts":     [contexts],
        "ground_truth": [""],    # No ground truth at runtime — context_recall skipped
    }
    dataset = Dataset.from_dict(data)
    metrics = [faithfulness, answer_relevancy, context_precision]

    try:
        if _RAGAS_ASYNC:
            result = await _ragas_evaluate(dataset, metrics=metrics)
        else:
            result = _ragas_evaluate(dataset, metrics=metrics)

        scores = result.to_pandas().iloc[0].to_dict()
        faithfulness_score = float(scores.get("faithfulness", 0.0))
        return {
            "faithfulness":      faithfulness_score,
            "answer_relevancy":  float(scores.get("answer_relevancy", 0.0)),
            "context_precision": float(scores.get("context_precision", 0.0)),
            "context_recall":    0.0,
            "review_required":   faithfulness_score < FAITHFULNESS_THRESHOLD,
        }

    except Exception as e:
        # Evaluation failure must never block the user-facing response
        return _zero_scores(eval_error=str(e))


def _zero_scores(eval_error: str = "") -> dict:
    return {
        "faithfulness":      0.0,
        "answer_relevancy":  0.0,
        "context_precision": 0.0,
        "context_recall":    0.0,
        "review_required":   False,
        "eval_error":        eval_error,
    }
