"""
Unit tests — evaluation/ragas_eval.py

Tests:
  - evaluate_response: returns all 4 metric keys
  - evaluate_response: sets review_required=True when faithfulness < 0.80
  - evaluate_response: sets review_required=False when faithfulness >= 0.80
  - evaluate_response: handles RAGAS evaluation errors gracefully (doesn't raise)
  - FAITHFULNESS_THRESHOLD is 0.80
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd


class TestEvaluateResponse:
    @pytest.mark.asyncio
    async def test_returns_all_metric_keys(self, sample_chunks):
        fake_df = pd.DataFrame([{
            "faithfulness":      0.90,
            "answer_relevancy":  0.85,
            "context_precision": 0.75,
        }])
        fake_result = MagicMock()
        fake_result.to_pandas.return_value = fake_df

        with patch("evaluation.ragas_eval.evaluate", return_value=fake_result):
            from evaluation.ragas_eval import evaluate_response
            result = await evaluate_response(
                "What is the primary endpoint?",
                "The primary endpoint was a 30% reduction.",
                sample_chunks,
            )

        assert "faithfulness" in result
        assert "answer_relevancy" in result
        assert "context_precision" in result
        assert "context_recall" in result
        assert "review_required" in result

    @pytest.mark.asyncio
    async def test_review_required_true_below_threshold(self, sample_chunks):
        fake_df = pd.DataFrame([{
            "faithfulness":      0.65,   # below 0.80
            "answer_relevancy":  0.80,
            "context_precision": 0.70,
        }])
        fake_result = MagicMock()
        fake_result.to_pandas.return_value = fake_df

        with patch("evaluation.ragas_eval.evaluate", return_value=fake_result):
            from evaluation.ragas_eval import evaluate_response
            result = await evaluate_response("q", "a", sample_chunks)

        assert result["review_required"] is True
        assert result["faithfulness"] == pytest.approx(0.65)

    @pytest.mark.asyncio
    async def test_review_required_false_above_threshold(self, sample_chunks):
        fake_df = pd.DataFrame([{
            "faithfulness":      0.92,
            "answer_relevancy":  0.88,
            "context_precision": 0.80,
        }])
        fake_result = MagicMock()
        fake_result.to_pandas.return_value = fake_df

        with patch("evaluation.ragas_eval.evaluate", return_value=fake_result):
            from evaluation.ragas_eval import evaluate_response
            result = await evaluate_response("q", "a", sample_chunks)

        assert result["review_required"] is False

    @pytest.mark.asyncio
    async def test_handles_ragas_error_gracefully(self, sample_chunks):
        """If RAGAS fails (e.g. no API key), it should return zero scores, not raise."""
        with patch("evaluation.ragas_eval.evaluate", side_effect=Exception("RAGAS error")):
            from evaluation.ragas_eval import evaluate_response
            result = await evaluate_response("q", "a", sample_chunks)

        # Should not raise; should return safe defaults
        assert result["faithfulness"] == 0.0
        assert result["review_required"] is False
        assert "eval_error" in result

    def test_faithfulness_threshold_is_0_80(self):
        from evaluation.ragas_eval import FAITHFULNESS_THRESHOLD
        assert FAITHFULNESS_THRESHOLD == 0.80
