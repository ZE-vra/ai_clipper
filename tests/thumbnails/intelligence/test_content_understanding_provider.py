import json

import pytest

from src.thumbnails.domain.brief import ThumbnailBrief
from src.thumbnails.domain.content import ContentUnderstanding
from src.thumbnails.intelligence.content_understanding_provider import (
    GeminiContentUnderstandingProvider,
    _call_with_transient_retries,
)


def _response():
    return {
        "brief": {
            "core_hook": "THE UNEXPECTED RESULT",
            "subject": "a science experiment",
            "promise": "understand the result",
            "curiosity_angle": "Why did the result change?",
            "important_entities": ["experiment"],
            "visual_evidence": ["the experiment setup"],
        },
        "understanding": {
            "entities": [
                {
                    "entity_id": "experiment",
                    "label": "Science experiment",
                    "kind": "concept",
                    "confidence": 0.9,
                    "evidence": ["The speaker describes the experiment."],
                }
            ],
            "events": [
                {
                    "event_id": "result",
                    "start_time": 0.0,
                    "end_time": 0.0,
                    "description": "The speaker explains the result.",
                    "entity_ids": ["experiment"],
                    "importance": 0.8,
                    "confidence": 0.9,
                }
            ],
            "themes": ["science"],
            "claims": ["The speaker describes an experiment."],
            "confidence": 0.85,
        },
    }


def test_provider_builds_validated_v2_inputs_from_transcript():
    captured = []
    provider = GeminiContentUnderstandingProvider(
        lambda prompt: captured.append(prompt) or json.dumps(_response())
    )

    brief, understanding = provider.understand("The speaker describes an experiment.")

    assert isinstance(brief, ThumbnailBrief)
    assert brief.important_entities == ("experiment",)
    assert isinstance(understanding, ContentUnderstanding)
    assert understanding.entities[0].evidence == ("The speaker describes the experiment.",)
    assert understanding.events[0].entity_ids == ("experiment",)
    assert "Do not invent visual details" in captured[0]
    assert "The speaker describes an experiment." in captured[0]


def test_provider_rejects_empty_transcript_without_calling_model():
    provider = GeminiContentUnderstandingProvider(
        lambda prompt: pytest.fail("model must not be called")
    )

    with pytest.raises(ValueError, match="Transcript must not be empty"):
        provider.understand("  ")


def test_provider_rejects_malformed_response():
    provider = GeminiContentUnderstandingProvider(lambda prompt: '{"brief": {}}')

    with pytest.raises(ValueError, match="must contain 'brief' and 'understanding'"):
        provider.understand("A non-empty transcript.")


def test_transient_gemini_503_retries_then_succeeds():
    outcomes = iter([
        RuntimeError("503 UNAVAILABLE: high demand"),
        RuntimeError("503 UNAVAILABLE: high demand"),
        "ok",
    ])
    waits = []

    result = _call_with_transient_retries(
        lambda: (lambda value: (_ for _ in ()).throw(value) if isinstance(value, Exception) else value)(next(outcomes)),
        sleep=waits.append,
    )

    assert result == "ok"
    assert waits == [2, 4]


def test_transient_gemini_errors_stop_after_three_attempts():
    attempts = []
    waits = []

    def request():
        attempts.append(1)
        raise RuntimeError("503 UNAVAILABLE: high demand")

    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        _call_with_transient_retries(request, sleep=waits.append)

    assert len(attempts) == 3
    assert waits == [2, 4]


def test_gemini_quota_error_is_not_retried():
    attempts = []

    def request():
        attempts.append(1)
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    with pytest.raises(RuntimeError, match="quota/rate limit reached"):
        _call_with_transient_retries(request, sleep=lambda _: pytest.fail("must not sleep"))

    assert len(attempts) == 1
