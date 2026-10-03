import json

import pytest

from src.thumbnails.domain.brief import ThumbnailBrief
from src.thumbnails.domain.content import ContentUnderstanding
from src.thumbnails.intelligence.content_understanding_provider import (
    GeminiContentUnderstandingProvider,
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
