import json
from pathlib import Path

import pytest

from src.thumbnails.cli import EmptySubjectAnalyzer, _load_request
from src.thumbnails.domain.content import ContentUnderstanding
from src.thumbnails.domain.brief import ThumbnailBrief
from src.thumbnails.domain.target import ThumbnailTarget
from src.thumbnails.perception.subjects import SubjectAnalysisEvidence


def test_load_request_builds_brief_understanding_and_default_target(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "brief": {
                    "core_hook": "THE HIDDEN REVEAL",
                    "subject": "a person",
                    "promise": "See what happened",
                    "curiosity_angle": "What is behind the door?",
                    "important_entities": ["host"],
                    "visual_evidence": ["door opening"],
                },
                "understanding": {
                    "entities": [
                        {
                            "entity_id": "host",
                            "label": "Host",
                            "kind": "person",
                            "confidence": 0.95,
                        }
                    ],
                    "events": [
                        {
                            "event_id": "reveal",
                            "start_time": 4.0,
                            "end_time": 8.0,
                            "description": "The host opens the door",
                            "entity_ids": ["host"],
                            "importance": 0.9,
                            "confidence": 0.9,
                        }
                    ],
                    "themes": ["surprise"],
                    "confidence": 0.9,
                },
            }
        ),
        encoding="utf-8",
    )

    brief, understanding, target = _load_request(request_path)

    assert isinstance(brief, ThumbnailBrief)
    assert brief.important_entities == ("host",)
    assert isinstance(understanding, ContentUnderstanding)
    assert understanding.entities[0].label == "Host"
    assert understanding.events[0].entity_ids == ("host",)
    assert isinstance(target, ThumbnailTarget)
    assert (target.size.width, target.size.height) == (1280, 720)


def test_load_request_rejects_missing_required_sections(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps({"brief": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="understanding"):
        _load_request(request_path)


def test_empty_subject_analyzer_makes_no_semantic_subject_claims() -> None:
    result = EmptySubjectAnalyzer().analyze(Path("unused.jpg"))

    assert isinstance(result, SubjectAnalysisEvidence)
    assert result.subjects == ()
