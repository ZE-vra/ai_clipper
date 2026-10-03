import json
from pathlib import Path

import pytest

from src.thumbnails.cli import EmptySubjectAnalyzer, _load_request, build_parser
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


def test_cli_supports_automatic_transcription_and_whisper_options():
    args = build_parser().parse_args([
        "source.mp4", "--auto-transcribe", "--whisper-model", "small",
        "--language", "en", "--output", "thumbnail.jpg",
    ])

    assert args.auto_transcribe is True
    assert args.whisper_model == "small"
    assert args.language == "en"
    assert args.transcript is None
    assert args.input is None


def test_main_wires_auto_transcription_into_gemini_and_orchestrator(tmp_path: Path, monkeypatch, capsys) -> None:
    from types import SimpleNamespace

    import src.thumbnails.cli as cli

    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"video")
    output_path = tmp_path / "output" / "thumbnail.png"
    calls = {}

    class FakeWhisperProvider:
        def __init__(self, model_name: str, language: str | None = None) -> None:
            calls["whisper_options"] = (model_name, language)

        def transcribe(self, media_path: Path) -> str:
            calls["media_path"] = media_path
            return "  The host reveals a hidden room.  "

    class FakeGeminiProvider:
        def understand(self, transcript: str):
            calls["transcript"] = transcript
            return ThumbnailBrief(core_hook="HIDDEN ROOM"), ContentUnderstanding()

    class FakeOrchestrator:
        def __init__(self, frame_discovery, subject_mask_provider=None) -> None:
            calls["orchestrator_mask_provider"] = subject_mask_provider

        def generate(self, **kwargs):
            calls["generate_kwargs"] = kwargs
            return SimpleNamespace(
                status=SimpleNamespace(value="success"),
                output_path=kwargs["output_path"],
                selected_attempt_id="attempt-1",
                failure_reason=None,
            )

    monkeypatch.setattr(cli, "WhisperTranscriptProvider", FakeWhisperProvider)
    monkeypatch.setattr(
        cli.GeminiContentUnderstandingProvider,
        "from_env",
        classmethod(lambda cls: FakeGeminiProvider()),
    )
    monkeypatch.setattr(cli, "ThumbnailOrchestrator", FakeOrchestrator)

    exit_code = cli.main([
        str(source_video),
        "--auto-transcribe",
        "--whisper-model", "small",
        "--language", "en",
        "--output", str(output_path),
    ])

    assert exit_code == 0
    assert calls["whisper_options"] == ("small", "en")
    assert calls["media_path"] == source_video
    assert calls["transcript"] == "  The host reveals a hidden room.  "
    assert calls["generate_kwargs"]["source_video_path"] == source_video
    assert calls["generate_kwargs"]["output_path"] == output_path
    assert calls["generate_kwargs"]["brief"].core_hook == "HIDDEN ROOM"
    assert calls["orchestrator_mask_provider"] is None
    assert output_path.parent.is_dir()
    assert "Thumbnail success:" in capsys.readouterr().out
