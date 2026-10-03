from pathlib import Path

import pytest

from src.thumbnails.intelligence.whisper_transcript_provider import (
    WhisperTranscriptProvider,
)


class FakeWhisperModel:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def transcribe(self, media_path, **options):
        self.calls.append((media_path, options))
        return self.result


def test_provider_transcribes_local_video_with_configured_model_and_language(tmp_path: Path):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    model = FakeWhisperModel({"text": "  A useful transcript.  "})
    loaded_models = []

    provider = WhisperTranscriptProvider(
        model_name="small",
        language="en",
        load_model=lambda name: loaded_models.append(name) or model,
    )

    transcript = provider.transcribe(video)

    assert transcript == "A useful transcript."
    assert loaded_models == ["small"]
    assert model.calls == [(str(video), {"language": "en"})]


def test_provider_joins_segments_when_whisper_text_is_missing(tmp_path: Path):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    model = FakeWhisperModel(
        {"segments": [{"text": " Hello "}, {"text": " world. "}]}
    )

    provider = WhisperTranscriptProvider(load_model=lambda name: model)

    assert provider.transcribe(video) == "Hello world."


def test_provider_rejects_missing_media(tmp_path: Path):
    provider = WhisperTranscriptProvider(load_model=lambda name: pytest.fail("must not load"))

    with pytest.raises(FileNotFoundError, match="Media file does not exist"):
        provider.transcribe(tmp_path / "missing.mp4")


def test_provider_rejects_empty_transcript(tmp_path: Path):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    provider = WhisperTranscriptProvider(
        load_model=lambda name: FakeWhisperModel({"text": "  ", "segments": []})
    )

    with pytest.raises(ValueError, match="empty transcript"):
        provider.transcribe(video)
