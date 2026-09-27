"""Module 2: Perception

Transcribes extracted audio using Whisper and returns structured
Transcript and TranscriptSegment contract objects.
"""

from pathlib import Path
from typing import Optional

import whisper

from src.config import ProjectWorkspace
from src.exceptions import AudioCorruptError, PerceptionError
from src.persistence.manifests import save_transcript
from src.schemas import (
    Transcript,
    TranscriptSegment,
    VideoSource,
)


def transcribe_audio(
    audio_path: Path,
    source: VideoSource,
    workspace: ProjectWorkspace,
    model_name: str = "base",
    language: Optional[str] = None,
) -> Transcript:
    """Transcribe an audio file into a structured Transcript."""

    if not audio_path.exists():
        raise PerceptionError(
            f"Audio file not found at: {audio_path}"
        )

    try:
        model = whisper.load_model(
            model_name
        )

        options = {}

        if language:
            options["language"] = language

        result = model.transcribe(
            str(audio_path),
            **options,
        )

    except Exception as exc:
        error_message = str(exc).lower()

        if (
            "corrupt" in error_message
            or "invalid" in error_message
        ):
            raise AudioCorruptError(
                f"Audio file at {audio_path} "
                "is unreadable or corrupt."
            ) from exc

        raise PerceptionError(
            f"Whisper transcription failed: {exc}"
        ) from exc

    raw_segments = result.get(
        "segments",
        [],
    )

    segments = []

    for idx, segment in enumerate(
        raw_segments
    ):
        segments.append(
            TranscriptSegment(
                id=idx,
                start=round(
                    float(segment["start"]),
                    3,
                ),
                end=round(
                    float(segment["end"]),
                    3,
                ),
                text=segment["text"].strip(),
            )
        )

    duration = (
        segments[-1].end
        if segments
        else 0.0
    )

    detected_language = result.get(
        "language",
        "en",
    )

    transcript = Transcript(
        source=source,
        duration=duration,
        language=detected_language,
        segments=segments,
    )

    save_transcript(
        transcript,
        workspace,
    )

    return transcript