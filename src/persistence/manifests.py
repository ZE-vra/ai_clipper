"""Persistence layer for pipeline artifacts."""

import json
from pathlib import Path
from typing import Any, Dict

from src.config import ProjectWorkspace
from src.exceptions import ClipperError
from src.schemas import (
    CandidateManifest,
    ClipManifest,
    EvaluationManifest,
    Transcript,
    TranscriptSegment,
    VideoSource,
)


class PersistenceError(ClipperError):
    """Raised when a pipeline artifact cannot be saved or loaded."""


SOURCE_FILENAME = "source.json"


def _save_json(
    path: Path,
    data: Dict[str, Any],
) -> Path:
    try:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    except OSError as exc:
        raise PersistenceError(
            f"Could not save artifact: {path}"
        ) from exc

    return path


def _load_json(
    path: Path,
) -> Dict[str, Any]:
    if not path.exists():
        raise PersistenceError(
            f"Artifact does not exist: {path}"
        )

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise PersistenceError(
            f"Could not load artifact: {path}"
        ) from exc


def _load_source(
    data: Dict[str, Any],
) -> VideoSource:
    return VideoSource(
        source_type=data["source_type"],
        location=data["location"],
        title=data.get("title"),
    )


def save_source(
    source: VideoSource,
    workspace: ProjectWorkspace,
) -> Path:
    """Persist source metadata for durable project recovery."""

    path = (
        workspace.source_dir
        / SOURCE_FILENAME
    )

    return _save_json(
        path,
        {
            "source_type": source.source_type,
            "location": source.location,
            "title": source.title,
        },
    )


def load_source(
    workspace: ProjectWorkspace,
) -> VideoSource:
    """Load persisted source metadata."""

    path = (
        workspace.source_dir
        / SOURCE_FILENAME
    )

    data = _load_json(path)

    return _load_source(data)


def _load_segments(
    data: list,
) -> list:
    return [
        TranscriptSegment(
            id=int(segment["id"]),
            start=float(segment["start"]),
            end=float(segment["end"]),
            text=str(segment["text"]),
        )
        for segment in data
    ]


def save_transcript(
    transcript: Transcript,
    workspace: ProjectWorkspace,
) -> Path:
    path = (
        workspace.transcript_dir
        / "transcript.json"
    )

    return _save_json(
        path,
        transcript.to_dict(),
    )


def load_transcript(
    workspace: ProjectWorkspace,
) -> Transcript:
    path = (
        workspace.transcript_dir
        / "transcript.json"
    )

    data = _load_json(path)

    return Transcript(
        source=_load_source(
            data["source"]
        ),
        duration=float(
            data["duration"]
        ),
        language=str(
            data["language"]
        ),
        segments=_load_segments(
            data.get(
                "segments",
                [],
            )
        ),
    )


def save_candidate_manifest(
    manifest: CandidateManifest,
    workspace: ProjectWorkspace,
) -> Path:
    path = (
        workspace.candidates_dir
        / "candidates.json"
    )

    return _save_json(
        path,
        manifest.to_dict(),
    )


def load_candidate_manifest(
    workspace: ProjectWorkspace,
) -> CandidateManifest:
    path = (
        workspace.candidates_dir
        / "candidates.json"
    )

    data = _load_json(path)

    from src.schemas import CandidateWindow

    candidates = []

    for item in data.get(
        "candidates",
        [],
    ):
        candidates.append(
            CandidateWindow(
                candidate_id=int(
                    item["candidate_id"]
                ),
                start_time=float(
                    item["start_time"]
                ),
                end_time=float(
                    item["end_time"]
                ),
                transcript_text=str(
                    item["transcript_text"]
                ),
                segments=_load_segments(
                    item.get(
                        "segments",
                        [],
                    )
                ),
            )
        )

    return CandidateManifest(
        source=_load_source(
            data["source"]
        ),
        total_candidates=int(
            data.get(
                "total_candidates",
                len(candidates),
            )
        ),
        candidates=candidates,
    )


def save_evaluation_manifest(
    manifest: EvaluationManifest,
    workspace: ProjectWorkspace,
) -> Path:
    path = (
        workspace.evaluations_dir
        / "evaluations.json"
    )

    return _save_json(
        path,
        manifest.to_dict(),
    )


def load_evaluation_manifest(
    workspace: ProjectWorkspace,
) -> EvaluationManifest:
    path = (
        workspace.evaluations_dir
        / "evaluations.json"
    )

    data = _load_json(path)

    from src.schemas import CandidateEvaluation

    evaluations = []

    for item in data.get(
        "evaluations",
        [],
    ):
        evaluations.append(
            CandidateEvaluation(
                candidate_id=int(
                    item["candidate_id"]
                ),
                score=float(
                    item["score"]
                ),
                reason=str(
                    item.get(
                        "reason",
                        "",
                    )
                ),
                suggested_title=item.get(
                    "suggested_title"
                ),
                start_segment_id=(
                    int(item["start_segment_id"])
                    if item.get("start_segment_id") is not None
                    else None
                ),
                end_segment_id=(
                    int(item["end_segment_id"])
                    if item.get("end_segment_id") is not None
                    else None
                ),
            )
        )

    return EvaluationManifest(
        source=_load_source(
            data["source"]
        ),
        evaluations=evaluations,
    )


def save_clip_manifest(
    manifest: ClipManifest,
    workspace: ProjectWorkspace,
) -> Path:
    path = (
        workspace.evaluations_dir
        / "clip_plan.json"
    )

    return _save_json(
        path,
        manifest.to_dict(),
    )


def load_clip_manifest(
    workspace: ProjectWorkspace,
) -> ClipManifest:
    path = (
        workspace.evaluations_dir
        / "clip_plan.json"
    )

    data = _load_json(path)

    from src.schemas import ClipDecision

    selected_clips = []

    for item in data.get(
        "selected_clips",
        [],
    ):
        selected_clips.append(
            ClipDecision(
                clip_id=int(
                    item["clip_id"]
                ),
                candidate_id=int(
                    item["candidate_id"]
                ),
                snapped_start_time=float(
                    item["snapped_start_time"]
                ),
                snapped_end_time=float(
                    item["snapped_end_time"]
                ),
                final_score=float(
                    item["final_score"]
                ),
                reason=str(
                    item.get(
                        "reason",
                        "",
                    )
                ),
                title=item.get(
                    "title"
                ),
            )
        )

    return ClipManifest(
        source=_load_source(
            data["source"]
        ),
        selected_clips=selected_clips,
    )