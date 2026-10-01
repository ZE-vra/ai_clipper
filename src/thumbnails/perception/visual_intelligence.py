from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.thumbnails.domain.geometry import BoundingBox


@dataclass(frozen=True)
class SemanticSubject:
    """Semantic subject evidence backed by an instance segmentation mask."""

    subject_id: str
    kind: str
    confidence: float
    bounds: BoundingBox
    mask_path: Path
    area_ratio: float
    importance: float

    def __post_init__(self) -> None:
        if not self.subject_id.strip():
            raise ValueError("subject_id must not be blank.")
        if not self.kind.strip():
            raise ValueError("kind must not be blank.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        if not 0.0 < self.area_ratio <= 1.0:
            raise ValueError("area_ratio must be between 0 and 1.")
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError("importance must be between 0 and 1.")
        if not self.mask_path.name:
            raise ValueError("mask_path must reference a file.")


@dataclass(frozen=True)
class VisualIntelligenceResult:
    """Semantic visual evidence used by thumbnail composition and rendering."""

    frame_path: Path
    subjects: tuple[SemanticSubject, ...] = field(default_factory=tuple)
    analysis_version: str = "1"

    def __post_init__(self) -> None:
        if not self.frame_path.name:
            raise ValueError("frame_path must reference a file.")
        if not self.analysis_version.strip():
            raise ValueError("analysis_version must not be blank.")

    @property
    def primary_subject(self) -> SemanticSubject | None:
        if not self.subjects:
            return None
        return max(self.subjects, key=lambda subject: subject.importance)
