from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from src.thumbnails.domain.geometry import BoundingBox, Point


class SubjectKind(str, Enum):
    PERSON = "person"
    FACE = "face"
    OBJECT = "object"
    VEHICLE = "vehicle"
    PRODUCT = "product"
    SCREEN = "screen"
    ENVIRONMENT = "environment"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SubjectEvidence:
    """
    Structured evidence about one detected semantic subject.

    A subject is perception evidence, not a creative decision.
    """

    subject_id: str
    kind: SubjectKind
    confidence: float
    bounds: BoundingBox
    focal_point: Point
    prominence: float
    occluded: bool = False

    def __post_init__(self) -> None:
        if not self.subject_id.strip():
            raise ValueError("subject_id must not be blank.")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "confidence must be between 0 and 1."
            )

        if not 0.0 <= self.prominence <= 1.0:
            raise ValueError(
                "prominence must be between 0 and 1."
            )


@dataclass(frozen=True)
class SubjectAnalysisEvidence:
    """
    Complete semantic-subject analysis for one frame.

    This record describes what a subject analyzer detected.
    It does not rank thumbnail candidates or make layout decisions.
    """

    subjects: tuple[SubjectEvidence, ...] = field(
        default_factory=tuple
    )
    analysis_version: str = "1"

    def __post_init__(self) -> None:
        if not self.analysis_version.strip():
            raise ValueError(
                "analysis_version must not be blank."
            )

        subject_ids = [
            subject.subject_id
            for subject in self.subjects
        ]

        if len(subject_ids) != len(set(subject_ids)):
            raise ValueError(
                "subject_ids must be unique within an analysis."
            )

    @property
    def people(self) -> tuple[SubjectEvidence, ...]:
        """Return detected person subjects."""
        return tuple(
            subject
            for subject in self.subjects
            if subject.kind is SubjectKind.PERSON
        )

    @property
    def faces(self) -> tuple[SubjectEvidence, ...]:
        """Return detected face subjects."""
        return tuple(
            subject
            for subject in self.subjects
            if subject.kind is SubjectKind.FACE
        )

    @property
    def primary_subject(self) -> SubjectEvidence | None:
        """
        Return the most prominent detected subject.

        This is only a convenience accessor over perception evidence.
        It is not a thumbnail-selection decision.
        """
        if not self.subjects:
            return None

        return max(
            self.subjects,
            key=lambda subject: subject.prominence,
        )