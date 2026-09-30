from __future__ import annotations

from dataclasses import dataclass, field

from src.thumbnails.domain.assets import FrameCandidate
from src.thumbnails.domain.geometry import BoundingBox, Point
from src.thumbnails.perception.quality import FrameQualityEvidence
from src.thumbnails.perception.subjects import SubjectEvidence


@dataclass(frozen=True)
class FrameVisualEvidence:
    """
    Complete visual evidence associated with one candidate frame.

    This is an observation record. It does not select or rank the frame.
    """

    candidate: FrameCandidate
    quality: FrameQualityEvidence

    subjects: tuple[SubjectEvidence, ...] = field(
        default_factory=tuple
    )

    focal_regions: tuple[BoundingBox, ...] = field(
        default_factory=tuple
    )

    analysis_version: str = "1"

    def __post_init__(self) -> None:
        if not self.analysis_version.strip():
            raise ValueError(
                "analysis_version must not be blank."
            )

    @property
    def primary_focal_point(self) -> Point | None:
        """
        Return the focal point of the most prominent subject.

        This is an observation convenience, not a creative selection.
        """
        if not self.subjects:
            return None

        return max(
            self.subjects,
            key=lambda subject: subject.prominence,
        ).focal_point
