from __future__ import annotations

from src.thumbnails.perception.frame_perception import FramePerception
from src.thumbnails.perception.scoring import (
    FrameCandidateScore,
    FrameCandidateScoringConfig,
)


class FrameCandidateScorer:
    """
    Converts frame perception evidence into candidate-scoring evidence.

    This component scores observed characteristics only.
    It does not select a frame or make creative decisions.
    """

    def __init__(
        self,
        config: FrameCandidateScoringConfig | None = None,
    ) -> None:
        self.config = (
            config
            or FrameCandidateScoringConfig()
        )

    def score(
        self,
        perception: FramePerception,
    ) -> FrameCandidateScore:
        quality_score = (
            perception.quality.overall_quality
        )

        subject_score = self._subject_score(
            perception
        )

        focal_score = self._focal_score(
            perception
        )

        crop_score = perception.crop.score

        overall_score = (
            self.config.quality_weight
            * quality_score
            + self.config.subject_weight
            * subject_score
            + self.config.focal_weight
            * focal_score
            + self.config.crop_weight
            * crop_score
        )

        return FrameCandidateScore(
            quality_score=quality_score,
            subject_score=subject_score,
            focal_score=focal_score,
            crop_score=crop_score,
            overall_score=overall_score,
        )

    @staticmethod
    def _subject_score(
        perception: FramePerception,
    ) -> float:
        subject = perception.subjects.primary_subject

        if subject is None:
            return 0.0

        return subject.prominence

    @staticmethod
    def _focal_score(
        perception: FramePerception,
    ) -> float:
        region = perception.focal.primary_region

        if region is None:
            return 0.0

        return region.strength