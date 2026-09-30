from __future__ import annotations

from dataclasses import dataclass


def _validate_score(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be between 0 and 1."
        )


@dataclass(frozen=True)
class FrameCandidateScore:
    """
    Structured scoring evidence for one frame.

    This record describes how useful a frame's observed
    characteristics may be. It does not select the frame.
    """

    quality_score: float
    subject_score: float
    focal_score: float
    crop_score: float
    overall_score: float

    def __post_init__(self) -> None:
        for name, value in (
            ("quality_score", self.quality_score),
            ("subject_score", self.subject_score),
            ("focal_score", self.focal_score),
            ("crop_score", self.crop_score),
            ("overall_score", self.overall_score),
        ):
            _validate_score(value, name)


@dataclass(frozen=True)
class FrameCandidateScoringConfig:
    """
    Weights used to combine independent frame-scoring signals.
    """

    quality_weight: float = 0.30
    subject_weight: float = 0.25
    focal_weight: float = 0.15
    crop_weight: float = 0.30

    def __post_init__(self) -> None:
        weights = (
            self.quality_weight,
            self.subject_weight,
            self.focal_weight,
            self.crop_weight,
        )

        if any(weight < 0.0 for weight in weights):
            raise ValueError(
                "scoring weights must not be negative."
            )

        if sum(weights) <= 0.0:
            raise ValueError(
                "at least one scoring weight must be greater than 0."
            )