from pathlib import Path

import pytest

from src.thumbnails.domain.geometry import BoundingBox, Point
from src.thumbnails.perception.crop import CropSuitabilityEvidence
from src.thumbnails.perception.focal import (
    FocalAnalysisEvidence,
    FocalRegionEvidence,
)
from src.thumbnails.perception.frame_candidate_scorer import (
    FrameCandidateScorer,
)
from src.thumbnails.perception.frame_perception import (
    FramePerception,
)
from src.thumbnails.perception.quality import (
    FrameQualityEvidence,
)
from src.thumbnails.perception.scoring import (
    FrameCandidateScoringConfig,
)
from src.thumbnails.perception.subjects import (
    SubjectAnalysisEvidence,
    SubjectEvidence,
    SubjectKind,
)


def build_quality() -> FrameQualityEvidence:
    return FrameQualityEvidence(
        sharpness=0.8,
        brightness=0.5,
        contrast=0.7,
        saturation=0.6,
        motion_blur=0.1,
        noise=0.1,
        overall_quality=0.80,
    )


def build_focal() -> FocalAnalysisEvidence:
    return FocalAnalysisEvidence(
        regions=(
            FocalRegionEvidence(
                region_id="focal_01",
                bounds=BoundingBox(
                    left=0.2,
                    top=0.2,
                    right=0.6,
                    bottom=0.8,
                ),
                focal_point=Point(
                    x=0.4,
                    y=0.5,
                ),
                strength=0.70,
                area_ratio=0.24,
            ),
        ),
    )


def build_subjects() -> SubjectAnalysisEvidence:
    return SubjectAnalysisEvidence(
        subjects=(
            SubjectEvidence(
                subject_id="person_01",
                kind=SubjectKind.PERSON,
                confidence=0.90,
                bounds=BoundingBox(
                    left=0.2,
                    top=0.1,
                    right=0.6,
                    bottom=0.9,
                ),
                focal_point=Point(
                    x=0.4,
                    y=0.5,
                ),
                prominence=0.60,
            ),
        ),
    )


def build_perception() -> FramePerception:
    return FramePerception(
        frame_path=Path("frame.jpg"),
        quality=build_quality(),
        focal=build_focal(),
        subjects=build_subjects(),
        crop=CropSuitabilityEvidence(0.50, 1.0, 1.0, 1.0, 1),
    )


def test_scorer_extracts_perception_scores() -> None:
    scorer = FrameCandidateScorer()

    result = scorer.score(
        build_perception()
    )

    assert result.quality_score == pytest.approx(0.80)
    assert result.subject_score == pytest.approx(0.60)
    assert result.focal_score == pytest.approx(0.70)
    assert result.crop_score == pytest.approx(0.50)


def test_scorer_calculates_weighted_overall_score() -> None:
    scorer = FrameCandidateScorer(
        FrameCandidateScoringConfig(
            quality_weight=0.30,
            subject_weight=0.25,
            focal_weight=0.15,
            crop_weight=0.30,
        )
    )

    result = scorer.score(
        build_perception()
    )

    expected = (
        0.30 * 0.80
        + 0.25 * 0.60
        + 0.15 * 0.70
        + 0.30 * 0.50
    )

    assert result.overall_score == pytest.approx(
        expected
    )


def test_scorer_returns_zero_for_missing_subjects() -> None:
    perception = FramePerception(
        frame_path=Path("frame.jpg"),
        quality=build_quality(),
        focal=build_focal(),
        subjects=SubjectAnalysisEvidence(
            subjects=(),
        ),
    )

    result = FrameCandidateScorer().score(
        perception
    )

    assert result.subject_score == 0.0


def test_scorer_returns_zero_for_missing_focal_regions() -> None:
    perception = FramePerception(
        frame_path=Path("frame.jpg"),
        quality=build_quality(),
        focal=FocalAnalysisEvidence(
            regions=(),
        ),
        subjects=build_subjects(),
    )

    result = FrameCandidateScorer().score(
        perception
    )

    assert result.focal_score == 0.0