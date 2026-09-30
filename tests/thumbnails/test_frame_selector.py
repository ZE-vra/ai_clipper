from pathlib import Path

import pytest

from src.thumbnails.perception.crop import CropSuitabilityEvidence
from src.thumbnails.perception.focal import FocalAnalysisEvidence
from src.thumbnails.perception.frame_perception import FramePerception
from src.thumbnails.perception.frame_selector import FrameSelector
from src.thumbnails.perception.quality import FrameQualityEvidence
from src.thumbnails.perception.scoring import FrameCandidateScoringConfig
from src.thumbnails.perception.subjects import SubjectAnalysisEvidence


def make_perception(
    *,
    path: str,
    quality: float,
    subject_prominence: float | None,
    focal_strength: float | None,
) -> FramePerception:
    subjects = SubjectAnalysisEvidence()

    if subject_prominence is not None:
        from src.thumbnails.domain.geometry import BoundingBox, Point
        from src.thumbnails.perception.subjects import (
            SubjectEvidence,
            SubjectKind,
        )

        subjects = SubjectAnalysisEvidence(
            subjects=(
                SubjectEvidence(
                    subject_id="person_01",
                    kind=SubjectKind.PERSON,
                    confidence=0.9,
                    bounds=BoundingBox(
                        left=0.2,
                        top=0.2,
                        right=0.7,
                        bottom=0.9,
                    ),
                    focal_point=Point(x=0.45, y=0.55),
                    prominence=subject_prominence,
                ),
            ),
        )

    crop = CropSuitabilityEvidence(0.9 if subject_prominence is not None else 0.0, 1.0 if subject_prominence is not None else 0.0, 1.0 if subject_prominence is not None else 0.0, 1.0 if focal_strength is not None else 0.0, 1 if subject_prominence is not None else 0)

    focal = FocalAnalysisEvidence(regions=())
    if focal_strength is not None:
        from src.thumbnails.domain.geometry import BoundingBox, Point
        from src.thumbnails.perception.focal import FocalRegionEvidence

        focal = FocalAnalysisEvidence(
            regions=(
                FocalRegionEvidence(
                    region_id="region_01",
                    bounds=BoundingBox(
                        left=0.2,
                        top=0.2,
                        right=0.7,
                        bottom=0.9,
                    ),
                    focal_point=Point(x=0.45, y=0.55),
                    strength=focal_strength,
                    area_ratio=0.35,
                ),
            ),
        )

    return FramePerception(
        frame_path=Path(path),
        quality=FrameQualityEvidence(
            sharpness=quality,
            brightness=quality,
            contrast=quality,
            saturation=quality,
            motion_blur=0.0,
            noise=0.0,
            overall_quality=quality,
        ),
        focal=focal,
        subjects=subjects,
        crop=crop,
    )


def test_selects_highest_scoring_candidate() -> None:
    candidates = [
        make_perception(
            path="frame_01.jpg",
            quality=0.5,
            subject_prominence=0.5,
            focal_strength=0.5,
        ),
        make_perception(
            path="frame_02.jpg",
            quality=0.9,
            subject_prominence=0.9,
            focal_strength=0.9,
        ),
    ]

    selected = FrameSelector().select(candidates)

    assert selected.perception.frame_path == Path("frame_02.jpg")
    assert selected.score.overall_score > 0.8


def test_returns_selected_score() -> None:
    candidate = make_perception(
        path="frame_01.jpg",
        quality=0.8,
        subject_prominence=0.6,
        focal_strength=0.4,
    )

    selected = FrameSelector().select([candidate])

    assert selected.score.quality_score == 0.8
    assert selected.score.subject_score == 0.6
    assert selected.score.focal_score == 0.4
    assert selected.score.crop_score == 0.9


def test_selection_is_deterministic_for_equal_scores() -> None:
    first = make_perception(
        path="frame_01.jpg",
        quality=0.8,
        subject_prominence=0.8,
        focal_strength=0.8,
    )
    second = make_perception(
        path="frame_02.jpg",
        quality=0.8,
        subject_prominence=0.8,
        focal_strength=0.8,
    )

    selected = FrameSelector().select([first, second])

    assert selected.perception.frame_path == Path("frame_01.jpg")


def test_rejects_empty_candidates() -> None:
    with pytest.raises(ValueError, match="candidates must not be empty"):
        FrameSelector().select([])