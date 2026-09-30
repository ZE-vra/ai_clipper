from __future__ import annotations

import pytest

from src.thumbnails.domain.geometry import BoundingBox
from src.thumbnails.perception.crop import (
    CropAnalyzerConfig,
    CropSuitabilityEvidence,
    calculate_horizontal_crop_bounds,
    horizontal_retention,
)


def test_crop_suitability_evidence_validates_scores() -> None:
    evidence = CropSuitabilityEvidence(
        score=0.8,
        retained_subject_ratio=0.75,
        primary_subject_retention=0.9,
        focal_retention=0.7,
        subject_count=3,
    )

    assert evidence.score == 0.8
    assert evidence.subject_count == 3


@pytest.mark.parametrize(
    "field",
    (
        "score",
        "retained_subject_ratio",
        "primary_subject_retention",
        "focal_retention",
    ),
)
def test_crop_suitability_evidence_rejects_invalid_scores(field: str) -> None:
    values = {
        "score": 0.5,
        "retained_subject_ratio": 0.5,
        "primary_subject_retention": 0.5,
        "focal_retention": 0.5,
        "subject_count": 1,
    }

    values[field] = 1.1

    with pytest.raises(ValueError):
        CropSuitabilityEvidence(**values)


def test_crop_analyzer_config_exposes_target_aspect_ratio() -> None:
    config = CropAnalyzerConfig(
        target_width=1080,
        target_height=1920,
    )

    assert config.target_aspect_ratio == pytest.approx(1080 / 1920)


def test_horizontal_crop_bounds_center_on_subject() -> None:
    left, right = calculate_horizontal_crop_bounds(
        source_aspect_ratio=16 / 9,
        target_aspect_ratio=9 / 16,
        center_x=0.5,
    )

    assert left == pytest.approx(0.341796875)
    assert right == pytest.approx(0.658203125)


def test_horizontal_crop_bounds_stay_inside_frame() -> None:
    left, right = calculate_horizontal_crop_bounds(
        source_aspect_ratio=16 / 9,
        target_aspect_ratio=9 / 16,
        center_x=0.1,
    )

    assert left == pytest.approx(0.0)
    assert right == pytest.approx(0.31640625)
    
def test_horizontal_retention_is_full_when_inside_crop() -> None:
    bounds = BoundingBox(
        left=0.3,
        top=0.2,
        right=0.5,
        bottom=0.8,
    )

    retention = horizontal_retention(
        bounds,
        crop_left=0.2,
        crop_right=0.8,
    )

    assert retention == pytest.approx(1.0)


def test_horizontal_retention_is_zero_when_outside_crop() -> None:
    bounds = BoundingBox(
        left=0.8,
        top=0.2,
        right=0.95,
        bottom=0.8,
    )

    retention = horizontal_retention(
        bounds,
        crop_left=0.1,
        crop_right=0.7,
    )

    assert retention == pytest.approx(0.0)


def test_horizontal_retention_is_partial_when_crop_cuts_subject() -> None:
    bounds = BoundingBox(
        left=0.4,
        top=0.2,
        right=0.8,
        bottom=0.8,
    )

    retention = horizontal_retention(
        bounds,
        crop_left=0.2,
        crop_right=0.6,
    )

    assert retention == pytest.approx(0.5)


def test_horizontal_crop_is_full_for_narrow_source() -> None:
    left, right = calculate_horizontal_crop_bounds(
        source_aspect_ratio=9 / 16,
        target_aspect_ratio=9 / 16,
        center_x=0.5,
    )

    assert left == 0.0
    assert right == 1.0