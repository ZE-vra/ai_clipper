from __future__ import annotations

import pytest

from src.thumbnails.domain.geometry import (
    BoundingBox,
    Point,
)
from src.thumbnails.perception.focal import (
    FocalAnalysisEvidence,
    FocalRegionEvidence,
)


def _bounds() -> BoundingBox:
    return BoundingBox(
        left=0.2,
        top=0.1,
        right=0.6,
        bottom=0.7,
    )


def _point() -> Point:
    return Point(
        x=0.4,
        y=0.4,
    )


def test_focal_region_accepts_valid_evidence() -> None:
    evidence = FocalRegionEvidence(
        region_id="region_01",
        bounds=_bounds(),
        focal_point=_point(),
        strength=0.8,
        area_ratio=0.24,
    )

    assert evidence.region_id == "region_01"
    assert evidence.strength == 0.8
    assert evidence.area_ratio == 0.24


def test_focal_region_rejects_blank_id() -> None:
    with pytest.raises(ValueError):
        FocalRegionEvidence(
            region_id="",
            bounds=_bounds(),
            focal_point=_point(),
            strength=0.8,
            area_ratio=0.24,
        )


@pytest.mark.parametrize(
    "strength",
    [-0.1, 1.1],
)
def test_focal_region_rejects_invalid_strength(
    strength: float,
) -> None:
    with pytest.raises(ValueError):
        FocalRegionEvidence(
            region_id="region_01",
            bounds=_bounds(),
            focal_point=_point(),
            strength=strength,
            area_ratio=0.24,
        )


@pytest.mark.parametrize(
    "area_ratio",
    [0.0, -0.1, 1.1],
)
def test_focal_region_rejects_invalid_area_ratio(
    area_ratio: float,
) -> None:
    with pytest.raises(ValueError):
        FocalRegionEvidence(
            region_id="region_01",
            bounds=_bounds(),
            focal_point=_point(),
            strength=0.8,
            area_ratio=area_ratio,
        )


def test_analysis_accepts_multiple_regions() -> None:
    first = FocalRegionEvidence(
        region_id="region_01",
        bounds=_bounds(),
        focal_point=_point(),
        strength=0.5,
        area_ratio=0.24,
    )

    second = FocalRegionEvidence(
        region_id="region_02",
        bounds=BoundingBox(
            left=0.6,
            top=0.2,
            right=0.9,
            bottom=0.8,
        ),
        focal_point=Point(
            x=0.75,
            y=0.5,
        ),
        strength=0.9,
        area_ratio=0.18,
    )

    analysis = FocalAnalysisEvidence(
        regions=(first, second),
    )

    assert len(analysis.regions) == 2


def test_primary_region_is_strongest() -> None:
    weak = FocalRegionEvidence(
        region_id="weak",
        bounds=_bounds(),
        focal_point=_point(),
        strength=0.4,
        area_ratio=0.24,
    )

    strong = FocalRegionEvidence(
        region_id="strong",
        bounds=BoundingBox(
            left=0.5,
            top=0.2,
            right=0.9,
            bottom=0.8,
        ),
        focal_point=Point(
            x=0.7,
            y=0.5,
        ),
        strength=0.9,
        area_ratio=0.24,
    )

    analysis = FocalAnalysisEvidence(
        regions=(weak, strong),
    )

    assert analysis.primary_region is strong


def test_primary_region_is_none_when_empty() -> None:
    analysis = FocalAnalysisEvidence(
        regions=(),
    )

    assert analysis.primary_region is None


def test_analysis_rejects_blank_version() -> None:
    with pytest.raises(ValueError):
        FocalAnalysisEvidence(
            regions=(),
            analysis_version="",
        )