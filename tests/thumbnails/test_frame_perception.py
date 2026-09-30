from pathlib import Path

import pytest

from src.thumbnails.perception.focal import (
    FocalAnalysisEvidence,
    FocalRegionEvidence,
)
from src.thumbnails.perception.frame_perception import (
    FramePerception,
)
from src.thumbnails.perception.quality import (
    FrameQualityEvidence,
)
from src.thumbnails.perception.subjects import (
    SubjectAnalysisEvidence,
)


def build_quality() -> FrameQualityEvidence:
    return FrameQualityEvidence(
        sharpness=0.8,
        brightness=0.5,
        contrast=0.7,
        saturation=0.6,
        motion_blur=0.1,
        noise=0.1,
        overall_quality=0.75,
    )


def build_focal() -> FocalAnalysisEvidence:
    from src.thumbnails.domain.geometry import (
        BoundingBox,
        Point,
    )

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
                strength=0.8,
                area_ratio=0.24,
            ),
        ),
    )


def build_subjects() -> SubjectAnalysisEvidence:
    return SubjectAnalysisEvidence(
        subjects=(),
    )


def test_frame_perception_stores_all_perception_evidence() -> None:
    frame_path = Path("frame_01.jpg")

    perception = FramePerception(
        frame_path=frame_path,
        quality=build_quality(),
        focal=build_focal(),
        subjects=build_subjects(),
    )

    assert perception.frame_path == frame_path
    assert perception.quality.overall_quality == pytest.approx(0.75)
    assert perception.focal.primary_region is not None
    assert perception.subjects.subjects == ()


def test_frame_perception_is_immutable() -> None:
    perception = FramePerception(
        frame_path=Path("frame_01.jpg"),
        quality=build_quality(),
        focal=build_focal(),
        subjects=build_subjects(),
    )

    with pytest.raises(AttributeError):
        perception.frame_path = Path("frame_02.jpg")