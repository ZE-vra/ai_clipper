from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.thumbnails.domain.geometry import BoundingBox, Point
from src.thumbnails.perception.crop import CropAnalyzerConfig
from src.thumbnails.perception.crop_analyzer import CropAnalyzer
from src.thumbnails.perception.focal import FocalAnalysisEvidence, FocalRegionEvidence
from src.thumbnails.perception.frame_perception import FramePerception
from src.thumbnails.perception.quality import FrameQualityEvidence
from src.thumbnails.perception.subjects import SubjectAnalysisEvidence, SubjectEvidence, SubjectKind


def make_perception(path: Path, subjects, focal_bounds=None) -> FramePerception:
    focal = FocalAnalysisEvidence(regions=())
    if focal_bounds is not None:
        focal = FocalAnalysisEvidence(
            regions=(
                FocalRegionEvidence(
                    region_id="focal_01",
                    bounds=focal_bounds,
                    focal_point=focal_bounds.center,
                    strength=0.8,
                    area_ratio=focal_bounds.width * focal_bounds.height,
                ),
            )
        )

    return FramePerception(
        frame_path=path,
        quality=FrameQualityEvidence(
            sharpness=0.8,
            brightness=0.8,
            contrast=0.8,
            saturation=0.8,
            motion_blur=0.0,
            noise=0.0,
            overall_quality=0.8,
        ),
        focal=focal,
        subjects=SubjectAnalysisEvidence(subjects=tuple(subjects)),
        crop=None,  # CropAnalyzer only consumes frame_path, subjects and focal evidence.
    )


def person(subject_id: str, left: float, right: float, prominence: float) -> SubjectEvidence:
    bounds = BoundingBox(left, 0.15, right, 0.85)
    return SubjectEvidence(
        subject_id=subject_id,
        kind=SubjectKind.PERSON,
        confidence=0.9,
        bounds=bounds,
        focal_point=bounds.center,
        prominence=prominence,
    )


def test_primary_subject_is_retained_by_centered_vertical_crop(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (1920, 1080), "white").save(image_path)

    primary = person("primary", 0.70, 0.90, 0.9)
    perception = make_perception(image_path, [primary])

    result = CropAnalyzer().analyze(perception)

    assert result.primary_subject_retention == 1.0
    assert result.retained_subject_ratio == 1.0
    assert result.score == 1.0


def test_wide_multi_subject_scene_reports_loss_of_secondary_subject(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (1920, 1080), "white").save(image_path)

    primary = person("primary", 0.70, 0.90, 0.9)
    secondary = person("secondary", 0.05, 0.25, 0.7)
    perception = make_perception(image_path, [primary, secondary])

    result = CropAnalyzer().analyze(perception)

    assert result.primary_subject_retention == 1.0
    assert result.retained_subject_ratio == 0.5
    assert result.score == 0.75


def test_crop_analyzer_config_can_make_primary_retention_dominant() -> None:
    config = CropAnalyzerConfig(
        subject_weight=0.10,
        primary_subject_weight=0.80,
        focal_weight=0.10,
    )

    assert config.primary_subject_weight > config.subject_weight
    assert sum(
        (
            config.subject_weight,
            config.primary_subject_weight,
            config.focal_weight,
        )
    ) == 1.0
