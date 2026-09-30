from __future__ import annotations

from pathlib import Path

from src.thumbnails.perception.crop import (
    CropAnalyzerConfig,
    CropSuitabilityEvidence,
    calculate_horizontal_crop_bounds,
    horizontal_retention,
)
from src.thumbnails.perception.frame_perception import FramePerception


class CropAnalyzer:
    """
    Evaluates how well a frame can support the configured vertical crop.

    This component evaluates geometric crop viability only.
    It does not make creative decisions or select a frame.
    """

    def __init__(
        self,
        *,
        config: CropAnalyzerConfig | None = None,
    ) -> None:
        self.config = config or CropAnalyzerConfig()

    def analyze(
        self,
        perception: FramePerception,
    ) -> CropSuitabilityEvidence:
        source_width, source_height = self._read_dimensions(
            perception.frame_path
        )

        source_aspect_ratio = source_width / source_height

        primary_subject = perception.subjects.primary_subject

        if primary_subject is None:
            center_x = 0.5
        else:
            center_x = primary_subject.focal_point.x

        crop_left, crop_right = calculate_horizontal_crop_bounds(
            source_aspect_ratio=source_aspect_ratio,
            target_aspect_ratio=self.config.target_aspect_ratio,
            center_x=center_x,
        )

        subjects = perception.subjects.subjects

        if not subjects:
            retained_subject_ratio = 0.0
            primary_subject_retention = 0.0
        else:
            retentions = [
                horizontal_retention(
                    subject.bounds,
                    crop_left,
                    crop_right,
                )
                for subject in subjects
            ]

            retained_subject_ratio = sum(
                retention >= self.config.minimum_retention
                for retention in retentions
            ) / len(retentions)

            if primary_subject is None:
                primary_subject_retention = 0.0
            else:
                primary_subject_retention = horizontal_retention(
                    primary_subject.bounds,
                    crop_left,
                    crop_right,
                )

        focal_region = perception.focal.primary_region

        if focal_region is None:
            focal_retention = 0.0
        else:
            focal_retention = horizontal_retention(
                focal_region.bounds,
                crop_left,
                crop_right,
            )

        score = (
            self.config.subject_weight * retained_subject_ratio
            + self.config.primary_subject_weight * primary_subject_retention
            + self.config.focal_weight * focal_retention
        )

        return CropSuitabilityEvidence(
            score=score,
            retained_subject_ratio=retained_subject_ratio,
            primary_subject_retention=primary_subject_retention,
            focal_retention=focal_retention,
            subject_count=len(subjects),
        )

    @staticmethod
    def _read_dimensions(path: Path) -> tuple[int, int]:
        from PIL import Image

        with Image.open(path) as image:
            return image.size