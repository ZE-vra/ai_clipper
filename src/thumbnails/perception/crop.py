from __future__ import annotations

from dataclasses import dataclass

from src.thumbnails.domain.geometry import BoundingBox


@dataclass(frozen=True)
class CropSuitabilityEvidence:
    score: float
    retained_subject_ratio: float
    primary_subject_retention: float
    focal_retention: float
    subject_count: int

    def __post_init__(self) -> None:
        values = (
            ("score", self.score),
            ("retained_subject_ratio", self.retained_subject_ratio),
            ("primary_subject_retention", self.primary_subject_retention),
            ("focal_retention", self.focal_retention),
        )

        for name, value in values:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")

        if self.subject_count < 0:
            raise ValueError("subject_count must not be negative.")


@dataclass(frozen=True)
class CropAnalyzerConfig:
    target_width: int = 1080
    target_height: int = 1920
    minimum_retention: float = 0.50
    subject_weight: float = 0.25
    primary_subject_weight: float = 0.50
    focal_weight: float = 0.25

    def __post_init__(self) -> None:
        if self.target_width <= 0:
            raise ValueError("target_width must be greater than 0.")

        if self.target_height <= 0:
            raise ValueError("target_height must be greater than 0.")

        if not 0.0 <= self.minimum_retention <= 1.0:
            raise ValueError("minimum_retention must be between 0 and 1.")

        weights = (
            self.subject_weight,
            self.primary_subject_weight,
            self.focal_weight,
        )

        if any(weight < 0.0 for weight in weights):
            raise ValueError("crop weights must not be negative.")

        if sum(weights) <= 0.0:
            raise ValueError("at least one crop weight must be greater than 0.")

    @property
    def target_aspect_ratio(self) -> float:
        return self.target_width / self.target_height


def calculate_horizontal_crop_bounds(
    *,
    source_aspect_ratio: float,
    target_aspect_ratio: float,
    center_x: float,
) -> tuple[float, float]:
    if source_aspect_ratio <= 0.0:
        raise ValueError("source_aspect_ratio must be greater than 0.")

    if target_aspect_ratio <= 0.0:
        raise ValueError("target_aspect_ratio must be greater than 0.")

    if not 0.0 <= center_x <= 1.0:
        raise ValueError("center_x must be between 0 and 1.")

    if source_aspect_ratio <= target_aspect_ratio:
        return (0.0, 1.0)

    crop_width = target_aspect_ratio / source_aspect_ratio
    half_width = crop_width / 2.0

    left = max(0.0, center_x - half_width)
    right = min(1.0, center_x + half_width)

    if right - left < crop_width:
        if left == 0.0:
            right = crop_width
        elif right == 1.0:
            left = 1.0 - crop_width

    return (left, right)


def horizontal_retention(
    bounds: BoundingBox,
    crop_left: float,
    crop_right: float,
) -> float:
    intersection_left = max(bounds.left, crop_left)
    intersection_right = min(bounds.right, crop_right)

    if intersection_right <= intersection_left:
        return 0.0

    intersection_width = intersection_right - intersection_left

    if bounds.width <= 0.0:
        return 0.0

    return min(1.0, intersection_width / bounds.width)