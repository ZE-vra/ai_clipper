from __future__ import annotations

from dataclasses import dataclass

from src.thumbnails.domain.assets import VisualAsset
from src.thumbnails.domain.geometry import BoundingBox, Point
from src.thumbnails.domain.plans import CompositionPlan, VisualPlacement
from src.thumbnails.domain.target import ThumbnailTarget
from src.thumbnails.perception.frame_selector import SelectedFrame
from src.thumbnails.perception.crop import calculate_horizontal_crop_bounds


@dataclass(frozen=True)
class V11CompositionConfig:
    """Subject-first composition policy for vertical thumbnail covers."""

    subject_retention: float = 0.92
    preferred_text_band_height: float = 0.27
    text_band_margin: float = 0.04
    subject_clearance: float = 0.04

    def __post_init__(self) -> None:
        if not 0.5 <= self.subject_retention <= 1.0:
            raise ValueError("subject_retention must be between 0.5 and 1.")
        if not 0.15 <= self.preferred_text_band_height <= 0.5:
            raise ValueError("preferred_text_band_height is out of range.")
        if not 0.0 <= self.text_band_margin < 0.2:
            raise ValueError("text_band_margin must be between 0 and 0.2.")
        if self.subject_clearance < 0:
            raise ValueError("subject_clearance must not be negative.")


class V11CompositionPlanner:
    """
    Builds a genuinely vertical composition from a clean source frame.

    Unlike the original planner, V1.1 treats the subject as the visual hero,
    computes the subject's position after cropping, and reserves a dedicated
    text band rather than placing text in an arbitrary side rectangle.
    """

    def __init__(self, config: V11CompositionConfig | None = None) -> None:
        self.config = config or V11CompositionConfig()

    def plan(
        self,
        *,
        selected_frame: SelectedFrame,
        asset: VisualAsset,
        target: ThumbnailTarget,
        source_aspect_ratio: float,
    ) -> CompositionPlan:
        if source_aspect_ratio <= 0:
            raise ValueError("source_aspect_ratio must be greater than 0.")

        target_ratio = target.size.width / target.size.height
        subject = selected_frame.perception.subjects.primary_subject

        if subject is None:
            center_x = 0.5
            crop_bounds = BoundingBox(0.0, 0.0, 1.0, 1.0)
            transformed_subject = BoundingBox(0.35, 0.20, 0.65, 0.80)
        else:
            center_x = subject.focal_point.x
            crop_bounds = self._crop_window(
                source_aspect_ratio=source_aspect_ratio,
                target_aspect_ratio=target_ratio,
                subject=subject.bounds,
                center_x=center_x,
            )
            transformed_subject = self._transform_bounds(
                subject.bounds,
                crop_bounds,
            )

        placement = VisualPlacement(
            asset_id=asset.asset_id,
            focal_point=Point(
                x=self._transform_x(center_x, crop_bounds),
                y=(
                    subject.focal_point.y
                    if subject is not None
                    else 0.5
                ),
            ),
            scale=1.0,
            center=Point(
                x=(crop_bounds.left + crop_bounds.right) / 2.0,
                y=(crop_bounds.top + crop_bounds.bottom) / 2.0,
            ),
            crop_bounds=crop_bounds,
        )

        text_region = self._choose_text_region(
            transformed_subject=transformed_subject,
            target=target,
        )

        return CompositionPlan(
            visual_placements=(placement,),
            negative_space_regions=(text_region,),
        )

    def _crop_window(
        self,
        *,
        source_aspect_ratio: float,
        target_aspect_ratio: float,
        subject: BoundingBox,
        center_x: float,
    ) -> BoundingBox:
        if source_aspect_ratio <= target_aspect_ratio:
            return BoundingBox(0.0, 0.0, 1.0, 1.0)

        # A full-height 9:16 crop is already a very aggressive crop from
        # landscape footage. Only reduce height when the primary subject
        # leaves enough breathing room to do so safely.
        crop_left, crop_right = calculate_horizontal_crop_bounds(
            source_aspect_ratio=source_aspect_ratio,
            target_aspect_ratio=target_aspect_ratio,
            center_x=center_x,
        )

        crop_width = crop_right - crop_left
        required_height_for_subject = (
            subject.height / self.config.subject_retention
            + 2 * self.config.subject_clearance
        )
        required_height_for_width = (
            (
                subject.width / self.config.subject_retention
                + 2 * self.config.subject_clearance
            )
            * source_aspect_ratio
            / target_aspect_ratio
        )

        max_zoom_height = min(
            1.0,
            max(
                required_height_for_subject,
                required_height_for_width,
                0.72,
            ),
        )

        if max_zoom_height >= 0.98:
            return BoundingBox(
                left=crop_left,
                top=0.0,
                right=crop_right,
                bottom=1.0,
            )

        crop_height = max_zoom_height
        required_width = crop_height * target_aspect_ratio / source_aspect_ratio

        if required_width < crop_width:
            crop_width = required_width

        left = min(
            max(0.0, center_x - crop_width / 2.0),
            1.0 - crop_width,
        )
        right = left + crop_width

        top = min(
            max(0.0, subject.focal_point.y - crop_height / 2.0),
            1.0 - crop_height,
        )
        bottom = top + crop_height

        return BoundingBox(left, top, right, bottom)

    @staticmethod
    def _transform_x(x: float, crop: BoundingBox) -> float:
        return max(0.0, min(1.0, (x - crop.left) / crop.width))

    @staticmethod
    def _transform_bounds(
        bounds: BoundingBox,
        crop: BoundingBox,
    ) -> BoundingBox:
        return BoundingBox(
            left=max(0.0, min(1.0, (bounds.left - crop.left) / crop.width)),
            top=max(0.0, min(1.0, (bounds.top - crop.top) / crop.height)),
            right=max(0.0, min(1.0, (bounds.right - crop.left) / crop.width)),
            bottom=max(0.0, min(1.0, (bounds.bottom - crop.top) / crop.height)),
        )

    def _choose_text_region(
        self,
        *,
        transformed_subject: BoundingBox,
        target: ThumbnailTarget,
    ) -> BoundingBox:
        margin = self.config.text_band_margin
        band_height = self.config.preferred_text_band_height

        top_candidate = BoundingBox(
            margin,
            margin,
            1.0 - margin,
            min(1.0 - margin, margin + band_height),
        )
        bottom_candidate = BoundingBox(
            margin,
            max(margin, 1.0 - margin - band_height),
            1.0 - margin,
            1.0 - margin,
        )

        candidates = [top_candidate, bottom_candidate]

        valid: list[BoundingBox] = []
        for candidate in candidates:
            overlap = self._overlap(candidate, transformed_subject)
            if overlap <= 0.08 and self._ui_overlap(candidate, target) <= 0.05:
                valid.append(candidate)

        if valid:
            # Prefer the upper band because it survives Shorts UI better and
            # keeps the subject's face unobstructed.
            return valid[0]

        return min(
            candidates,
            key=lambda candidate: (
                self._overlap(candidate, transformed_subject)
                + self._ui_overlap(candidate, target)
            ),
        )

    @staticmethod
    def _intersection(a: BoundingBox, b: BoundingBox) -> BoundingBox | None:
        left, top = max(a.left, b.left), max(a.top, b.top)
        right, bottom = min(a.right, b.right), min(a.bottom, b.bottom)
        if left >= right or top >= bottom:
            return None
        return BoundingBox(left, top, right, bottom)

    @classmethod
    def _overlap(cls, a: BoundingBox, b: BoundingBox) -> float:
        intersection = cls._intersection(a, b)
        if intersection is None:
            return 0.0
        return (intersection.width * intersection.height) / (a.width * a.height)

    @classmethod
    def _ui_overlap(cls, region: BoundingBox, target: ThumbnailTarget) -> float:
        return min(
            1.0,
            sum(
                cls._overlap(region, occlusion.region.bounds) * occlusion.weight
                for occlusion in target.ui_occlusion_regions
            ),
        )
