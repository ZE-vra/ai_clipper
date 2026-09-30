from __future__ import annotations

from dataclasses import dataclass

from src.thumbnails.domain.assets import VisualAsset
from src.thumbnails.domain.geometry import BoundingBox, Point
from src.thumbnails.domain.plans import CompositionPlan, VisualPlacement
from src.thumbnails.domain.target import ThumbnailTarget
from src.thumbnails.perception.frame_selector import SelectedFrame


@dataclass(frozen=True)
class CompositionPlannerConfig:
    """Deterministic visual-composition policy for the first thumbnail pass."""

    negative_space_width: float = 0.42
    negative_space_top: float = 0.10
    negative_space_bottom: float = 0.90
    ui_overlap_tolerance: float = 0.15

    def __post_init__(self) -> None:
        if not 0.0 < self.negative_space_width <= 1.0:
            raise ValueError("negative_space_width must be between 0 and 1.")
        if not 0.0 <= self.negative_space_top < self.negative_space_bottom <= 1.0:
            raise ValueError(
                "negative_space_top and negative_space_bottom must form a valid normalized range."
            )
        if not 0.0 <= self.ui_overlap_tolerance <= 1.0:
            raise ValueError("ui_overlap_tolerance must be between 0 and 1.")


class CompositionPlanner:
    """
    Turns frame perception into a deterministic visual composition.

    V1 deliberately solves only the image composition problem: where the
    selected source frame sits, what part of it is cropped, and where useful
    negative space exists. Typography is planned separately.
    """

    def __init__(
        self,
        config: CompositionPlannerConfig | None = None,
    ) -> None:
        self.config = config or CompositionPlannerConfig()

    def plan(
        self,
        *,
        selected_frame: SelectedFrame,
        asset: VisualAsset,
        target: ThumbnailTarget,
        source_aspect_ratio: float,
    ) -> CompositionPlan:
        if source_aspect_ratio <= 0.0:
            raise ValueError("source_aspect_ratio must be greater than 0.")

        primary_subject = selected_frame.perception.subjects.primary_subject
        focal_point = (
            primary_subject.focal_point
            if primary_subject is not None
            else Point(x=0.5, y=0.5)
        )

        target_aspect_ratio = target.size.width / target.size.height
        crop_bounds = self._calculate_crop_bounds(
            source_aspect_ratio=source_aspect_ratio,
            target_aspect_ratio=target_aspect_ratio,
            center_x=focal_point.x,
        )

        crop_center = Point(
            x=(crop_bounds.left + crop_bounds.right) / 2.0,
            y=0.5,
        )

        placement = VisualPlacement(
            asset_id=asset.asset_id,
            focal_point=focal_point,
            scale=1.0,
            center=crop_center,
            crop_bounds=crop_bounds,
        )

        negative_space = self._select_negative_space(
            focal_point=focal_point,
            target=target,
        )

        return CompositionPlan(
            visual_placements=(placement,),
            negative_space_regions=(negative_space,),
        )

    @staticmethod
    def _calculate_crop_bounds(
        *,
        source_aspect_ratio: float,
        target_aspect_ratio: float,
        center_x: float,
    ) -> BoundingBox:
        if source_aspect_ratio <= target_aspect_ratio:
            return BoundingBox(
                left=0.0,
                top=0.0,
                right=1.0,
                bottom=1.0,
            )

        crop_width = target_aspect_ratio / source_aspect_ratio
        half_width = crop_width / 2.0
        left = max(0.0, min(1.0 - crop_width, center_x - half_width))

        return BoundingBox(
            left=left,
            top=0.0,
            right=left + crop_width,
            bottom=1.0,
        )

    def _select_negative_space(
        self,
        *,
        focal_point: Point,
        target: ThumbnailTarget,
    ) -> BoundingBox:
        left = self._side_region("left")
        right = self._side_region("right")

        preferred = right if focal_point.x < 0.5 else left
        alternate = left if preferred is right else right

        if self._weighted_ui_overlap(preferred, target) <= self.config.ui_overlap_tolerance:
            return preferred
        return alternate

    def _side_region(self, side: str) -> BoundingBox:
        width = self.config.negative_space_width
        if side == "left":
            return BoundingBox(
                left=0.0,
                top=self.config.negative_space_top,
                right=width,
                bottom=self.config.negative_space_bottom,
            )

        return BoundingBox(
            left=1.0 - width,
            top=self.config.negative_space_top,
            right=1.0,
            bottom=self.config.negative_space_bottom,
        )

    @staticmethod
    def _intersection_area(a: BoundingBox, b: BoundingBox) -> float:
        width = max(0.0, min(a.right, b.right) - max(a.left, b.left))
        height = max(0.0, min(a.bottom, b.bottom) - max(a.top, b.top))
        return width * height

    def _weighted_ui_overlap(
        self,
        region: BoundingBox,
        target: ThumbnailTarget,
    ) -> float:
        total = 0.0
        for occlusion in target.ui_occlusion_regions:
            overlap = self._intersection_area(region, occlusion.region.bounds)
            total += overlap * occlusion.weight
        return min(1.0, total / (region.width * region.height))
