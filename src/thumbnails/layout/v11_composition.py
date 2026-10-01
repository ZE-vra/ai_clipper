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
    text_subject_clearance: float = 0.03
    preferred_text_side: str = "top"
    preferred_subject_side: str = "center"

    def __post_init__(self) -> None:
        if not 0.5 <= self.subject_retention <= 1.0:
            raise ValueError("subject_retention must be between 0.5 and 1.")
        if not 0.15 <= self.preferred_text_band_height <= 0.5:
            raise ValueError("preferred_text_band_height is out of range.")
        if not 0.0 <= self.text_band_margin < 0.2:
            raise ValueError("text_band_margin must be between 0 and 0.2.")
        if self.subject_clearance < 0:
            raise ValueError("subject_clearance must not be negative.")
        if self.text_subject_clearance < 0:
            raise ValueError("text_subject_clearance must not be negative.")
        if self.preferred_text_side not in {"top", "bottom"}:
            raise ValueError("preferred_text_side must be 'top' or 'bottom'.")
        if self.preferred_subject_side not in {"left", "center", "right"}:
            raise ValueError(
                "preferred_subject_side must be 'left', 'center', or 'right'."
            )


class V11CompositionPlanner:
    """
    Builds a genuinely vertical composition from a clean source frame.

    V1.1 originally treated the crop as fixed around the detected subject and
    then asked typography to find somewhere to sit. That makes a centered
    person and a full-width text band collide by construction.

    This planner now treats subject placement as a composition decision:
    when a side is requested, the crop is shifted so the subject occupies that
    side of the canvas and the opposite side becomes the text field.
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
            crop_bounds = BoundingBox(0.0, 0.0, 1.0, 1.0)
            transformed_subject = BoundingBox(0.35, 0.20, 0.65, 0.80)
            transformed_focal = Point(0.5, 0.5)
        else:
            crop_bounds = self._crop_window(
                source_aspect_ratio=source_aspect_ratio,
                target_aspect_ratio=target_ratio,
                subject=subject.bounds,
                center_x=subject.focal_point.x,
                text_side=self.config.preferred_text_side,
                subject_side=self.config.preferred_subject_side,
            )
            transformed_subject = self._transform_bounds(
                subject.bounds,
                crop_bounds,
            )
            transformed_focal = Point(
                x=self._transform_x(subject.focal_point.x, crop_bounds),
                y=self._transform_y(subject.focal_point.y, crop_bounds),
            )

        placement = VisualPlacement(
            asset_id=asset.asset_id,
            focal_point=transformed_focal,
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
        text_side: str = "top",
        subject_side: str = "center",
    ) -> BoundingBox:
        if source_aspect_ratio <= target_aspect_ratio:
            return BoundingBox(0.0, 0.0, 1.0, 1.0)

        base_left, base_right = calculate_horizontal_crop_bounds(
            source_aspect_ratio=source_aspect_ratio,
            target_aspect_ratio=target_aspect_ratio,
            center_x=center_x,
        )
        crop_width = base_right - base_left

        required_height = max(
            subject.height / self.config.subject_retention,
            subject.height + 2.0 * self.config.subject_clearance,
            0.72,
        )
        required_width = (
            (
                subject.width / self.config.subject_retention
                + 2.0 * self.config.subject_clearance
            )
            * source_aspect_ratio
            / target_aspect_ratio
        )
        crop_height = min(1.0, max(required_height, required_width))

        if crop_height >= 0.98:
            crop_height = 1.0

        required_crop_width = (
            crop_height * target_aspect_ratio / source_aspect_ratio
        )
        if subject_side == "center":
            if required_crop_width < crop_width:
                crop_width = required_crop_width
        else:
            # Side staging needs enough horizontal room to move the subject
            # without clipping its protected bounds. The base 9:16 crop is
            # the minimum useful width for that job, so do not shrink it just
            # because the subject itself could fit in a tighter crop.
            crop_height = 1.0

        # Decide where the subject should land in the output. This is the
        # critical V1.1 change: the image is staged for the text instead of
        # simply cropped around the person.
        desired_x = {
            "left": 0.32,
            "center": 0.50,
            "right": 0.68,
        }[subject_side]
        desired_y = {
            "top": 0.68,
            "bottom": 0.32,
        }[text_side]

        subject_center_x = center_x
        subject_center_y = (subject.top + subject.bottom) / 2.0
        desired_left = subject_center_x - desired_x * crop_width
        min_left = max(
            0.0,
            subject.right
            + self.config.subject_clearance
            - crop_width,
        )
        max_left = min(
            1.0 - crop_width,
            subject.left - self.config.subject_clearance,
        )

        if min_left <= max_left:
            left = min(
                max(desired_left, min_left),
                max_left,
            )
        else:
            # The subject is too wide to honor the requested horizontal
            # staging. Preserve it rather than manufacturing a clipped crop.
            left = min(
                max(0.0, base_left),
                1.0 - crop_width,
            )

        right = left + crop_width

        min_top = max(
            0.0,
            subject.bottom + self.config.subject_clearance - crop_height,
        )
        max_top = min(
            1.0 - crop_height,
            subject.top - self.config.subject_clearance,
        )

        if min_top > max_top:
            return BoundingBox(
                left=left,
                top=0.0,
                right=right,
                bottom=1.0,
            )

        desired_top = subject_center_y - desired_y * crop_height
        top = min(max(desired_top, min_top), max_top)
        bottom = top + crop_height

        return BoundingBox(left, top, right, bottom)

    @staticmethod
    def _transform_x(x: float, crop: BoundingBox) -> float:
        return max(0.0, min(1.0, (x - crop.left) / crop.width))

    @staticmethod
    def _transform_y(y: float, crop: BoundingBox) -> float:
        return max(0.0, min(1.0, (y - crop.top) / crop.height))

    @classmethod
    def _transform_bounds(
        cls,
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

        exclusion = self._expand(
            transformed_subject,
            self.config.text_subject_clearance,
        )

        if self.config.preferred_subject_side == "right":
            horizontal = BoundingBox(
                margin,
                0.0,
                max(margin, exclusion.left - self.config.text_subject_clearance),
                1.0,
            )
        elif self.config.preferred_subject_side == "left":
            horizontal = BoundingBox(
                min(1.0 - margin, exclusion.right + self.config.text_subject_clearance),
                0.0,
                1.0 - margin,
                1.0,
            )
        else:
            horizontal = BoundingBox(
                margin,
                0.0,
                1.0 - margin,
                1.0,
            )

        if self.config.preferred_text_side == "top":
            vertical = BoundingBox(
                0.0,
                margin,
                1.0,
                min(1.0 - margin, margin + band_height),
            )
        else:
            vertical = BoundingBox(
                0.0,
                max(margin, 1.0 - margin - band_height),
                1.0,
                1.0 - margin,
            )

        candidate = self._intersection(horizontal, vertical)

        if candidate is None:
            # Preserve the old full-width band as a safe fallback. The
            # typography planner and negotiator can reject it if necessary.
            candidate = (
                vertical
            )

        if self._overlap(candidate, exclusion) > 0.0:
            alternatives = self._candidate_text_regions(
                transformed_subject=transformed_subject,
                target=target,
            )
            if alternatives:
                return min(
                    alternatives,
                    key=lambda region: (
                        self._overlap(region, exclusion) * 10.0
                        + self._ui_overlap(region, target),
                        -region.width * region.height,
                    ),
                )

        return candidate

    def _candidate_text_regions(
        self,
        *,
        transformed_subject: BoundingBox,
        target: ThumbnailTarget,
    ) -> list[BoundingBox]:
        margin = self.config.text_band_margin
        band_height = self.config.preferred_text_band_height
        exclusion = self._expand(
            transformed_subject,
            self.config.text_subject_clearance,
        )

        candidates: list[BoundingBox] = []
        for top in (True, False):
            vertical = (
                BoundingBox(
                    margin,
                    margin,
                    1.0 - margin,
                    min(1.0 - margin, margin + band_height),
                )
                if top
                else BoundingBox(
                    margin,
                    max(margin, 1.0 - margin - band_height),
                    1.0 - margin,
                    1.0 - margin,
                )
            )

            horizontal_regions: list[BoundingBox] = [
                BoundingBox(margin, 0.0, 1.0 - margin, 1.0),
            ]
            left_right = exclusion.left - margin
            if left_right > margin:
                horizontal_regions.append(
                    BoundingBox(margin, 0.0, left_right, 1.0)
                )

            right_left = exclusion.right + margin
            if right_left < 1.0 - margin:
                horizontal_regions.append(
                    BoundingBox(right_left, 0.0, 1.0 - margin, 1.0)
                )

            for horizontal in horizontal_regions:
                intersection = self._intersection(vertical, horizontal)
                if intersection is not None and intersection.width > 0:
                    candidates.append(intersection)

        return [
            candidate
            for candidate in candidates
            if self._overlap(candidate, exclusion) == 0.0
        ]

    @staticmethod
    def _expand(bounds: BoundingBox, margin: float) -> BoundingBox:
        return BoundingBox(
            left=max(0.0, bounds.left - margin),
            top=max(0.0, bounds.top - margin),
            right=min(1.0, bounds.right + margin),
            bottom=min(1.0, bounds.bottom + margin),
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
