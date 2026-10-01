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


class V11CompositionPlanner:
    """
    Builds a genuinely vertical composition from a clean source frame.

    The primary subject is treated as a protected visual region. Cropping must
    preserve the subject with breathing room, and text bands must stay outside
    an expanded subject exclusion zone whenever a viable band exists.
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
            transformed_focal = Point(0.5, 0.5)
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
    ) -> BoundingBox:
        if source_aspect_ratio <= target_aspect_ratio:
            return BoundingBox(0.0, 0.0, 1.0, 1.0)

        crop_left, crop_right = calculate_horizontal_crop_bounds(
            source_aspect_ratio=source_aspect_ratio,
            target_aspect_ratio=target_aspect_ratio,
            center_x=center_x,
        )
        crop_width = crop_right - crop_left

        # The crop height is constrained by the subject's actual bounds, not
        # merely its focal point. This prevents heads/hats or lower body parts
        # from being sliced by an otherwise "valid" zoom.
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
        required_height = max(
            required_height,
            required_width,
        )

        crop_height = min(1.0, required_height)

        # If the requested retention cannot geometrically contain the subject
        # with clearance, fall back to the full source height rather than
        # accepting a visibly clipped subject.
        if crop_height >= 0.98:
            return BoundingBox(
                left=crop_left,
                top=0.0,
                right=crop_right,
                bottom=1.0,
            )

        required_crop_width = crop_height * target_aspect_ratio / source_aspect_ratio
        if required_crop_width < crop_width:
            crop_width = required_crop_width

        left = min(
            max(0.0, center_x - crop_width / 2.0),
            1.0 - crop_width,
        )
        right = left + crop_width

        # Find every legal vertical window that contains the subject plus
        # clearance. Then choose the one closest to the subject focal point.
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
                left=crop_left,
                top=0.0,
                right=crop_right,
                bottom=1.0,
            )

        preferred_top = subject.top + subject.height / 2.0 - crop_height / 2.0
        top = min(max(preferred_top, min_top), max_top)
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

        candidates = (
            [top_candidate, bottom_candidate]
            if self.config.preferred_text_side == "top"
            else [bottom_candidate, top_candidate]
        )

        exclusion = self._expand(
            transformed_subject,
            self.config.text_subject_clearance,
        )

        valid: list[BoundingBox] = []
        for candidate in candidates:
            subject_overlap = self._overlap(candidate, exclusion)
            ui_overlap = self._ui_overlap(candidate, target)
            if subject_overlap == 0.0 and ui_overlap <= 0.05:
                valid.append(candidate)

        if valid:
            return valid[0]

        # If neither full band is viable, choose the candidate with the
        # greatest usable area after accounting for subject and UI conflicts.
        # TypographyPlanner will further trim the selected region around UI.
        return min(
            candidates,
            key=lambda candidate: (
                self._overlap(candidate, exclusion) * 3.0
                + self._ui_overlap(candidate, target),
                -candidate.width * candidate.height,
            ),
        )

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
