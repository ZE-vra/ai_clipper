from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

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
    max_semantic_text_overlap: float = 0.08
    semantic_sample_width: int = 96
    semantic_sample_height: int = 160
    head_region_ratio: float = 0.30
    important_region_clearance: float = 0.035

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
        if not 0.0 <= self.max_semantic_text_overlap <= 1.0:
            raise ValueError("max_semantic_text_overlap must be between 0 and 1.")
        if self.semantic_sample_width <= 0 or self.semantic_sample_height <= 0:
            raise ValueError("semantic sample dimensions must be positive.")
        if not 0.15 <= self.head_region_ratio <= 0.5:
            raise ValueError("head_region_ratio must be between 0.15 and 0.5.")
        if self.important_region_clearance < 0:
            raise ValueError("important_region_clearance must not be negative.")
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
            asset=asset,
            crop_bounds=crop_bounds,
            target_width=target.size.width,
            target_height=target.size.height,
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
        asset: VisualAsset,
        crop_bounds: BoundingBox,
        target_width: int,
        target_height: int,
    ) -> BoundingBox:
        """
        Select the best usable text field.

        Semantic occupancy is a ranking signal, not a hard candidate-generation
        veto. This is important for crowded frames: the system should choose
        the least intrusive viable region rather than produce zero layouts.
        """
        candidates = self._candidate_text_regions(
            transformed_subject=transformed_subject,
            target=target,
        )
        if not candidates:
            raise ValueError("composition produced no geometric typography regions.")

        exclusion = self._expand(
            transformed_subject,
            self.config.text_subject_clearance,
        )

        scored: list[tuple[float, BoundingBox]] = []
        for region in candidates:
            semantic = self._semantic_overlap(
                region,
                asset=asset,
                crop_bounds=crop_bounds,
                target_width=target_width,
                target_height=target_height,
            )
            subject_overlap = self._overlap(region, exclusion)
            important_overlap = self._important_region_overlap(
                region,
                transformed_subject,
            )
            ui_overlap = self._ui_overlap(region, target)

            # A person is not a uniform blob. The upper portion containing
            # the face/head/cap is a protected visual region. Text crossing
            # an arm can sometimes work; text crossing the head almost always
            # creates the exact collision we are trying to eliminate.
            score = (
                semantic * 12.0
                + subject_overlap * 20.0
                + important_overlap * 70.0
                + ui_overlap * 40.0
                - min(1.0, region.width / 0.70) * 1.5
            )

            center_x = (region.left + region.right) / 2.0
            if self.config.preferred_subject_side == "right" and center_x > 0.55:
                score += 2.0
            elif self.config.preferred_subject_side == "left" and center_x < 0.45:
                score += 2.0

            scored.append((score, region))

        scored.sort(key=lambda item: item[0])
        return scored[0][1]

    def _candidate_text_regions(
        self,
        *,
        transformed_subject: BoundingBox,
        target: ThumbnailTarget,
    ) -> list[BoundingBox]:
        margin = self.config.text_band_margin
        exclusion = self._expand(
            transformed_subject,
            self.config.text_subject_clearance,
        )

        # Keep multiple geometrically viable fields. Semantic occupancy is
        # scored later; it must not erase every candidate on a crowded frame.
        candidates = [
            BoundingBox(margin, margin, 1.0 - margin, 0.27),
            BoundingBox(margin, 0.10, 1.0 - margin, 0.37),
            BoundingBox(margin, 0.04, 1.0 - margin, 0.22),
            BoundingBox(margin, 0.51, 1.0 - margin, 0.78),
            BoundingBox(margin, 0.59, 1.0 - margin, 0.78),
            BoundingBox(margin, 0.63, 1.0 - margin, 0.82),
        ]

        left_width = exclusion.left - margin
        right_width = (1.0 - margin) - exclusion.right

        # Preserve a side field even when it is narrower than the general
        # full-height layout threshold. A constrained but subject-free field
        # is preferable to silently falling back to a full-width band.
        minimum_side_field_width = 0.20
        if (
            left_width >= minimum_side_field_width
            and self.config.preferred_subject_side != "right"
        ):
            candidates.append(
                BoundingBox(margin, 0.10, exclusion.left - margin, 0.78)
            )

        if (
            right_width >= minimum_side_field_width
            and self.config.preferred_subject_side != "left"
        ):
            candidates.append(
                BoundingBox(exclusion.right + margin, 0.10, 1.0 - margin, 0.78)
            )

        # When the subject is deliberately staged to one side, provide an
        # opposing text field in the requested top/bottom band. Do not also
        # offer a full-height version of that same field: the band preference
        # should be honored when the opposing field is available.
        if (
            self.config.preferred_subject_side == "right"
            and left_width >= minimum_side_field_width
        ):
            if self.config.preferred_text_side == "top":
                candidates.append(
                    BoundingBox(margin, margin, exclusion.left - margin, 0.44)
                )
            else:
                candidates.append(
                    BoundingBox(
                        margin, 0.55, exclusion.left - margin, 1.0 - margin
                    )
                )
        elif (
            self.config.preferred_subject_side == "left"
            and right_width >= minimum_side_field_width
        ):
            if self.config.preferred_text_side == "top":
                candidates.append(
                    BoundingBox(
                        exclusion.right + margin, margin, 1.0 - margin, 0.44
                    )
                )
            else:
                candidates.append(
                    BoundingBox(
                        exclusion.right + margin, 0.55, 1.0 - margin, 1.0 - margin
                    )
                )

        safe_candidates: list[BoundingBox] = []
        for candidate in candidates:
            intersections = [
                intersection
                for safe in target.safe_regions
                if (intersection := self._intersection(candidate, safe.bounds)) is not None
            ]
            if intersections:
                safe_candidates.append(
                    max(intersections, key=lambda region: region.width * region.height)
                )

        unique: dict[tuple[float, float, float, float], BoundingBox] = {}
        for candidate in safe_candidates:
            if candidate.width <= 0 or candidate.height <= 0:
                continue
            key = (
                round(candidate.left, 4),
                round(candidate.top, 4),
                round(candidate.right, 4),
                round(candidate.bottom, 4),
            )
            unique[key] = candidate

        return list(unique.values())

    def _semantic_overlap(
        self,
        region: BoundingBox,
        *,
        asset: VisualAsset,
        crop_bounds: BoundingBox,
        target_width: int,
        target_height: int,
    ) -> float:
        """Measure actual segmented-subject occupancy inside a canvas region."""
        if not asset.subject_mask_path:
            return 0.0

        mask_path = Path(asset.subject_mask_path)
        if not mask_path.is_file():
            return 0.0

        with Image.open(mask_path) as source:
            mask = source.convert("L")

        mask = self._crop_image(mask, crop_bounds)
        mask = self._cover(mask, target_width, target_height)

        left = max(0, min(target_width, round(region.left * target_width)))
        top = max(0, min(target_height, round(region.top * target_height)))
        right = max(left + 1, min(target_width, round(region.right * target_width)))
        bottom = max(top + 1, min(target_height, round(region.bottom * target_height)))

        sample = mask.crop((left, top, right, bottom)).resize(
            (self.config.semantic_sample_width, self.config.semantic_sample_height),
            Image.Resampling.BILINEAR,
        )
        pixels = list(sample.getdata())
        if not pixels:
            return 0.0
        return sum(1 for value in pixels if value >= 64) / len(pixels)

    @staticmethod
    def _crop_image(image: Image.Image, bounds: BoundingBox) -> Image.Image:
        width, height = image.size
        left = max(0, min(round(bounds.left * width), width - 1))
        top = max(0, min(round(bounds.top * height), height - 1))
        right = max(left + 1, min(round(bounds.right * width), width))
        bottom = max(top + 1, min(round(bounds.bottom * height), height))
        return image.crop((left, top, right, bottom))

    @staticmethod
    def _cover(image: Image.Image, width: int, height: int) -> Image.Image:
        source_width, source_height = image.size
        source_ratio = source_width / source_height
        target_ratio = width / height

        if source_ratio > target_ratio:
            new_height = height
            new_width = round(height * source_ratio)
        else:
            new_width = width
            new_height = round(width / source_ratio)

        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        left = max(0, (new_width - width) // 2)
        top = max(0, (new_height - height) // 2)
        return resized.crop((left, top, left + width, top + height))

    def _important_region_overlap(
        self,
        region: BoundingBox,
        subject: BoundingBox,
    ) -> float:
        """Return the strongest overlap with the subject's protected head zone."""
        head_bottom = min(
            1.0,
            subject.top + subject.height * self.config.head_region_ratio,
        )
        head = BoundingBox(
            left=subject.left,
            top=max(0.0, subject.top - self.config.important_region_clearance),
            right=subject.right,
            bottom=min(1.0, head_bottom + self.config.important_region_clearance),
        )
        return self._overlap(region, head)

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
