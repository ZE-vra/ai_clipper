from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from src.thumbnails.domain.assets import VisualAsset
from src.thumbnails.domain.concepts import (
    CopyBlock,
)
from src.thumbnails.domain.geometry import (
    BoundingBox,
    Point,
)


@dataclass(frozen=True)
class VisualPlacement:
    """Placement of a visual asset on the canvas."""

    asset_id: str
    focal_point: Point

    scale: float = 1.0
    center: Point = field(
        default_factory=lambda: Point(
            x=0.5,
            y=0.5,
        )
    )

    crop_bounds: Optional[BoundingBox] = None

    def __post_init__(self) -> None:
        if not self.asset_id.strip():
            raise ValueError(
                "asset_id must not be blank."
            )

        if self.scale <= 0:
            raise ValueError(
                "scale must be greater than 0."
            )


@dataclass(frozen=True)
class CompositionPlan:
    """Complete spatial arrangement of thumbnail visuals."""

    visual_placements: Tuple[
        VisualPlacement, ...
    ]

    negative_space_regions: Tuple[
        BoundingBox, ...
    ] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.visual_placements:
            raise ValueError(
                "visual_placements must contain "
                "at least one item."
            )


@dataclass(frozen=True)
class TypographyBlockPlan:
    """Visual treatment for one copy block."""

    copy: CopyBlock

    font_name: str
    font_size: int
    weight: str

    alignment: str
    color: str

    position: Point
    text_bounds: BoundingBox

    max_lines: int = 2
    line_spacing: float = 0.92
    rendered_text: Optional[str] = None

    stroke_color: Optional[str] = None
    stroke_width: int = 0

    def __post_init__(self) -> None:
        if not self.font_name.strip():
            raise ValueError(
                "font_name must not be blank."
            )

        if self.font_size <= 0:
            raise ValueError(
                "font_size must be greater than 0."
            )

        if not self.weight.strip():
            raise ValueError(
                "weight must not be blank."
            )

        if self.alignment not in {
            "left",
            "center",
            "right",
        }:
            raise ValueError(
                "alignment must be left, center, or right."
            )

        if not self.color.strip():
            raise ValueError(
                "color must not be blank."
            )

        if self.max_lines <= 0:
            raise ValueError(
                "max_lines must be greater than 0."
            )

        if self.line_spacing <= 0:
            raise ValueError(
                "line_spacing must be greater than 0."
            )

        if self.rendered_text is not None and not self.rendered_text.strip():
            raise ValueError(
                "rendered_text must not be blank when provided."
            )

        if self.stroke_width < 0:
            raise ValueError(
                "stroke_width must not be negative."
            )


@dataclass(frozen=True)
class TypographyPlan:
    """Complete typography decision."""

    blocks: Tuple[
        TypographyBlockPlan, ...
    ]

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError(
                "blocks must contain at least one item."
            )


@dataclass(frozen=True)
class VisualTreatmentPlan:
    """Image-processing instructions for the renderer."""

    contrast: float = 1.0
    saturation: float = 1.0
    sharpness: float = 1.0
    vignette: float = 0.0
    overlay_opacity: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("contrast", self.contrast),
            ("saturation", self.saturation),
            ("sharpness", self.sharpness),
        ):
            if value < 0:
                raise ValueError(
                    f"{name} must not be negative."
                )

        for name, value in (
            ("vignette", self.vignette),
            ("overlay_opacity", self.overlay_opacity),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1."
                )


@dataclass(frozen=True)
class ThumbnailRenderPlan:
    """
    The complete deterministic instruction set consumed
    by a thumbnail renderer.
    """

    canvas_width: int
    canvas_height: int

    composition: CompositionPlan
    typography: TypographyPlan
    visual_treatment: VisualTreatmentPlan

    background_asset: Optional[VisualAsset] = None

    def __post_init__(self) -> None:
        if self.canvas_width <= 0:
            raise ValueError(
                "canvas_width must be greater than 0."
            )

        if self.canvas_height <= 0:
            raise ValueError(
                "canvas_height must be greater than 0."
            )


@dataclass(frozen=True)
class AssetPlan:
    """Executable visual-asset construction recipe for a concept."""

    plan_id: str
    concept_id: str
    strategy: "VisualStrategy"
    source_asset_ids: tuple[str, ...] = ()
    operations: tuple["AssetOperation", ...] = ()
    requirements: tuple[str, ...] = ()
    rationale: str = ""
    estimated_cost: float = 0.0
    plan_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.plan_id.strip():
            raise ValueError("plan_id must not be blank.")
        if not self.concept_id.strip():
            raise ValueError("concept_id must not be blank.")
        if self.estimated_cost < 0:
            raise ValueError("estimated_cost must not be negative.")
