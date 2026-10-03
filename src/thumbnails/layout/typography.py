from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from src.thumbnails.domain.concepts import CopyBlock, CopyRole
from src.thumbnails.domain.geometry import BoundingBox, Point
from src.thumbnails.domain.plans import CompositionPlan, TypographyBlockPlan, TypographyPlan
from src.thumbnails.domain.target import ThumbnailTarget


@dataclass(frozen=True)
class TextMeasurement:
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width < 0 or self.height < 0:
            raise ValueError("measurement dimensions must not be negative.")


class TextMeasurer(Protocol):
    def measure(self, text: str, *, font_name: str, font_size: int, weight: str) -> TextMeasurement:
        ...


class HeuristicTextMeasurer:
    """Deterministic fallback metrics behind a replaceable measurement boundary."""

    def measure(self, text: str, *, font_name: str, font_size: int, weight: str) -> TextMeasurement:
        del font_name
        lines = text.splitlines() or [""]
        factor = 0.60 if weight.lower() in {"bold", "heavy", "black"} else 0.54
        width = max(
            (
                sum(font_size * (0.30 if c.isspace() else factor) for c in line)
                for line in lines
            ),
            default=0.0,
        )
        return TextMeasurement(width=width, height=len(lines) * font_size)


class PillowTextMeasurer:
    """Measure using the same font engine used by the Pillow renderer."""

    def __init__(self, fallback: TextMeasurer | None = None) -> None:
        self.fallback = fallback or HeuristicTextMeasurer()

    def measure(self, text: str, *, font_name: str, font_size: int, weight: str) -> TextMeasurement:
        from pathlib import Path
        from PIL import ImageFont

        candidates: list[Path] = []
        if font_name.lower() == "arial":
            if weight.lower() == "bold":
                candidates.extend((
                    Path("C:/Windows/Fonts/arialbd.ttf"),
                    Path("C:/Windows/Fonts/Arial_Bold.ttf"),
                ))
            else:
                candidates.append(Path("C:/Windows/Fonts/arial.ttf"))
        candidates.append(Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))

        for candidate in candidates:
            if candidate.exists():
                font = ImageFont.truetype(str(candidate), font_size)
                lines = text.splitlines() or [""]
                boxes = [font.getbbox(line) for line in lines]
                width = max((box[2] - box[0] for box in boxes), default=0.0)
                height = max((box[3] - box[1] for box in boxes), default=0.0)
                return TextMeasurement(width=width, height=height * len(lines))

        return self.fallback.measure(
            text,
            font_name=font_name,
            font_size=font_size,
            weight=weight,
        )


@dataclass(frozen=True)
class TypographyPlannerConfig:
    font_name: str = "Arial"
    weight: str = "bold"
    color: str = "#FFFFFF"
    stroke_color: str = "#000000"
    stroke_width: int = 4
    base_font_size: int = 96
    minimum_font_size: int = 42
    maximum_font_size: int = 132
    hook_scale: float = 0.90
    payoff_scale: float = 1.18
    context_scale: float = 0.62
    accent_scale: float = 0.52
    max_lines_per_block: int = 2
    line_spacing: float = 0.92
    block_gap: float = 0.025
    horizontal_padding: float = 0.025
    vertical_padding: float = 0.025
    ui_overlap_tolerance: float = 0.05
    minimum_region_width: float = 0.18
    minimum_region_height: float = 0.10

    def __post_init__(self) -> None:
        if not self.font_name.strip() or not self.weight.strip():
            raise ValueError("font_name and weight must not be blank.")
        if not self.color.strip() or not self.stroke_color.strip():
            raise ValueError("colors must not be blank.")
        if self.stroke_width < 0:
            raise ValueError("stroke_width must not be negative.")
        if not 0 < self.minimum_font_size <= self.base_font_size <= self.maximum_font_size:
            raise ValueError("font sizes must satisfy minimum <= base <= maximum.")
        for name, value in (("hook_scale", self.hook_scale), ("payoff_scale", self.payoff_scale),
                            ("context_scale", self.context_scale), ("accent_scale", self.accent_scale)):
            if value <= 0:
                raise ValueError(f"{name} must be greater than 0.")
        if self.max_lines_per_block <= 0 or self.line_spacing <= 0:
            raise ValueError("line limits and line spacing must be positive.")
        if self.block_gap < 0:
            raise ValueError("block_gap must not be negative.")
        for name, value in (("horizontal_padding", self.horizontal_padding),
                            ("vertical_padding", self.vertical_padding),
                            ("ui_overlap_tolerance", self.ui_overlap_tolerance)):
            if not 0 <= value < 0.5:
                raise ValueError(f"{name} must be between 0 and 0.5.")
        if not 0 < self.minimum_region_width <= 1 or not 0 < self.minimum_region_height <= 1:
            raise ValueError("minimum region dimensions must be between 0 and 1.")


class TypographyPlanner:
    """Turns semantic copy and composition geometry into deterministic typography."""

    _ROLE_SCALES = {
        CopyRole.HOOK: "hook_scale",
        CopyRole.PAYOFF: "payoff_scale",
        CopyRole.CONTEXT: "context_scale",
        CopyRole.ACCENT: "accent_scale",
    }

    def __init__(self, config: TypographyPlannerConfig | None = None,
                 measurer: TextMeasurer | None = None) -> None:
        self.config = config or TypographyPlannerConfig()
        self.measurer = measurer or PillowTextMeasurer()

    def plan(self, *, copy: Sequence[CopyBlock], composition: CompositionPlan,
             target: ThumbnailTarget) -> TypographyPlan:
        blocks = tuple(copy)
        if not blocks:
            raise ValueError("copy must contain at least one block.")
        region = self._usable_region(composition, target)
        region = self._avoid_ui(region, target)
        if region.width < self.config.minimum_region_width or region.height < self.config.minimum_region_height:
            raise ValueError("No usable typography region remains.")
        alignment = self._alignment_for(region)
        return TypographyPlan(blocks=tuple(self._fit(
            blocks, region, alignment, target.size.width, target.size.height
        )))

    def _fit(
        self,
        blocks: tuple[CopyBlock, ...],
        region: BoundingBox,
        alignment: str,
        canvas_width: int,
        canvas_height: int,
    ) -> list[TypographyBlockPlan]:
        width = (
            region.width - 2 * self.config.horizontal_padding
        ) * canvas_width
        height = (
            region.height - 2 * self.config.vertical_padding
        ) * canvas_height

        if width <= 0 or height <= 0:
            raise ValueError("Typography region has no usable interior.")

        minimum_scale = min(
            self.config.minimum_font_size
            / (
                self.config.base_font_size
                * getattr(self.config, self._ROLE_SCALES[block.role])
                * block.emphasis
            )
            for block in blocks
        )

        scale = 1.0

        while scale >= minimum_scale - 0.0001:
            drafts: list[tuple[CopyBlock, int, str, float]] = []
            total = 0.0
            fits = True

            for block in blocks:
                size = self._font_size(block, scale)
                # Reserve horizontal space for the renderer's text stroke;
                # Pillow includes the stroke in its rendered text bounds.
                text_width = width - (2 * self.config.stroke_width)
                if text_width <= 0:
                    fits = False
                    break
                rendered, lines = self._wrap(block.text, size, text_width)

                if len(lines) > self.config.max_lines_per_block:
                    fits = False
                    break

                measured = self.measurer.measure(
                    rendered,
                    font_name=self.config.font_name,
                    font_size=size,
                    weight=self.config.weight,
                )

                if measured.width + (2 * self.config.stroke_width) > width + 0.5:
                    fits = False
                    break

                line_height = max(
                    1,
                    round(size * self.config.line_spacing),
                )
                block_height = (
                    line_height * (len(lines) - 1)
                    + measured.height
                    + (2 * self.config.stroke_width)
                )
                drafts.append((block, size, rendered, block_height))
                total += block_height

            if fits:
                total += (
                    max(0, len(drafts) - 1)
                    * self.config.block_gap
                    * canvas_height
                )

                if total <= height + 0.5:
                    cursor = (
                        region.top * canvas_height
                        + self.config.vertical_padding * canvas_height
                    )
                    cursor += (height - total) / 2

                    result: list[TypographyBlockPlan] = []

                    for index, (
                        block,
                        size,
                        rendered,
                        block_height,
                    ) in enumerate(drafts):
                        top = cursor / canvas_height
                        bottom = (cursor + block_height) / canvas_height

                        bounds = BoundingBox(
                            region.left + self.config.horizontal_padding,
                            top,
                            region.right - self.config.horizontal_padding,
                            bottom,
                        )

                        result.append(
                            TypographyBlockPlan(
                                copy=block,
                                font_name=self.config.font_name,
                                font_size=size,
                                weight=self.config.weight,
                                alignment=alignment,
                                color=self.config.color,
                                position=Point(
                                    self._anchor_x(bounds, alignment),
                                    (top + bottom) / 2,
                                ),
                                text_bounds=bounds,
                                max_lines=self.config.max_lines_per_block,
                                line_spacing=self.config.line_spacing,
                                rendered_text=rendered,
                                stroke_color=self.config.stroke_color,
                                stroke_width=self.config.stroke_width,
                            )
                        )

                        cursor += block_height

                        if index < len(drafts) - 1:
                            cursor += (
                                self.config.block_gap
                                * canvas_height
                            )

                    return result

            scale -= 0.01

        raise ValueError(
            "Copy cannot fit without violating minimum font size or line limits."
        )
    def _wrap(self, text: str, size: int, width: float) -> tuple[str, list[str]]:
        words = text.split()
        if not words:
            return "", [""]
        lines: list[str] = []
        current: list[str] = []
        for word in words:
            candidate = " ".join((*current, word))
            measured = self.measurer.measure(
                candidate, font_name=self.config.font_name, font_size=size, weight=self.config.weight
            )
            if current and measured.width > width:
                lines.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            lines.append(" ".join(current))
        return "\n".join(lines), lines

    def _font_size(self, block: CopyBlock, scale: float) -> int:
        role_scale = getattr(self.config, self._ROLE_SCALES[block.role])
        raw = self.config.base_font_size * role_scale * block.emphasis * scale
        return int(max(self.config.minimum_font_size, min(self.config.maximum_font_size, round(raw))))

    def _usable_region(self, composition: CompositionPlan, target: ThumbnailTarget) -> BoundingBox:
        if not composition.negative_space_regions:
            raise ValueError("Composition does not provide negative space.")
        base = max(composition.negative_space_regions, key=lambda r: r.width * r.height)
        if not target.safe_regions:
            return base
        intersections = [
            i for safe in target.safe_regions
            if (i := self._intersection(base, safe.bounds)) is not None
        ]
        if not intersections:
            raise ValueError("Negative space does not intersect a target safe region.")
        return max(intersections, key=lambda r: r.width * r.height)

    def _avoid_ui(self, region: BoundingBox, target: ThumbnailTarget) -> BoundingBox:
        current = region
        for occlusion in target.ui_occlusion_regions:
            if self._overlap(current, occlusion.region.bounds) <= self.config.ui_overlap_tolerance:
                continue
            candidates = self._split(current, occlusion.region.bounds)
            candidates = [
                c for c in candidates
                if c.width >= self.config.minimum_region_width and c.height >= self.config.minimum_region_height
            ]
            if candidates:
                current = min(candidates, key=lambda c: (
                    self._overlap(c, occlusion.region.bounds), -(c.width * c.height)
                ))
        return current

    @staticmethod
    def _split(region: BoundingBox, occlusion: BoundingBox) -> list[BoundingBox]:
        i = TypographyPlanner._intersection(region, occlusion)
        if i is None:
            return [region]
        result: list[BoundingBox] = []
        if i.left > region.left:
            result.append(BoundingBox(region.left, region.top, i.left, region.bottom))
        if i.right < region.right:
            result.append(BoundingBox(i.right, region.top, region.right, region.bottom))
        if i.top > region.top:
            result.append(BoundingBox(region.left, region.top, region.right, i.top))
        if i.bottom < region.bottom:
            result.append(BoundingBox(region.left, i.bottom, region.right, region.bottom))
        return result

    @staticmethod
    def _intersection(a: BoundingBox, b: BoundingBox) -> BoundingBox | None:
        left, top = max(a.left, b.left), max(a.top, b.top)
        right, bottom = min(a.right, b.right), min(a.bottom, b.bottom)
        return None if left >= right or top >= bottom else BoundingBox(left, top, right, bottom)

    @staticmethod
    def _overlap(a: BoundingBox, b: BoundingBox) -> float:
        i = TypographyPlanner._intersection(a, b)
        return 0.0 if i is None else (i.width * i.height) / (a.width * a.height)

    @staticmethod
    def _alignment_for(region: BoundingBox) -> str:
        center = (region.left + region.right) / 2
        return "left" if center < 0.40 else "right" if center > 0.60 else "center"

    @staticmethod
    def _anchor_x(region: BoundingBox, alignment: str) -> float:
        return region.left if alignment == "left" else region.right if alignment == "right" else (region.left + region.right) / 2
