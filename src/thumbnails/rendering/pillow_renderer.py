from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol

from PIL import Image, ImageDraw, ImageEnhance, ImageFont

from ..domain.assets import VisualAsset
from ..domain.geometry import BoundingBox
from ..domain.plans import ThumbnailRenderPlan, TypographyBlockPlan


class VisualAssetResolver(Protocol):
    def resolve(self, asset_id: str) -> VisualAsset:
        ...


class InMemoryVisualAssetResolver:
    """Simple deterministic resolver for local rendering and tests."""

    def __init__(self, assets: Mapping[str, VisualAsset]) -> None:
        self._assets = dict(assets)

    def resolve(self, asset_id: str) -> VisualAsset:
        try:
            return self._assets[asset_id]
        except KeyError as exc:
            raise ValueError(
                f"visual asset '{asset_id}' could not be resolved."
            ) from exc


class PillowThumbnailRenderer:
    """Executes a ThumbnailRenderPlan without making creative decisions."""

    def __init__(
        self,
        *,
        asset_resolver: VisualAssetResolver,
    ) -> None:
        self._asset_resolver = asset_resolver

    def render(
        self,
        *,
        plan: ThumbnailRenderPlan,
        output_path: str | Path,
    ) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        canvas = self._render_visuals(plan)
        canvas = self._apply_visual_treatment(canvas, plan)
        self._render_typography(canvas, plan)

        canvas.save(
            output,
            format="JPEG",
            quality=95,
            optimize=True,
        )

        return output

    def _render_visuals(
        self,
        plan: ThumbnailRenderPlan,
    ) -> Image.Image:
        width = plan.canvas_width
        height = plan.canvas_height

        if plan.background_asset is not None:
            canvas = self._cover(
                self._open_asset(plan.background_asset.path),
                width,
                height,
            ).convert("RGB")
        else:
            canvas = Image.new(
                "RGB",
                (width, height),
                "black",
            )

        for placement in plan.composition.visual_placements:
            asset = self._asset_resolver.resolve(
                placement.asset_id
            )

            image = self._open_asset(asset.path).convert("RGB")

            if placement.crop_bounds is not None:
                image = self._crop_normalized(
                    image,
                    placement.crop_bounds,
                )

            image = self._cover(
                image,
                width,
                height,
            )

            # V1 composition contains the primary visual as a full
            # canvas placement. More advanced compositing belongs later.
            canvas = image

        return canvas

    @staticmethod
    def _open_asset(path: str | Path) -> Image.Image:
        source = Path(path)

        if not source.exists():
            raise FileNotFoundError(
                f"visual asset does not exist: {source}"
            )

        with Image.open(source) as image:
            return image.copy()

    @staticmethod
    def _crop_normalized(
        image: Image.Image,
        bounds: BoundingBox,
    ) -> Image.Image:
        width, height = image.size

        left = round(bounds.left * width)
        top = round(bounds.top * height)
        right = round(bounds.right * width)
        bottom = round(bounds.bottom * height)

        left = max(0, min(left, width - 1))
        top = max(0, min(top, height - 1))
        right = max(left + 1, min(right, width))
        bottom = max(top + 1, min(bottom, height))

        return image.crop(
            (left, top, right, bottom)
        )

    @staticmethod
    def _cover(
        image: Image.Image,
        width: int,
        height: int,
    ) -> Image.Image:
        source_width, source_height = image.size

        if source_width <= 0 or source_height <= 0:
            raise ValueError("visual asset has invalid dimensions.")

        source_ratio = source_width / source_height
        target_ratio = width / height

        if source_ratio > target_ratio:
            new_height = height
            new_width = round(height * source_ratio)
        else:
            new_width = width
            new_height = round(width / source_ratio)

        resized = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS,
        )

        left = max(0, (new_width - width) // 2)
        top = max(0, (new_height - height) // 2)

        return resized.crop(
            (
                left,
                top,
                left + width,
                top + height,
            )
        )

    def _apply_visual_treatment(
        self,
        image: Image.Image,
        plan: ThumbnailRenderPlan,
    ) -> Image.Image:
        treatment = plan.visual_treatment

        image = ImageEnhance.Contrast(image).enhance(
            treatment.contrast
        )

        image = ImageEnhance.Color(image).enhance(
            treatment.saturation
        )

        image = ImageEnhance.Sharpness(image).enhance(
            treatment.sharpness
        )

        if treatment.overlay_opacity > 0:
            overlay = Image.new(
                "RGBA",
                image.size,
                (
                    0,
                    0,
                    0,
                    round(255 * treatment.overlay_opacity),
                ),
            )

            image = Image.alpha_composite(
                image.convert("RGBA"),
                overlay,
            ).convert("RGB")

        if treatment.vignette > 0:
            image = self._apply_vignette(
                image,
                treatment.vignette,
            )

        return image

    @staticmethod
    def _apply_vignette(
        image: Image.Image,
        strength: float,
    ) -> Image.Image:
        width, height = image.size

        mask = Image.new(
            "L",
            (width, height),
            0,
        )

        pixels = mask.load()

        center_x = width / 2
        center_y = height / 2
        max_distance = (
            center_x**2 + center_y**2
        ) ** 0.5

        for y in range(height):
            for x in range(width):
                distance = (
                    (x - center_x) ** 2
                    + (y - center_y) ** 2
                ) ** 0.5

                normalized = min(
                    1.0,
                    distance / max_distance,
                )

                pixels[x, y] = round(
                    255 * normalized * strength
                )

        overlay = Image.new(
            "RGBA",
            image.size,
            (0, 0, 0, 0),
        )

        overlay.putalpha(mask)

        return Image.alpha_composite(
            image.convert("RGBA"),
            overlay,
        ).convert("RGB")

    def _render_typography(
        self,
        canvas: Image.Image,
        plan: ThumbnailRenderPlan,
    ) -> None:
        draw = ImageDraw.Draw(canvas)

        for block in plan.typography.blocks:
            self._render_text_block(
                draw=draw,
                canvas=canvas,
                block=block,
            )

    def _render_text_block(
        self,
        *,
        draw: ImageDraw.ImageDraw,
        canvas: Image.Image,
        block: TypographyBlockPlan,
    ) -> None:
        if not block.rendered_text:
            raise ValueError(
                "typography block must contain rendered_text."
            )

        bounds = block.text_bounds

        left = round(bounds.left * canvas.width)
        top = round(bounds.top * canvas.height)
        right = round(bounds.right * canvas.width)
        bottom = round(bounds.bottom * canvas.height)

        available_width = right - left
        available_height = bottom - top

        font = self._load_font(
            font_name=block.font_name,
            font_size=block.font_size,
            weight=block.weight,
        )

        lines = block.rendered_text.splitlines()

        if not lines:
            return

        line_height = max(
            1,
            round(
                block.font_size
                * block.line_spacing
            ),
        )

        metrics = []

        for line in lines:
            bbox = draw.textbbox(
                (0, 0),
                line,
                font=font,
                stroke_width=block.stroke_width,
            )

            metrics.append(
                (
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                )
            )

        total_height = (
            line_height * (len(lines) - 1)
            + max(height for _, height in metrics)
        )

        if total_height > available_height:
            raise ValueError(
                "text block exceeds planned height: "
                f"{total_height}px > "
                f"{available_height}px"
            )

        if any(
            width > available_width
            for width, _ in metrics
        ):
            raise ValueError(
                "text block exceeds planned width."
            )

        y = (
            top
            + (available_height - total_height) // 2
        )

        for line, (line_width, _) in zip(
            lines,
            metrics,
        ):
            if block.alignment == "left":
                x = left
            elif block.alignment == "right":
                x = right - line_width
            else:
                x = (
                    left
                    + (available_width - line_width)
                    // 2
                )

            draw.text(
                (x, y),
                line,
                font=font,
                fill=block.color,
                stroke_width=block.stroke_width,
                stroke_fill=block.stroke_color,
            )

            y += line_height

    @staticmethod
    def _load_font(
        *,
        font_name: str,
        font_size: int,
        weight: str,
    ) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        candidates: list[Path] = []

        if font_name.lower() == "arial":
            if weight.lower() == "bold":
                candidates.extend(
                    [
                        Path("C:/Windows/Fonts/arialbd.ttf"),
                        Path("C:/Windows/Fonts/Arial_Bold.ttf"),
                    ]
                )
            else:
                candidates.append(
                    Path("C:/Windows/Fonts/arial.ttf")
                )

        candidates.append(
            Path(
                "/usr/share/fonts/truetype/dejavu/"
                "DejaVuSans-Bold.ttf"
            )
        )

        for candidate in candidates:
            if candidate.exists():
                return ImageFont.truetype(
                    str(candidate),
                    font_size,
                )

        return ImageFont.load_default(
            size=font_size
        )
