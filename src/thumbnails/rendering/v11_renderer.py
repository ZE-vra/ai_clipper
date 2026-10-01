from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageFilter

from src.thumbnails.domain.assets import VisualAsset
from src.thumbnails.domain.concepts import CopyRole
from src.thumbnails.domain.geometry import BoundingBox
from src.thumbnails.domain.plans import ThumbnailRenderPlan, TypographyBlockPlan


class V11AssetResolver(Protocol):
    def resolve(self, asset_id: str) -> VisualAsset:
        ...


class V11InMemoryAssetResolver:
    def __init__(self, assets: Mapping[str, VisualAsset]) -> None:
        self._assets = dict(assets)

    def resolve(self, asset_id: str) -> VisualAsset:
        try:
            return self._assets[asset_id]
        except KeyError as exc:
            raise ValueError(f"visual asset '{asset_id}' could not be resolved.") from exc


class V11PillowThumbnailRenderer:
    """
    Renderer for the V1.1 art direction.

    It keeps the source frame clean, fills the vertical canvas with a true crop,
    adds a restrained top/bottom tonal gradient, and gives semantic copy roles
    different visual weight.
    """

    def __init__(self, *, asset_resolver: V11AssetResolver) -> None:
        self._asset_resolver = asset_resolver

    def render(self, *, plan: ThumbnailRenderPlan, output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        canvas = self._render_visual(plan)
        canvas = self._treat(canvas, plan)
        self._render_text(canvas, plan)

        canvas.save(output, format="JPEG", quality=95, optimize=True)
        return output

    def _render_visual(self, plan: ThumbnailRenderPlan) -> Image.Image:
        placement = plan.composition.visual_placements[0]
        asset = self._asset_resolver.resolve(placement.asset_id)
        image = self._open(asset.path).convert("RGB")

        if placement.crop_bounds is not None:
            image = self._crop(image, placement.crop_bounds)

        return self._cover(image, plan.canvas_width, plan.canvas_height)

    def _treat(self, image: Image.Image, plan: ThumbnailRenderPlan) -> Image.Image:
        treatment = plan.visual_treatment

        image = ImageEnhance.Contrast(image).enhance(treatment.contrast)
        image = ImageEnhance.Color(image).enhance(treatment.saturation)
        image = ImageEnhance.Sharpness(image).enhance(treatment.sharpness)

        if treatment.vignette > 0:
            image = self._vignette(image, treatment.vignette)

        # A restrained top gradient creates a readable title field without
        # turning the thumbnail into a letterboxed video frame.
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        pixels = overlay.load()
        height = image.height
        gradient_height = int(height * 0.38)

        for y in range(gradient_height):
            strength = int(145 * max(0.0, 1.0 - y / gradient_height))
            for x in range(image.width):
                pixels[x, y] = (0, 0, 0, strength)

        return Image.alpha_composite(
            image.convert("RGBA"),
            overlay,
        ).convert("RGB")

    def _render_text(self, canvas: Image.Image, plan: ThumbnailRenderPlan) -> None:
        draw = ImageDraw.Draw(canvas)

        for block in plan.typography.blocks:
            self._draw_block(draw, canvas, block)

    def _draw_block(
        self,
        draw: ImageDraw.ImageDraw,
        canvas: Image.Image,
        block: TypographyBlockPlan,
    ) -> None:
        if not block.rendered_text:
            raise ValueError("typography block must contain rendered_text.")

        bounds = block.text_bounds
        left = round(bounds.left * canvas.width)
        top = round(bounds.top * canvas.height)
        right = round(bounds.right * canvas.width)
        bottom = round(bounds.bottom * canvas.height)

        font = self._font(block.font_name, block.font_size, block.weight)
        lines = block.rendered_text.splitlines()

        line_height = max(1, round(block.font_size * block.line_spacing))
        total_height = line_height * (len(lines) - 1) + max(
            draw.textbbox(
                (0, 0),
                line,
                font=font,
                stroke_width=block.stroke_width,
            )[3]
            - draw.textbbox(
                (0, 0),
                line,
                font=font,
                stroke_width=block.stroke_width,
            )[1]
            for line in lines
        )

        y = top + max(0, (bottom - top - total_height) // 2)

        color = "#FFD400" if block.copy.role is CopyRole.PAYOFF else "#FFFFFF"

        for line in lines:
            bbox = draw.textbbox(
                (0, 0),
                line,
                font=font,
                stroke_width=block.stroke_width,
            )
            line_width = bbox[2] - bbox[0]

            if block.alignment == "left":
                x = left
            elif block.alignment == "right":
                x = right - line_width
            else:
                x = left + (right - left - line_width) // 2

            draw.text(
                (x + 3, y + 5),
                line,
                font=font,
                fill=(0, 0, 0, 170),
                stroke_width=block.stroke_width + 2,
                stroke_fill=(0, 0, 0, 120),
            )
            draw.text(
                (x, y),
                line,
                font=font,
                fill=color,
                stroke_width=block.stroke_width,
                stroke_fill="#111111",
            )
            y += line_height

    @staticmethod
    def _open(path: str | Path) -> Image.Image:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"visual asset does not exist: {source}")
        with Image.open(source) as image:
            return image.copy()

    @staticmethod
    def _crop(image: Image.Image, bounds: BoundingBox) -> Image.Image:
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

    @staticmethod
    def _vignette(image: Image.Image, strength: float) -> Image.Image:
        width, height = image.size
        mask = Image.new("L", (width, height), 255)
        draw = ImageDraw.Draw(mask)
        draw.ellipse(
            (
                -width * 0.18,
                -height * 0.10,
                width * 1.18,
                height * 1.10,
            ),
            fill=0,
        )

        # Blur the actual mask, not the ImageFilter object.
        mask = mask.filter(ImageFilter.GaussianBlur(radius=36))

        alpha = mask.point(
            lambda value: min(255, int(value * strength))
        )
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay.putalpha(alpha)

        return Image.alpha_composite(
            image.convert("RGBA"),
            overlay,
        ).convert("RGB")

    @staticmethod
    def _font(name: str, size: int, weight: str):
        candidates = []
        if name.lower() == "arial":
            if weight.lower() == "bold":
                candidates.extend(
                    [
                        Path("C:/Windows/Fonts/arialbd.ttf"),
                        Path("C:/Windows/Fonts/Arial_Bold.ttf"),
                    ]
                )
            else:
                candidates.append(Path("C:/Windows/Fonts/arial.ttf"))

        candidates.append(
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        )

        for candidate in candidates:
            if candidate.exists():
                return ImageFont.truetype(str(candidate), size)

        return ImageFont.load_default(size=size)
