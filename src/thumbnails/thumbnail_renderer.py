from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.thumbnails.models import ThumbnailPlan


class ThumbnailRenderer:
    """
    Renders a ThumbnailPlan into a JPEG image using Pillow.

    The renderer does not:
    - extract video frames
    - call FFmpeg
    - select frames
    - generate thumbnail text
    - call AI services

    It only turns an already-selected source frame and a deterministic
    ThumbnailPlan into pixels.
    """

    def render(
        self,
        *,
        plan: ThumbnailPlan,
        source_frame_path: Path,
        output_path: Path,
    ) -> Path:
        if not source_frame_path.exists():
            raise FileNotFoundError(
                f"Source frame does not exist: {source_frame_path}"
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with Image.open(source_frame_path) as source_frame:
            source_frame = source_frame.convert("RGB")

            canvas = self._create_background(
                plan=plan,
            )

            visual = self._prepare_visual(
                source_frame=source_frame,
                plan=plan,
            )

            self._place_visual(
                canvas=canvas,
                visual=visual,
                plan=plan,
            )

            self._draw_text(
                canvas=canvas,
                plan=plan,
            )

            canvas.save(
                output_path,
                format="JPEG",
                quality=95,
                optimize=True,
            )

        if not output_path.exists():
            raise RuntimeError(
                "Thumbnail rendering completed without "
                f"producing the expected output: {output_path}"
            )

        return output_path

    def _create_background(
        self,
        *,
        plan: ThumbnailPlan,
    ) -> Image.Image:
        style = plan.style

        if style.background_mode == "solid":
            return Image.new(
                "RGB",
                (style.width, style.height),
                style.background_color,
            )

        if style.background_mode == "gradient":
            return self._create_gradient(
                width=style.width,
                height=style.height,
                start_color=style.background_color,
                end_color=style.background_accent_color,
            )

        raise ValueError(
            f"Unsupported background mode: "
            f"{style.background_mode}"
        )

    def _create_gradient(
        self,
        *,
        width: int,
        height: int,
        start_color: str,
        end_color: str,
    ) -> Image.Image:
        start = self._parse_hex_color(start_color)
        end = self._parse_hex_color(end_color)

        image = Image.new(
            "RGB",
            (width, height),
        )

        pixels = image.load()

        for y in range(height):
            ratio = y / max(height - 1, 1)

            red = int(
                start[0] + (end[0] - start[0]) * ratio
            )
            green = int(
                start[1] + (end[1] - start[1]) * ratio
            )
            blue = int(
                start[2] + (end[2] - start[2]) * ratio
            )

            for x in range(width):
                pixels[x, y] = (
                    red,
                    green,
                    blue,
                )

        return image

    def _prepare_visual(
        self,
        *,
        source_frame: Image.Image,
        plan: ThumbnailPlan,
    ) -> Image.Image:
        composition = plan.composition

        target_height = int(
            plan.style.height * 0.52
            * composition.visual_scale
        )

        if target_height <= 0:
            raise ValueError(
                "visual height must be greater than zero."
            )

        aspect_ratio = (
            source_frame.width / source_frame.height
        )

        target_width = max(
            1,
            int(target_height * aspect_ratio),
        )

        return source_frame.resize(
            (target_width, target_height),
            Image.Resampling.LANCZOS,
        )

    def _place_visual(
        self,
        *,
        canvas: Image.Image,
        visual: Image.Image,
        plan: ThumbnailPlan,
    ) -> None:
        composition = plan.composition

        center_x = int(
            canvas.width * composition.visual_center_x
        )
        center_y = int(
            canvas.height * composition.visual_center_y
        )

        left = center_x - visual.width // 2
        top = center_y - visual.height // 2

        canvas.paste(
            visual,
            (left, top),
        )

    def _draw_text(
        self,
        *,
        canvas: Image.Image,
        plan: ThumbnailPlan,
    ) -> None:
        style = plan.style
        composition = plan.composition

        draw = ImageDraw.Draw(canvas)

        font = self._load_font(
            font_name=style.font_name,
            font_size=style.font_size,
        )

        max_width = int(
            canvas.width * composition.text_width_fraction
        )

        lines = self._wrap_text(
            draw=draw,
            text=plan.text,
            font=font,
            max_width=max_width,
        )

        text = "\n".join(lines)

        bbox = draw.multiline_textbbox(
            (0, 0),
            text,
            font=font,
            stroke_width=style.text_stroke_width,
            spacing=8,
        )

        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        if style.text_position == "top":
            y = int(canvas.height * 0.08)

        elif style.text_position == "center":
            y = (canvas.height - text_height) // 2

        elif style.text_position == "bottom":
            y = int(
                canvas.height * 0.82
                - text_height
            )

        else:
            raise ValueError(
                f"Unsupported text position: "
                f"{style.text_position}"
            )

        x = (canvas.width - text_width) // 2

        draw.multiline_text(
            (x, y),
            text,
            font=font,
            fill=style.text_color,
            stroke_width=style.text_stroke_width,
            stroke_fill=style.text_stroke_color,
            spacing=8,
            align="center",
        )

    def _wrap_text(
        self,
        *,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont,
        max_width: int,
    ) -> list[str]:
        words = text.split()

        if not words:
            raise ValueError(
                "text must not be empty."
            )

        lines: list[str] = []
        current: list[str] = []

        for word in words:
            candidate = (
                " ".join(current + [word])
            )

            bbox = draw.textbbox(
                (0, 0),
                candidate,
                font=font,
            )

            width = bbox[2] - bbox[0]

            if current and width > max_width:
                lines.append(" ".join(current))
                current = [word]
            else:
                current.append(word)

        if current:
            lines.append(" ".join(current))

        if len(lines) > 2:
            raise ValueError(
                "Thumbnail text exceeds the maximum "
                "of two lines."
            )

        return lines

    def _load_font(
        self,
        *,
        font_name: str,
        font_size: int,
    ) -> ImageFont.FreeTypeFont:
        candidates = [
            Path("C:/Windows/Fonts") / "arialbd.ttf",
            Path("C:/Windows/Fonts") / "arial.ttf",
            Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf"),
            Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        ]

        for path in candidates:
            if path.exists():
                return ImageFont.truetype(
                    str(path),
                    font_size,
                )

        raise FileNotFoundError(
            f"Could not locate a usable font for "
            f"{font_name!r}."
        )

    @staticmethod
    def _parse_hex_color(
        value: str,
    ) -> tuple[int, int, int]:
        value = value.strip()

        if value.startswith("#"):
            value = value[1:]

        if len(value) != 6:
            raise ValueError(
                f"Invalid hex color: {value!r}"
            )

        try:
            return (
                int(value[0:2], 16),
                int(value[2:4], 16),
                int(value[4:6], 16),
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid hex color: {value!r}"
            ) from exc