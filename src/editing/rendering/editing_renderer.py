from __future__ import annotations

import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

from src.editing.models import EditingPlan
from src.exceptions import ClipperError


class EditingRenderingError(ClipperError):
    """Raised when an editing render operation fails."""


class EditingRenderer:
    """
    Execute an EditingPlan against an already-acquired local video.

    The renderer is intentionally downstream of planning:
    - it does not decide composition
    - it does not generate captions
    - it does not call AI
    - it only translates the EditingPlan into FFmpeg execution
    """

    def __init__(self, ffmpeg_path: str = "ffmpeg") -> None:
        self.ffmpeg_path = ffmpeg_path

    def render(
        self,
        *,
        source_path: Path,
        output_path: Path,
        plan: EditingPlan,
    ) -> Path:
        """Render one complete edited clip."""

        if not source_path.exists():
            raise EditingRenderingError(
                f"Source media does not exist: {source_path}"
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if output_path.exists():
            output_path.unlink()

        caption_file: Path | None = None

        try:
            command, caption_file = self._build_command(
                source_path=source_path,
                output_path=output_path,
                plan=plan,
            )

            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                )

            except FileNotFoundError as exc:
                raise EditingRenderingError(
                    "FFmpeg was not found. "
                    "Make sure FFmpeg is installed and available on your PATH."
                ) from exc

            if result.returncode != 0:
                if output_path.exists():
                    output_path.unlink()

                details = (
                    result.stderr.strip()
                    or result.stdout.strip()
                    or "Unknown FFmpeg editing error."
                )

                raise EditingRenderingError(
                    f"FFmpeg failed to render the edited clip:\n{details}"
                )

            if not output_path.exists():
                raise EditingRenderingError(
                    "FFmpeg completed but the edited output file "
                    f"was not created: {output_path}"
                )

            try:
                self.validate_media(output_path)
            except EditingRenderingError:
                if output_path.exists():
                    output_path.unlink()
                raise

            return output_path

        finally:
            if caption_file is not None and caption_file.exists():
                caption_file.unlink()

    def validate_media(self, media_path: Path) -> None:
        """Fully decode audio and video to detect corruption."""

        if not media_path.exists():
            raise EditingRenderingError(
                f"Cannot validate missing media file: {media_path}"
            )

        command = [
            self.ffmpeg_path,
            "-v",
            "error",
            "-i",
            str(media_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-f",
            "null",
            "NUL",
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
            )

        except FileNotFoundError as exc:
            raise EditingRenderingError(
                "FFmpeg was not found. "
                "Make sure FFmpeg is installed and available on your PATH."
            ) from exc

        if result.returncode != 0:
            details = (
                result.stderr.strip()
                or result.stdout.strip()
                or "Unknown FFmpeg validation error."
            )

            raise EditingRenderingError(
                "Edited media failed full audio/video validation:\n"
                f"{details}"
            )

    def _build_command(
        self,
        *,
        source_path: Path,
        output_path: Path,
        plan: EditingPlan,
    ) -> tuple[list[str], Path | None]:
        """
        Build the FFmpeg command for an EditingPlan.

        Returns the command and an optional temporary subtitle file.
        """

        canvas = plan.composition.canvas
        background = plan.composition.background

        filter_parts = [
            "[0:v]split=2[background][foreground]",
            (
                "[background]"
                f"scale={canvas.width}:{canvas.height}:"
                "force_original_aspect_ratio=increase,"
                f"crop={canvas.width}:{canvas.height},"
                f"boxblur={background.blur_radius}:1,"
                f"eq=brightness={background.brightness - 1.0}"
                "[background_processed]"
            ),
            (
                "[foreground]"
                f"scale={canvas.width}:{canvas.height}:"
                "force_original_aspect_ratio=decrease"
                "[foreground_scaled]"
            ),
            (
                "[background_processed][foreground_scaled]"
                "overlay="
                "(W-w)/2:"
                "(H-h)/2"
                "[composed]"
            ),
        ]

        current_video_label = "[composed]"
        caption_file: Path | None = None

        if plan.captions.enabled and plan.captions.segments:
            caption_file = self._create_ass_file(
                plan=plan,
                output_directory=output_path.parent,
            )

            subtitle_path = self._escape_filter_path(
                caption_file
            )

            filter_parts.append(
                f"{current_video_label}"
                f"subtitles='{subtitle_path}'"
                "[captioned]"
            )

            current_video_label = "[captioned]"

        filter_parts.append(
            f"{current_video_label}"
            "setsar=1"
            "[final]"
        )

        filter_complex = ";".join(filter_parts)

        render = plan.render_config

        command = [
            self.ffmpeg_path,
            "-y",
            "-i",
            str(source_path),
            "-filter_complex",
            filter_complex,
            "-map",
            "[final]",
            "-map",
            "0:a:0",
            "-c:v",
            render.video_codec,
            "-preset",
            "fast",
            "-crf",
            str(render.crf),
            "-c:a",
            render.audio_codec,
            "-b:a",
            render.audio_bitrate,
            "-movflags",
            "+faststart",
            str(output_path),
        ]

        return command, caption_file

    @staticmethod
    def _create_ass_file(
        *,
        plan: EditingPlan,
        output_directory: Path,
    ) -> Path:
        """
        Create a temporary ASS subtitle file from CaptionPlan.

        ASS gives us deterministic burned-in captions while keeping
        caption generation separate from rendering.
        """

        style = plan.captions.style

        if style is None:
            raise EditingRenderingError(
                "CaptionPlan is enabled but has no caption style."
            )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = NamedTemporaryFile(
            mode="w",
            suffix=".ass",
            prefix=f"{plan.clip_id}_captions_",
            dir=output_directory,
            encoding="utf-8",
            delete=False,
        )

        path = Path(temporary.name)

        try:
            alignment = EditingRenderer._ass_alignment(
                horizontal_alignment=style.horizontal_alignment,
                vertical_position=style.vertical_position,
            )

            vertical_margin = EditingRenderer._ass_vertical_margin(
                style=style,
            )

            shadow = 1 if style.shadow else 0

            temporary.write(
                "[Script Info]\n"
                "ScriptType: v4.00+\n"
                "PlayResX: 1080\n"
                "PlayResY: 1920\n"
                "ScaledBorderAndShadow: yes\n"
                "\n"
                "[V4+ Styles]\n"
                "Format: Name, Fontname, Fontsize, "
                "PrimaryColour, SecondaryColour, "
                "OutlineColour, BackColour, Bold, Italic, "
                "Underline, StrikeOut, ScaleX, ScaleY, "
                "Spacing, Angle, BorderStyle, Outline, "
                "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
                f"Style: Default,{style.font_name},{style.font_size},"
                "&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,"
                f"{1 if style.font_weight == 'bold' else 0},0,0,0,"
                "100,100,0,0,1,"
                f"{style.outline_width},{shadow},"
                f"{alignment},"
                f"{style.horizontal_margin},"
                f"{style.horizontal_margin},"
                f"{vertical_margin},1\n"
                "\n"
                "[Events]\n"
                "Format: Layer, Start, End, Style, Name, "
                "MarginL, MarginR, MarginV, Effect, Text\n"
            )

            for segment in plan.captions.segments:
                start = EditingRenderer._format_ass_time(
                    segment.start_time
                )

                end = EditingRenderer._format_ass_time(
                    segment.end_time
                )

                text = EditingRenderer._escape_ass_text(
                    segment.text
                )

                temporary.write(
                    f"Dialogue: 0,{start},{end},Default,,"
                    "0,0,0,,"
                    f"{text}\n"
                )

        finally:
            temporary.close()

        return path

    @staticmethod
    def _ass_alignment(
        *,
        horizontal_alignment: str,
        vertical_position: str,
    ) -> int:
        """
        Convert semantic caption positioning into ASS alignment.

        ASS alignment values:
            1 = bottom-left
            2 = bottom-center
            3 = bottom-right
            4 = middle-left
            5 = center
            6 = middle-right
            7 = top-left
            8 = top-center
            9 = top-right
        """

        horizontal = {
            "left": {
                "top": 7,
                "center": 4,
                "lower_middle": 4,
                "bottom": 1,
            },
            "center": {
                "top": 8,
                "center": 5,
                "lower_middle": 2,
                "bottom": 2,
            },
            "right": {
                "top": 9,
                "center": 6,
                "lower_middle": 6,
                "bottom": 3,
            },
        }

        try:
            return horizontal[horizontal_alignment][vertical_position]
        except KeyError as exc:
            raise EditingRenderingError(
                "Unsupported caption positioning: "
                f"{horizontal_alignment=}, {vertical_position=}"
            ) from exc

    @staticmethod
    def _ass_vertical_margin(*, style) -> int:
        """
        Resolve the semantic vertical margin used by ASS.

        lower_middle intentionally uses a large bottom margin so
        captions sit above the platform UI region rather than at
        the absolute bottom of the frame.
        """

        if style.vertical_position == "lower_middle":
            return style.vertical_margin

        if style.vertical_position == "bottom":
            return style.vertical_margin

        if style.vertical_position == "top":
            return style.vertical_margin

        # ASS center alignment ignores MarginV for practical
        # vertical placement, so the value is irrelevant here.
        return 0

    @staticmethod
    def _format_ass_time(seconds: float) -> str:
        """Convert seconds to ASS H:MM:SS.cc format."""

        if seconds < 0:
            seconds = 0.0

        total_centiseconds = round(seconds * 100)

        hours, remainder = divmod(
            total_centiseconds,
            360000,
        )

        minutes, remainder = divmod(
            remainder,
            6000,
        )

        seconds_value, centiseconds = divmod(
            remainder,
            100,
        )

        return (
            f"{hours}:"
            f"{minutes:02d}:"
            f"{seconds_value:02d}."
            f"{centiseconds:02d}"
        )

    @staticmethod
    def _escape_ass_text(text: str) -> str:
        """
        Escape text for ASS dialogue events.

        Newlines are converted into ASS line breaks.
        """

        return (
            text
            .replace("\\", r"\\")
            .replace("{", r"\{")
            .replace("}", r"\}")
            .replace("\n", r"\N")
        )

    @staticmethod
    def _escape_filter_path(path: Path) -> str:
        """
        Escape a Windows path for an FFmpeg filter argument.

        FFmpeg filter syntax treats backslashes, colons, and apostrophes
        specially.
        """

        value = str(path.resolve())

        return (
            value
            .replace("\\", "/")
            .replace(":", r"\:")
            .replace("'", r"\'")
        )