"""Module 6: Video rendering.

The renderer operates on already-acquired local media.

Source acquisition is deliberately handled by a separate provider so
that rendering does not need to know whether the original source was
YouTube, a local file, or another future source type.
"""

import subprocess
from pathlib import Path

from src.exceptions import ClipperError


class RenderingError(ClipperError):
    """Raised when video rendering or validation fails."""


class FFmpegRenderer:
    """Render and validate final clips from locally acquired media."""

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg_path = ffmpeg_path

    def render_clip(
        self,
        source_path: Path,
        output_path: Path,
        start_time: float,
        end_time: float,
    ) -> Path:
        """Render a requested range and fully validate the final output."""

        if end_time <= start_time:
            raise RenderingError(
                "End time must be greater than start time."
            )

        if not source_path.exists():
            raise RenderingError(
                f"Source media does not exist: {source_path}"
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if output_path.exists():
            output_path.unlink()

        duration = end_time - start_time

        command = [
            self.ffmpeg_path,
            "-y",
            "-ss",
            str(start_time),
            "-i",
            str(source_path),
            "-t",
            str(duration),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
            )

        except FileNotFoundError as exc:
            raise RenderingError(
                "FFmpeg was not found. "
                "Make sure FFmpeg is installed and available on your PATH."
            ) from exc

        if result.returncode != 0:
            if output_path.exists():
                output_path.unlink()

            details = (
                result.stderr.strip()
                or result.stdout.strip()
                or "Unknown FFmpeg rendering error."
            )

            raise RenderingError(
                f"FFmpeg failed to render the clip:\n{details}"
            )

        if not output_path.exists():
            raise RenderingError(
                "FFmpeg completed but the output file was not created: "
                f"{output_path}"
            )

        try:
            self.validate_media(output_path)
        except RenderingError:
            if output_path.exists():
                output_path.unlink()
            raise

        return output_path

    def validate_media(self, media_path: Path) -> None:
        """Fully decode audio and video to detect corruption."""

        if not media_path.exists():
            raise RenderingError(
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
            raise RenderingError(
                "FFmpeg was not found. "
                "Make sure FFmpeg is installed and available on your PATH."
            ) from exc

        if result.returncode != 0:
            details = (
                result.stderr.strip()
                or result.stdout.strip()
                or "Unknown FFmpeg validation error."
            )

            raise RenderingError(
                "Rendered media failed full audio/video validation:\n"
                f"{details}"
            )