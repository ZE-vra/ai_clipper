"""Source acquisition for local and YouTube media sections."""

import subprocess
from pathlib import Path

from src.exceptions import ClipperError


class SourceAcquisitionError(ClipperError):
    """Raised when a requested source section cannot be acquired."""


class LocalSourceProvider:
    """Acquire a requested section from an already-local media file."""

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg_path = ffmpeg_path

    def acquire_section(
        self,
        source_path: Path,
        output_path: Path,
        start_time: float,
        end_time: float,
    ) -> Path:
        """Extract and fully validate one local media section."""

        if end_time <= start_time:
            raise SourceAcquisitionError(
                "End time must be greater than start time."
            )

        if not source_path.exists():
            raise SourceAcquisitionError(
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
            raise SourceAcquisitionError(
                "FFmpeg was not found. "
                "Make sure FFmpeg is installed and available on your PATH."
            ) from exc

        if result.returncode != 0:
            if output_path.exists():
                output_path.unlink()

            details = (
                result.stderr.strip()
                or result.stdout.strip()
                or "Unknown FFmpeg acquisition error."
            )

            raise SourceAcquisitionError(
                f"FFmpeg failed to acquire the requested section:\n"
                f"{details}"
            )

        if not output_path.exists():
            raise SourceAcquisitionError(
                f"FFmpeg completed but the expected media file "
                f"was not created: {output_path}"
            )

        try:
            self.validate_media(output_path)
        except SourceAcquisitionError:
            if output_path.exists():
                output_path.unlink()
            raise

        return output_path

    def validate_media(self, media_path: Path) -> None:
        """Fully decode audio and video to detect corruption."""

        if not media_path.exists():
            raise SourceAcquisitionError(
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
            raise SourceAcquisitionError(
                "FFmpeg was not found. "
                "Make sure FFmpeg is installed and available on your PATH."
            ) from exc

        if result.returncode != 0:
            details = (
                result.stderr.strip()
                or result.stdout.strip()
                or "Unknown FFmpeg validation error."
            )

            raise SourceAcquisitionError(
                f"Acquired media failed full audio/video validation:\n"
                f"{details}"
            )


class YouTubeSourceProvider:
    """Acquire individual video sections from YouTube."""

    def __init__(
        self,
        ytdlp_path: str = "yt-dlp",
        ffmpeg_path: str = "ffmpeg",
    ):
        self.ytdlp_path = ytdlp_path
        self.ffmpeg_path = ffmpeg_path

    def acquire_section(
        self,
        source_url: str,
        output_path: Path,
        start_time: float,
        end_time: float,
    ) -> Path:
        """Download one requested YouTube section and validate it."""

        if end_time <= start_time:
            raise SourceAcquisitionError(
                "End time must be greater than start time."
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if output_path.exists():
            output_path.unlink()

        command = [
            self.ytdlp_path,
            "--download-sections",
            f"*{start_time}-{end_time}",
            "-f",
            "bv*[vcodec^=avc1][ext=mp4]+ba[ext=m4a]/b[ext=mp4]",
            "--merge-output-format",
            "mp4",
            "--postprocessor-args",
            "Merger:-c:a aac -b:a 128k",
            "-o",
            str(output_path),
            source_url,
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
            )

        except FileNotFoundError as exc:
            raise SourceAcquisitionError(
                "yt-dlp was not found. "
                "Make sure yt-dlp is installed and available on your PATH."
            ) from exc

        if result.returncode != 0:
            message = self._format_process_error(
                "yt-dlp failed to acquire the requested section.",
                result.stdout,
                result.stderr,
            )

            if output_path.exists():
                output_path.unlink()

            raise SourceAcquisitionError(message)

        if not output_path.exists():
            raise SourceAcquisitionError(
                f"yt-dlp completed but the expected media file "
                f"was not created: {output_path}"
            )

        try:
            self.validate_media(output_path)
        except SourceAcquisitionError:
            if output_path.exists():
                output_path.unlink()
            raise

        return output_path

    def validate_media(self, media_path: Path) -> None:
        """Fully decode audio and video to detect corrupt streams."""

        if not media_path.exists():
            raise SourceAcquisitionError(
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
            raise SourceAcquisitionError(
                "FFmpeg was not found. "
                "Make sure FFmpeg is installed and available on your PATH."
            ) from exc

        if result.returncode != 0:
            details = (
                result.stderr.strip()
                or result.stdout.strip()
                or "Unknown FFmpeg validation error."
            )

            raise SourceAcquisitionError(
                f"Acquired media failed full audio/video validation:\n"
                f"{details}"
            )

    @staticmethod
    def _format_process_error(
        message: str,
        stdout: str,
        stderr: str,
    ) -> str:
        details = stderr.strip() or stdout.strip()

        if not details:
            return message

        return f"{message}\n{details}"