from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Protocol


class ThumbnailFrameExtractor(Protocol):
    """
    Extracts a single video frame at a requested timestamp.

    The interface keeps FFmpeg-specific behavior outside the thumbnail
    planning and rendering layers.
    """

    def extract(
        self,
        *,
        source_path: Path,
        timestamp: float,
        output_path: Path,
    ) -> Path:
        ...


class FFmpegFrameExtractor:
    """
    Extracts one video frame using FFmpeg.

    FFmpeg is treated as an external media boundary. The rest of the
    thumbnail subsystem only needs to know that an image was produced.
    """

    def __init__(
        self,
        *,
        ffmpeg_binary: str = "ffmpeg",
    ) -> None:
        if not ffmpeg_binary.strip():
            raise ValueError(
                "ffmpeg_binary must not be empty."
            )

        self.ffmpeg_binary = ffmpeg_binary

    def extract(
        self,
        *,
        source_path: Path,
        timestamp: float,
        output_path: Path,
    ) -> Path:
        if not source_path.exists():
            raise FileNotFoundError(
                f"Source video does not exist: {source_path}"
            )

        if timestamp < 0:
            raise ValueError(
                "timestamp must not be negative."
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        command = [
            self.ffmpeg_binary,
            "-y",
            "-ss",
            str(timestamp),
            "-i",
            str(source_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output_path),
        ]

        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "FFmpeg failed to extract thumbnail frame."
            ) from exc

        if not output_path.exists():
            raise RuntimeError(
                "FFmpeg completed without producing the "
                f"expected frame: {output_path}"
            )

        return output_path