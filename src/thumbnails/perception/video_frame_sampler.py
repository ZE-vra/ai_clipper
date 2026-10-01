from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.thumbnails.frame_extractor import ThumbnailFrameExtractor


@dataclass(frozen=True)
class ThumbnailFrameSample:
    timestamp: float
    path: Path


class FFprobeVideoDurationReader:
    """Read video duration without decoding the video."""

    def __init__(self, *, ffprobe_binary: str = "ffprobe") -> None:
        if not ffprobe_binary.strip():
            raise ValueError("ffprobe_binary must not be empty.")
        self.ffprobe_binary = ffprobe_binary

    def read(self, video_path: Path) -> float:
        if not video_path.is_file():
            raise FileNotFoundError(f"Source video does not exist: {video_path}")

        command = [
            self.ffprobe_binary,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]

        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise RuntimeError(
                f"Could not read video duration with {self.ffprobe_binary}."
            ) from exc

        try:
            duration = float(result.stdout.strip())
        except ValueError as exc:
            raise RuntimeError("FFprobe returned an invalid video duration.") from exc

        if duration <= 0:
            raise ValueError("Video duration must be greater than zero.")

        return duration


class ThumbnailFrameSampler:
    """Extract clean source-video candidates for thumbnail perception."""

    def __init__(
        self,
        *,
        extractor: ThumbnailFrameExtractor,
        duration_reader: Callable[[Path], float],
        sample_count: int = 9,
        start_fraction: float = 0.08,
        end_fraction: float = 0.92,
    ) -> None:
        if sample_count < 1:
            raise ValueError("sample_count must be at least 1.")
        if not 0.0 <= start_fraction < end_fraction <= 1.0:
            raise ValueError("sample fractions must satisfy 0 <= start < end <= 1.")

        self.extractor = extractor
        self.duration_reader = duration_reader
        self.sample_count = sample_count
        self.start_fraction = start_fraction
        self.end_fraction = end_fraction

    def sample(self, *, video_path: Path, output_dir: Path) -> tuple[ThumbnailFrameSample, ...]:
        duration = self.duration_reader(video_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.sample_count == 1:
            fractions = ((self.start_fraction + self.end_fraction) / 2.0,)
        else:
            step = (
                self.end_fraction - self.start_fraction
            ) / (self.sample_count - 1)
            fractions = tuple(
                self.start_fraction + step * index
                for index in range(self.sample_count)
            )

        samples: list[ThumbnailFrameSample] = []

        for index, fraction in enumerate(fractions, start=1):
            timestamp = min(duration - 0.01, max(0.0, duration * fraction))
            output_path = output_dir / f"candidate_{index:02d}.jpg"
            path = self.extractor.extract(
                source_path=video_path,
                timestamp=timestamp,
                output_path=output_path,
            )
            samples.append(
                ThumbnailFrameSample(
                    timestamp=timestamp,
                    path=path,
                )
            )

        return tuple(samples)
