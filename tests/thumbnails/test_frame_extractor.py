from __future__ import annotations

from pathlib import Path
import subprocess

import pytest
from src.thumbnails.frame_extractor import FFmpegFrameExtractor


def test_empty_ffmpeg_binary_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="ffmpeg_binary must not be empty",
    ):
        FFmpegFrameExtractor(
            ffmpeg_binary="   "
        )


def test_missing_source_is_rejected(
    tmp_path: Path,
) -> None:
    extractor = FFmpegFrameExtractor()

    with pytest.raises(
        FileNotFoundError,
        match="Source video does not exist",
    ):
        extractor.extract(
            source_path=tmp_path / "missing.mp4",
            timestamp=10.0,
            output_path=tmp_path / "frame.jpg",
        )


def test_negative_timestamp_is_rejected(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fake video")

    extractor = FFmpegFrameExtractor()

    with pytest.raises(
        ValueError,
        match="timestamp must not be negative",
    ):
        extractor.extract(
            source_path=source,
            timestamp=-1.0,
            output_path=tmp_path / "frame.jpg",
        )


def test_output_directory_is_created(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fake video")

    def fake_run(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> None:
        output_path = Path(command[-1])
        output_path.write_bytes(b"fake image")

    monkeypatch.setattr(
        "src.thumbnails.frame_extractor.subprocess.run",
        fake_run,
    )

    output = tmp_path / "nested" / "frames" / "frame.jpg"

    extractor = FFmpegFrameExtractor()

    result = extractor.extract(
        source_path=source,
        timestamp=10.0,
        output_path=output,
    )

    assert result == output
    assert output.exists()


def test_ffmpeg_command_is_constructed_correctly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fake video")

    captured_command: list[str] = []

    def fake_run(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> None:
        captured_command.extend(command)

        output_path = Path(command[-1])
        output_path.write_bytes(b"fake image")

    monkeypatch.setattr(
        "src.thumbnails.frame_extractor.subprocess.run",
        fake_run,
    )

    output = tmp_path / "frame.jpg"

    extractor = FFmpegFrameExtractor(
        ffmpeg_binary="custom-ffmpeg",
    )

    extractor.extract(
        source_path=source,
        timestamp=42.5,
        output_path=output,
    )

    assert captured_command == [
        "custom-ffmpeg",
        "-y",
        "-ss",
        "42.5",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output),
    ]


def test_ffmpeg_failure_is_wrapped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fake video")

    def fake_run(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> None:
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=command,
            stderr="ffmpeg failed",
        )

    monkeypatch.setattr(
        "src.thumbnails.frame_extractor.subprocess.run",
        fake_run,
    )

    extractor = FFmpegFrameExtractor()

    with pytest.raises(
        RuntimeError,
        match="FFmpeg failed to extract thumbnail frame",
    ):
        extractor.extract(
            source_path=source,
            timestamp=10.0,
            output_path=tmp_path / "frame.jpg",
        )