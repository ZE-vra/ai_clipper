from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from PIL import Image

from src.thumbnails.frame_extractor import FFmpegFrameExtractor


@pytest.mark.integration
def test_extracts_real_frame_with_ffmpeg(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "frame.jpg"

    create_command = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=red:s=640x360:d=2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(source),
    ]

    subprocess.run(
        create_command,
        check=True,
        capture_output=True,
        text=True,
    )

    extractor = FFmpegFrameExtractor()

    result = extractor.extract(
        source_path=source,
        timestamp=1.0,
        output_path=output,
    )

    assert result == output
    assert output.exists()
    assert output.stat().st_size > 0

    with Image.open(output) as image:
        assert image.format == "JPEG"
        assert image.size == (640, 360)