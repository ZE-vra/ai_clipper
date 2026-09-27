import pytest

from src.rendering.ffmpeg_renderer import (
    FFmpegRenderer,
    RenderingError,
)


def test_renderer_rejects_missing_source(tmp_path):
    renderer = FFmpegRenderer()

    missing_source = (
        tmp_path / "missing_source.mp4"
    )

    output_path = (
        tmp_path / "output.mp4"
    )

    with pytest.raises(RenderingError):
        renderer.render_clip(
            source_path=missing_source,
            output_path=output_path,
            start_time=0.0,
            end_time=10.0,
        )