from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.editing.models import (
    BackgroundPlan,
    CaptionPlan,
    CaptionSegment,
    CaptionStyle,
    CanvasPlan,
    CompositionPlan,
    EditingPlan,
    ForegroundPlan,
    RenderConfig,
)
from src.editing.rendering.editing_renderer import EditingRenderer


def _probe_streams(media_path: Path) -> list[dict]:
    """Return FFprobe stream metadata for a rendered media file."""

    ffprobe_path = shutil.which("ffprobe")

    if ffprobe_path is None:
        pytest.fail(
            "FFprobe is required for this integration test but was not "
            "found on PATH."
        )

    result = subprocess.run(
        [
            ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_type,width,height",
            "-of",
            "json",
            str(media_path),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        "FFprobe failed while inspecting rendered output:\n"
        f"{result.stderr}"
    )

    payload = json.loads(result.stdout)

    return payload.get("streams", [])


def _video_stream(streams: list[dict]) -> dict:
    video_streams = [
        stream
        for stream in streams
        if stream.get("codec_type") == "video"
    ]

    assert video_streams, "Rendered output does not contain a video stream."

    return video_streams[0]


def _has_audio_stream(streams: list[dict]) -> bool:
    return any(
        stream.get("codec_type") == "audio"
        for stream in streams
    )


def _has_subtitle_stream(streams: list[dict]) -> bool:
    return any(
        stream.get("codec_type") == "subtitle"
        for stream in streams
    )


@pytest.mark.integration
def test_editing_renderer_produces_valid_vertical_video(
    tmp_path: Path,
) -> None:
    source_path = Path("test_clip.mp4")

    if not source_path.exists():
        pytest.fail(
            f"Required integration fixture does not exist: {source_path}"
        )

    output_path = tmp_path / "edited_clip.mp4"

    plan = EditingPlan(
        clip_id="integration_test",
        composition=CompositionPlan(
            canvas=CanvasPlan(
                width=1080,
                height=1920,
            ),
            background=BackgroundPlan(
                source="same_video",
                blur_radius=32.0,
                brightness=0.58,
            ),
            foreground=ForegroundPlan(
                preserve_aspect_ratio=True,
                scale=1.0,
            ),
        ),
        captions=CaptionPlan(
            segments=[
                CaptionSegment(
                    start_time=1.0,
                    end_time=3.0,
                    text="This is a real FFmpeg integration test.",
                ),
                CaptionSegment(
                    start_time=4.0,
                    end_time=6.0,
                    text="The captions should be burned into the video.",
                ),
            ],
            style=CaptionStyle(
                font_size=54,
                font_name="Arial",
                font_weight="bold",
                horizontal_alignment="center",
                vertical_position="lower_middle",
                horizontal_margin=80,
                vertical_margin=500,
                max_lines=2,
                outline_width=4,
                shadow=True,
            ),
            enabled=True,
        ),
        render_config=RenderConfig(),
    )

    renderer = EditingRenderer()

    result = renderer.render(
        source_path=source_path,
        output_path=output_path,
        plan=plan,
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    # The renderer's public validation contract.
    renderer.validate_media(output_path)

    streams = _probe_streams(output_path)
    video = _video_stream(streams)

    assert video["width"] == 1080
    assert video["height"] == 1920
    assert _has_audio_stream(streams)


@pytest.mark.integration
def test_editing_renderer_burns_caption_into_real_output(
    tmp_path: Path,
) -> None:
    source_path = Path("test_clip.mp4")
    output_path = tmp_path / "captioned_clip.mp4"

    if not source_path.exists():
        pytest.fail(
            f"Required integration fixture does not exist: {source_path}"
        )

    plan = EditingPlan(
        clip_id="caption_integration_test",
        composition=CompositionPlan(
            canvas=CanvasPlan(
                width=1080,
                height=1920,
            ),
            background=BackgroundPlan(
                source="same_video",
                blur_radius=32.0,
                brightness=0.58,
            ),
            foreground=ForegroundPlan(
                preserve_aspect_ratio=True,
                scale=1.0,
            ),
        ),
        captions=CaptionPlan(
            segments=[
                CaptionSegment(
                    start_time=1.0,
                    end_time=3.0,
                    text="VISIBLE INTEGRATION CAPTION",
                ),
            ],
            style=CaptionStyle(
                font_size=54,
                font_name="Arial",
                font_weight="bold",
                horizontal_alignment="center",
                vertical_position="lower_middle",
                horizontal_margin=80,
                vertical_margin=500,
                max_lines=2,
                outline_width=4,
                shadow=True,
            ),
            enabled=True,
        ),
        render_config=RenderConfig(),
    )

    renderer = EditingRenderer()

    result = renderer.render(
        source_path=source_path,
        output_path=output_path,
        plan=plan,
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    # The renderer burns captions into the video during rendering.
    renderer.validate_media(output_path)

    streams = _probe_streams(output_path)
    video = _video_stream(streams)

    assert video["width"] == 1080
    assert video["height"] == 1920
    assert _has_audio_stream(streams)

    # Captions are burned into the video rather than stored as a subtitle
    # stream, which keeps the final artifact self-contained for social use.
    assert not _has_subtitle_stream(streams)