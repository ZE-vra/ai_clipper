from __future__ import annotations

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
                blur_radius=20.0,
                brightness=0.65,
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
                position="center",
                max_lines=2,
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

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream="
            "codec_type,"
            "codec_name,"
            "width,"
            "height,"
            "sample_aspect_ratio",
            "-of",
            "json",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert probe.stdout

    import json

    data = json.loads(probe.stdout)
    streams = data["streams"]

    video_streams = [
        stream
        for stream in streams
        if stream.get("codec_type") == "video"
    ]

    audio_streams = [
        stream
        for stream in streams
        if stream.get("codec_type") == "audio"
    ]

    assert len(video_streams) == 1
    assert len(audio_streams) == 1

    video = video_streams[0]

    assert video["codec_name"] == "h264"
    assert video["width"] == 1080
    assert video["height"] == 1920
    assert video["sample_aspect_ratio"] == "1:1"


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
                blur_radius=20.0,
                brightness=0.65,
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
                position="center",
                max_lines=2,
            ),
            enabled=True,
        ),
        render_config=RenderConfig(),
    )

    renderer = EditingRenderer()

    renderer.render(
        source_path=source_path,
        output_path=output_path,
        plan=plan,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    # Burned-in captions are part of the video pixels.
    # Therefore there should be no separate subtitle stream.
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "s",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert probe.stdout.strip() == ""