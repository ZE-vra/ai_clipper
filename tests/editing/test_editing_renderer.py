from pathlib import Path
from unittest.mock import patch

from src.editing.models import (
    BackgroundPlan,
    CanvasPlan,
    CaptionPlan,
    CaptionSegment,
    CaptionStyle,
    CompositionPlan,
    EditingPlan,
    ForegroundPlan,
    RenderConfig,
)
from src.editing.rendering.editing_renderer import EditingRenderer


def create_editing_plan(
    *,
    captions_enabled: bool = True,
) -> EditingPlan:
    return EditingPlan(
        clip_id="clip_01",
        composition=CompositionPlan(
            canvas=CanvasPlan(
                width=1080,
                height=1920,
            ),
            background=BackgroundPlan(
                source="same_video",
                blur_radius=60.0,
                brightness=0.58,
            ),
            foreground=ForegroundPlan(
                preserve_aspect_ratio=True,
                scale=1.12,
            ),
        ),
        captions=CaptionPlan(
            enabled=captions_enabled,
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
            segments=[
                CaptionSegment(
                    start_time=0.0,
                    end_time=2.0,
                    text="This is a test caption.",
                ),
            ],
        ),
        render_config=RenderConfig(),
    )


def test_renderer_builds_vertical_composition_command(tmp_path):
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"

    source.write_bytes(b"fake source")

    renderer = EditingRenderer()

    plan = create_editing_plan()

    command, caption_file = renderer._build_command(
        source_path=source,
        output_path=output,
        plan=plan,
    )

    assert command[0] == "ffmpeg"
    assert "-filter_complex" in command

    filter_complex = command[
        command.index("-filter_complex") + 1
    ]

    assert "split=2" in filter_complex
    assert "scale=1210:2150" in filter_complex
    assert "crop=1080:1920" in filter_complex
    assert "boxblur=60.0:1" in filter_complex
    assert "eq=brightness=-0.42000000000000004" in filter_complex
    assert "overlay=(W-w)/2:(H-h)/2" in filter_complex
    assert "subtitles=" in filter_complex
    assert "[final_audio]" in filter_complex
    assert "-map" in command
    assert "[final_audio]" in command

    assert caption_file is not None
    assert caption_file.exists()

    caption_file.unlink()


def test_renderer_can_build_plan_without_captions(tmp_path):
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"

    source.write_bytes(b"fake source")

    renderer = EditingRenderer()

    plan = create_editing_plan(
        captions_enabled=False,
    )

    command, caption_file = renderer._build_command(
        source_path=source,
        output_path=output,
        plan=plan,
    )

    filter_complex = command[
        command.index("-filter_complex") + 1
    ]

    assert "split=2" in filter_complex
    assert "overlay=(W-w)/2:(H-h)/2" in filter_complex
    assert "subtitles=" not in filter_complex
    assert caption_file is None


def test_renderer_creates_valid_ass_caption_file(tmp_path):
    renderer = EditingRenderer()

    plan = create_editing_plan()

    caption_file = renderer._create_ass_file(
        plan=plan,
        output_directory=tmp_path,
    )

    assert caption_file.exists()

    content = caption_file.read_text(
        encoding="utf-8"
    )

    assert "[Script Info]" in content
    assert "[V4+ Styles]" in content
    assert "[Events]" in content
    assert "Arial" in content
    assert "54" in content
    assert "This is a test caption." in content

    assert "Style: Default,Arial,54" in content
    assert ",1,4,1,2,80,80,500,1" in content

    caption_file.unlink()


def test_renderer_formats_ass_time():
    assert EditingRenderer._format_ass_time(0.0) == "0:00:00.00"
    assert EditingRenderer._format_ass_time(1.25) == "0:00:01.25"
    assert EditingRenderer._format_ass_time(65.5) == "0:01:05.50"
    assert EditingRenderer._format_ass_time(3661.75) == "1:01:01.75"


def test_renderer_maps_caption_position_to_ass_alignment():
    assert (
        EditingRenderer._ass_alignment(
            horizontal_alignment="center",
            vertical_position="lower_middle",
        )
        == 2
    )

    assert (
        EditingRenderer._ass_alignment(
            horizontal_alignment="left",
            vertical_position="top",
        )
        == 7
    )

    assert (
        EditingRenderer._ass_alignment(
            horizontal_alignment="right",
            vertical_position="bottom",
        )
        == 3
    )


def test_renderer_resolves_lower_middle_caption_margin():
    plan = create_editing_plan()

    style = plan.captions.style
    assert style is not None

    assert (
        EditingRenderer._ass_vertical_margin(
            style=style,
        )
        == 500
    )


def test_renderer_render_executes_ffmpeg_and_validates_output(tmp_path):
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"

    source.write_bytes(b"fake source")

    renderer = EditingRenderer()
    plan = create_editing_plan()

    fake_result = type(
        "FakeResult",
        (),
        {
            "returncode": 0,
            "stdout": "",
            "stderr": "",
        },
    )()

    def fake_run(command, capture_output, text):
        output.write_bytes(b"fake rendered video")
        return fake_result

    with patch(
        "src.editing.rendering.editing_renderer.subprocess.run",
        side_effect=fake_run,
    ) as mocked_run:
        with patch.object(
            renderer,
            "validate_media",
        ) as validate_media:
            result = renderer.render(
                source_path=source,
                output_path=output,
                plan=plan,
            )

    assert result == output
    assert output.exists()
    validate_media.assert_called_once_with(output)
    assert mocked_run.called


def test_renderer_rejects_missing_source(tmp_path):
    renderer = EditingRenderer()

    plan = create_editing_plan()

    missing_source = tmp_path / "missing.mp4"
    output = tmp_path / "output.mp4"

    try:
        renderer.render(
            source_path=missing_source,
            output_path=output,
            plan=plan,
        )
    except Exception as exc:
        assert "does not exist" in str(exc)
    else:
        raise AssertionError(
            "Expected renderer to reject missing source"
        )