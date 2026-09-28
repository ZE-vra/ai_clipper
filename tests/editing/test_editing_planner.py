from src.editing.models import RenderConfig
from src.editing.planning.caption_planner import CaptionPlanner
from src.editing.planning.editing_planner import EditingPlanner
from src.schemas import Transcript, TranscriptSegment


def test_editing_planner_assembles_complete_editing_plan():
    transcript = Transcript(
        source="test_video",
        duration=6.0,
        language="en",
        segments=[
            TranscriptSegment(
                id="segment_001",
                start=0.0,
                end=6.0,
                text=(
                    "The biggest mistake I made when I started "
                    "my business was trying to do everything myself."
                ),
            )
        ],
    )

    planner = EditingPlanner()

    plan = planner.create_plan(
        clip_id="clip_01",
        transcript=transcript,
    )

    assert plan.clip_id == "clip_01"

    assert plan.composition.canvas.width == 1080
    assert plan.composition.canvas.height == 1920

    assert plan.composition.background.source == "same_video"
    assert plan.composition.foreground.preserve_aspect_ratio is True

    assert plan.captions.enabled is True
    assert len(plan.captions.segments) > 0

    assert plan.render_config.video_codec == "libx264"
    assert plan.render_config.audio_codec == "aac"
    assert plan.render_config.output_format == "mp4"


def test_editing_planner_preserves_transcript_content():
    transcript = Transcript(
        source="test_video",
        duration=3.0,
        language="en",
        segments=[
            TranscriptSegment(
                id="segment_001",
                start=0.0,
                end=3.0,
                text="This is a simple test.",
            )
        ],
    )

    planner = EditingPlanner(
        caption_planner=CaptionPlanner(
            max_words=100,
            max_characters=500,
        )
    )

    plan = planner.create_plan(
        clip_id="clip_01",
        transcript=transcript,
    )

    output_words = " ".join(
        segment.text
        for segment in plan.captions.segments
    ).split()

    assert output_words == [
        "This",
        "is",
        "a",
        "simple",
        "test.",
    ]


def test_editing_planner_converts_absolute_transcript_times_to_clip_relative_times():
    transcript = Transcript(
        source="test_video",
        duration=60.0,
        language="en",
        segments=[
            TranscriptSegment(
                id="segment_001",
                start=20.0,
                end=24.0,
                text="First part of the selected clip.",
            ),
            TranscriptSegment(
                id="segment_002",
                start=25.0,
                end=29.0,
                text="Second part of the selected clip.",
            ),
        ],
    )

    planner = EditingPlanner(
        caption_planner=CaptionPlanner(
            max_words=100,
            max_characters=500,
        )
    )

    plan = planner.create_plan(
        clip_id="clip_01",
        transcript=transcript,
        clip_start_time=20.0,
        clip_end_time=29.0,
    )

    assert len(plan.captions.segments) == 2

    assert plan.captions.segments[0].start_time == 0.0
    assert plan.captions.segments[0].end_time == 4.0

    assert plan.captions.segments[1].start_time == 5.0
    assert plan.captions.segments[1].end_time == 9.0


def test_editing_planner_passes_clip_range_to_caption_planner():
    transcript = Transcript(
        source="test_video",
        duration=60.0,
        language="en",
        segments=[
            TranscriptSegment(
                id="segment_001",
                start=10.0,
                end=14.0,
                text="Before selected clip.",
            ),
            TranscriptSegment(
                id="segment_002",
                start=20.0,
                end=24.0,
                text="Inside selected clip.",
            ),
            TranscriptSegment(
                id="segment_003",
                start=30.0,
                end=34.0,
                text="After selected clip.",
            ),
        ],
    )

    planner = EditingPlanner(
        caption_planner=CaptionPlanner(
            max_words=100,
            max_characters=500,
        )
    )

    plan = planner.create_plan(
        clip_id="clip_01",
        transcript=transcript,
        clip_start_time=18.0,
        clip_end_time=26.0,
    )

    rendered_text = " ".join(
        segment.text
        for segment in plan.captions.segments
    )

    assert rendered_text == "Inside selected clip."


def test_editing_planner_preserves_custom_render_config():
    transcript = Transcript(
        source="test_video",
        duration=3.0,
        language="en",
        segments=[],
    )

    render_config = RenderConfig(
        video_codec="libx264",
        audio_codec="aac",
        crf=22,
        audio_bitrate="96k",
        output_format="mp4",
    )

    planner = EditingPlanner(
        render_config=render_config,
    )

    plan = planner.create_plan(
        clip_id="clip_01",
        transcript=transcript,
    )

    assert plan.render_config is render_config
    assert plan.render_config.crf == 22
    assert plan.render_config.audio_bitrate == "96k"


def test_editing_planner_rejects_empty_clip_id():
    transcript = Transcript(
        source="test_video",
        duration=1.0,
        language="en",
        segments=[],
    )

    planner = EditingPlanner()

    try:
        planner.create_plan(
            clip_id="",
            transcript=transcript,
        )
    except ValueError as exc:
        assert str(exc) == "clip_id must not be empty."
    else:
        raise AssertionError(
            "Expected ValueError for empty clip_id"
        )


def test_editing_planner_rejects_negative_clip_start():
    transcript = Transcript(
        source="test_video",
        duration=10.0,
        language="en",
        segments=[],
    )

    planner = EditingPlanner()

    try:
        planner.create_plan(
            clip_id="clip_01",
            transcript=transcript,
            clip_start_time=-1.0,
        )
    except ValueError as exc:
        assert str(exc) == (
            "clip_start_time must not be negative."
        )
    else:
        raise AssertionError(
            "Expected ValueError for negative clip start"
        )


def test_editing_planner_rejects_invalid_clip_range():
    transcript = Transcript(
        source="test_video",
        duration=10.0,
        language="en",
        segments=[],
    )

    planner = EditingPlanner()

    try:
        planner.create_plan(
            clip_id="clip_01",
            transcript=transcript,
            clip_start_time=8.0,
            clip_end_time=4.0,
        )
    except ValueError as exc:
        assert str(exc) == (
            "clip_end_time must be greater than "
            "clip_start_time."
        )
    else:
        raise AssertionError(
            "Expected ValueError for invalid clip range"
        )