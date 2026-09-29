import pytest

from src.editing.planning.caption_planner import CaptionPlanner
from src.schemas import (
    Transcript,
    TranscriptSegment,
    VideoSource,
)


def test_caption_planner_creates_caption_segments():
    transcript = Transcript(
        source=VideoSource(
            source_type="local",
            location="test.mp4",
        ),
        duration=30.0,
        language="en",
        segments=[
            TranscriptSegment(
                id=1,
                start=0.0,
                end=4.0,
                text="This is a test caption.",
            ),
        ],
    )

    planner = CaptionPlanner()

    plan = planner.create_plan(transcript)

    assert plan.enabled is True
    assert plan.style is not None
    assert plan.style.font_size == 54
    assert plan.style.font_name == "Arial"
    assert plan.style.font_weight == "bold"
    assert plan.style.horizontal_alignment == "center"
    assert plan.style.vertical_position == "lower_middle"
    assert plan.style.horizontal_margin == 80
    assert plan.style.vertical_margin == 500
    assert plan.style.max_lines == 2
    assert plan.style.outline_width == 4
    assert plan.style.shadow is True
    assert plan.segments

    assert plan.segments[0].start_time == 0.0
    assert plan.segments[0].end_time == 4.0
    assert plan.segments[0].text == "This is a test caption."


def test_caption_planner_splits_long_caption():
    transcript = Transcript(
        source=VideoSource(
            source_type="local",
            location="test.mp4",
        ),
        duration=30.0,
        language="en",
        segments=[
            TranscriptSegment(
                id=1,
                start=0.0,
                end=6.0,
                text=(
                    "This is a longer sentence that "
                    "should be split into multiple captions."
                ),
            ),
        ],
    )

    planner = CaptionPlanner()

    plan = planner.create_plan(transcript)

    assert len(plan.segments) > 1

    assert all(
        segment.start_time < segment.end_time
        for segment in plan.segments
    )

    assert plan.segments[0].start_time == 0.0
    assert plan.segments[-1].end_time == 6.0


def test_caption_planner_ignores_empty_transcript_segments():
    transcript = Transcript(
        source=VideoSource(
            source_type="local",
            location="test.mp4",
        ),
        duration=10.0,
        language="en",
        segments=[
            TranscriptSegment(
                id=1,
                start=0.0,
                end=2.0,
                text="",
            ),
            TranscriptSegment(
                id=2,
                start=2.0,
                end=4.0,
                text="   ",
            ),
        ],
    )

    planner = CaptionPlanner()

    plan = planner.create_plan(transcript)

    assert plan.segments == []


def test_caption_planner_converts_absolute_transcript_times_to_clip_relative_times():
    transcript = Transcript(
        source=VideoSource(
            source_type="local",
            location="test.mp4",
        ),
        duration=120.0,
        language="en",
        segments=[
            TranscriptSegment(
                id=1,
                start=48.0,
                end=52.0,
                text="This caption belongs to the clip.",
            ),
            TranscriptSegment(
                id=2,
                start=53.0,
                end=57.0,
                text="This one does too.",
            ),
        ],
    )

    planner = CaptionPlanner(
        max_words=100,
        max_characters=500,
    )

    plan = planner.create_plan(
        transcript,
        clip_start_time=48.0,
        clip_end_time=57.0,
    )

    assert len(plan.segments) == 2

    assert plan.segments[0].start_time == 0.0
    assert plan.segments[0].end_time == 4.0

    assert plan.segments[1].start_time == 5.0
    assert plan.segments[1].end_time == 9.0

    rendered_text = " ".join(
        segment.text
        for segment in plan.segments
    )

    assert rendered_text == (
        "This caption belongs to the clip. "
        "This one does too."
    )


def test_caption_planner_clips_transcript_segments_to_selected_range():
    transcript = Transcript(
        source=VideoSource(
            source_type="local",
            location="test.mp4",
        ),
        duration=120.0,
        language="en",
        segments=[
            TranscriptSegment(
                id=1,
                start=45.0,
                end=55.0,
                text="This sentence overlaps the clip.",
            ),
        ],
    )

    planner = CaptionPlanner(
        max_words=100,
        max_characters=500,
    )

    plan = planner.create_plan(
        transcript,
        clip_start_time=48.0,
        clip_end_time=52.0,
    )

    assert len(plan.segments) == 1

    assert plan.segments[0].start_time == 0.0
    assert plan.segments[0].end_time == 4.0
    assert (
        plan.segments[0].text
        == "This sentence overlaps the clip."
    )


def test_caption_planner_ignores_segments_outside_clip_range():
    transcript = Transcript(
        source=VideoSource(
            source_type="local",
            location="test.mp4",
        ),
        duration=120.0,
        language="en",
        segments=[
            TranscriptSegment(
                id=1,
                start=10.0,
                end=20.0,
                text="Before the clip.",
            ),
            TranscriptSegment(
                id=2,
                start=48.0,
                end=52.0,
                text="Inside the clip.",
            ),
            TranscriptSegment(
                id=3,
                start=60.0,
                end=70.0,
                text="After the clip.",
            ),
        ],
    )

    planner = CaptionPlanner(
        max_words=100,
        max_characters=500,
    )

    plan = planner.create_plan(
        transcript,
        clip_start_time=45.0,
        clip_end_time=55.0,
    )

    rendered_text = " ".join(
        segment.text
        for segment in plan.segments
    )

    assert rendered_text == "Inside the clip."


def test_caption_planner_rejects_invalid_clip_range():
    transcript = Transcript(
        source=VideoSource(
            source_type="local",
            location="test.mp4",
        ),
        duration=120.0,
        language="en",
        segments=[],
    )

    planner = CaptionPlanner()

    with pytest.raises(
        ValueError,
        match=(
            "clip_end_time must be greater than "
            "clip_start_time."
        ),
    ):
        planner.create_plan(
            transcript,
            clip_start_time=50.0,
            clip_end_time=40.0,
        )


def test_caption_planner_rejects_negative_clip_start():
    transcript = Transcript(
        source=VideoSource(
            source_type="local",
            location="test.mp4",
        ),
        duration=120.0,
        language="en",
        segments=[],
    )

    planner = CaptionPlanner()

    with pytest.raises(
        ValueError,
        match="clip_start_time must not be negative.",
    ):
        planner.create_plan(
            transcript,
            clip_start_time=-1.0,
        )