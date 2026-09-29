import pytest

from src.thumbnails.models import (
    ThumbnailComposition,
    ThumbnailFrameCandidate,
    ThumbnailPlan,
    ThumbnailStyle,
)


def test_frame_candidate_accepts_zero_timestamp() -> None:
    candidate = ThumbnailFrameCandidate(timestamp=0.0)

    assert candidate.timestamp == 0.0


def test_frame_candidate_rejects_negative_timestamp() -> None:
    with pytest.raises(ValueError, match="timestamp"):
        ThumbnailFrameCandidate(timestamp=-0.1)


def test_style_defaults_are_valid() -> None:
    style = ThumbnailStyle()

    assert style.width == 1080
    assert style.height == 1920
    assert style.font_size == 88
    assert style.background_mode == "gradient"
    assert style.text_position == "bottom"


def test_style_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="width"):
        ThumbnailStyle(width=0)

    with pytest.raises(ValueError, match="height"):
        ThumbnailStyle(height=-1)


def test_style_rejects_invalid_font_configuration() -> None:
    with pytest.raises(ValueError, match="font_name"):
        ThumbnailStyle(font_name=" ")

    with pytest.raises(ValueError, match="font_size"):
        ThumbnailStyle(font_size=0)

    with pytest.raises(ValueError, match="text_stroke_width"):
        ThumbnailStyle(text_stroke_width=-1)


def test_composition_defaults_are_valid() -> None:
    composition = ThumbnailComposition()

    assert composition.visual_center_x == 0.5
    assert composition.visual_center_y == 0.42
    assert composition.visual_scale == 1.0
    assert composition.text_width_fraction == 0.82


def test_composition_rejects_invalid_normalized_coordinates() -> None:
    with pytest.raises(ValueError, match="visual_center_x"):
        ThumbnailComposition(visual_center_x=-0.1)

    with pytest.raises(ValueError, match="visual_center_y"):
        ThumbnailComposition(visual_center_y=1.1)

    with pytest.raises(ValueError, match="visual_scale"):
        ThumbnailComposition(visual_scale=0)


def test_composition_rejects_invalid_text_width() -> None:
    with pytest.raises(ValueError, match="text_width_fraction"):
        ThumbnailComposition(text_width_fraction=0)

    with pytest.raises(ValueError, match="text_width_fraction"):
        ThumbnailComposition(text_width_fraction=1.1)


def test_thumbnail_plan_accepts_valid_values() -> None:
    plan = ThumbnailPlan(
        clip_id=1,
        frame_timestamp=12.5,
        text="$1 PLANE?!",
        composition=ThumbnailComposition(),
        style=ThumbnailStyle(),
    )

    assert plan.clip_id == 1
    assert plan.frame_timestamp == 12.5
    assert plan.text == "$1 PLANE?!"


def test_thumbnail_plan_rejects_negative_clip_id() -> None:
    with pytest.raises(ValueError, match="clip_id"):
        ThumbnailPlan(
            clip_id=-1,
            frame_timestamp=1.0,
            text="TEST",
            composition=ThumbnailComposition(),
            style=ThumbnailStyle(),
        )


def test_thumbnail_plan_rejects_negative_frame_timestamp() -> None:
    with pytest.raises(ValueError, match="frame_timestamp"):
        ThumbnailPlan(
            clip_id=1,
            frame_timestamp=-1.0,
            text="TEST",
            composition=ThumbnailComposition(),
            style=ThumbnailStyle(),
        )


def test_thumbnail_plan_rejects_blank_text() -> None:
    with pytest.raises(ValueError, match="text"):
        ThumbnailPlan(
            clip_id=1,
            frame_timestamp=1.0,
            text="   ",
            composition=ThumbnailComposition(),
            style=ThumbnailStyle(),
        )


def test_thumbnail_models_are_immutable() -> None:
    candidate = ThumbnailFrameCandidate(timestamp=10.0)

    with pytest.raises((AttributeError, TypeError)):
        candidate.timestamp = 20.0