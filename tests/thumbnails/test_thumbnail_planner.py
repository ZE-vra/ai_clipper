from __future__ import annotations

import pytest

from src.packaging.models import ClipPackaging
from src.thumbnails.models import (
    ThumbnailComposition,
    ThumbnailStyle,
)
from src.thumbnails.thumbnail_planner import ThumbnailPlanner


def make_packaging(
    *,
    clip_id: int = 1,
    thumbnail_text: str = "FLYING FOR $1",
) -> ClipPackaging:
    return ClipPackaging(
        clip_id=clip_id,
        title="Flying On A $1 Plane Ticket",
        hook="How is a $1 plane ticket possible?",
        caption="A look inside an unusual flight.",
        description="A short clip about an unusual flight.",
        thumbnail_text=thumbnail_text,
        content_angle="unexpected travel experience",
        hashtags=["travel", "aviation"],
    )


def test_create_plan_uses_packaging_thumbnail_text() -> None:
    packaging = make_packaging(
        thumbnail_text="FLYING FOR $1"
    )

    planner = ThumbnailPlanner()

    plan = planner.create_plan(
        packaging=packaging,
        duration=100.0,
    )

    assert plan.clip_id == 1
    assert plan.text == "FLYING FOR $1"
    assert plan.frame_timestamp == pytest.approx(55.0)


def test_create_plan_uses_custom_style() -> None:
    style = ThumbnailStyle(
        font_size=100,
        background_mode="solid",
    )

    planner = ThumbnailPlanner(
        style=style,
    )

    plan = planner.create_plan(
        packaging=make_packaging(),
        duration=100.0,
    )

    assert plan.style is style
    assert plan.style.font_size == 100
    assert plan.style.background_mode == "solid"


def test_create_plan_uses_custom_composition() -> None:
    composition = ThumbnailComposition(
        visual_center_y=0.35,
        visual_scale=1.2,
    )

    planner = ThumbnailPlanner(
        composition=composition,
    )

    plan = planner.create_plan(
        packaging=make_packaging(),
        duration=100.0,
    )

    assert plan.composition is composition
    assert plan.composition.visual_center_y == 0.35
    assert plan.composition.visual_scale == 1.2


def test_empty_thumbnail_text_is_rejected() -> None:
    packaging = make_packaging(
        thumbnail_text="   "
    )

    planner = ThumbnailPlanner()

    with pytest.raises(
        ValueError,
        match="thumbnail_text must not be empty",
    ):
        planner.create_plan(
            packaging=packaging,
            duration=100.0,
        )


def test_invalid_duration_is_rejected() -> None:
    planner = ThumbnailPlanner()

    with pytest.raises(
        ValueError,
        match="duration must be greater than zero",
    ):
        planner.create_plan(
            packaging=make_packaging(),
            duration=0.0,
        )


def test_negative_duration_is_rejected() -> None:
    planner = ThumbnailPlanner()

    with pytest.raises(
        ValueError,
        match="duration must be greater than zero",
    ):
        planner.create_plan(
            packaging=make_packaging(),
            duration=-10.0,
        )


def test_clip_id_is_taken_from_packaging() -> None:
    packaging = make_packaging(
        clip_id=7,
    )

    planner = ThumbnailPlanner()

    plan = planner.create_plan(
        packaging=packaging,
        duration=100.0,
    )

    assert plan.clip_id == 7