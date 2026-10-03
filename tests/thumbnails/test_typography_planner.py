from __future__ import annotations

import pytest

from src.thumbnails.domain.concepts import CopyBlock, CopyRole
from src.thumbnails.domain.geometry import BoundingBox, Point, Region, Size
from src.thumbnails.domain.plans import CompositionPlan, VisualPlacement
from src.thumbnails.domain.target import ThumbnailTarget, UIOcclusionRegion
from src.thumbnails.layout.typography import TypographyPlanner, TypographyPlannerConfig


def _composition(region: BoundingBox) -> CompositionPlan:
    return CompositionPlan(
        visual_placements=(
            VisualPlacement(asset_id="frame", focal_point=Point(0.8, 0.5)),
        ),
        negative_space_regions=(region,),
    )


def _target(*, safe_regions=(), occlusions=()) -> ThumbnailTarget:
    return ThumbnailTarget(
        target_id="youtube",
        platform="youtube",
        size=Size(1080, 1920),
        safe_regions=safe_regions,
        ui_occlusion_regions=occlusions,
    )


def test_payoff_gets_stronger_hierarchy_than_hook() -> None:
    plan = TypographyPlanner().plan(
        copy=(
            CopyBlock("THIS JET", CopyRole.HOOK),
            CopyBlock("$500,000", CopyRole.PAYOFF),
        ),
        composition=_composition(BoundingBox(0.0, 0.0, 0.50, 1.0)),
        target=_target(),
    )

    assert plan.blocks[1].font_size > plan.blocks[0].font_size
    assert plan.blocks[0].alignment == "left"
    assert plan.blocks[1].alignment == "left"


def test_right_side_negative_space_uses_right_alignment() -> None:
    plan = TypographyPlanner().plan(
        copy=(CopyBlock("INSIDE THE JET", CopyRole.HOOK),),
        composition=_composition(BoundingBox(0.50, 0.0, 1.0, 1.0)),
        target=_target(),
    )

    block = plan.blocks[0]
    assert block.alignment == "right"
    assert block.position.x == pytest.approx(block.text_bounds.right)


def test_center_negative_space_uses_center_alignment() -> None:
    plan = TypographyPlanner().plan(
        copy=(CopyBlock("A PRIVATE JET", CopyRole.CONTEXT),),
        composition=_composition(BoundingBox(0.25, 0.0, 0.75, 1.0)),
        target=_target(),
    )

    block = plan.blocks[0]
    assert block.alignment == "center"
    assert block.position.x == pytest.approx(0.5)


def test_copy_is_wrapped_before_rendering() -> None:
    config = TypographyPlannerConfig(base_font_size=72, maximum_font_size=100)
    plan = TypographyPlanner(config=config).plan(
        copy=(CopyBlock("THIS PRIVATE JET IS INSANE", CopyRole.HOOK),),
        composition=_composition(BoundingBox(0.0, 0.0, 0.45, 1.0)),
        target=_target(),
    )

    assert "\n" in plan.blocks[0].rendered_text
    assert len(plan.blocks[0].rendered_text.splitlines()) <= 2


def test_safe_region_intersection_constrains_text_bounds() -> None:
    safe = Region("safe", BoundingBox(0.05, 0.10, 0.35, 0.80))
    plan = TypographyPlanner().plan(
        copy=(CopyBlock("SAFE TEXT", CopyRole.HOOK),),
        composition=_composition(BoundingBox(0.0, 0.0, 0.50, 1.0)),
        target=_target(safe_regions=(safe,)),
    )

    bounds = plan.blocks[0].text_bounds
    assert bounds.left >= safe.bounds.left
    assert bounds.right <= safe.bounds.right
    assert bounds.top >= safe.bounds.top
    assert bounds.bottom <= safe.bounds.bottom


def test_ui_occlusion_pushes_text_into_remaining_space() -> None:
    occlusion = UIOcclusionRegion(
        region=Region("ui", BoundingBox(0.0, 0.0, 0.20, 1.0)),
        weight=1.0,
        reason="platform controls",
    )
    plan = TypographyPlanner().plan(
        copy=(CopyBlock("TEXT", CopyRole.HOOK),),
        composition=_composition(BoundingBox(0.0, 0.0, 0.50, 1.0)),
        target=_target(occlusions=(occlusion,)),
    )

    assert plan.blocks[0].text_bounds.left >= 0.20


def test_overflow_fails_instead_of_making_text_unreadably_small() -> None:
    config = TypographyPlannerConfig(
        base_font_size=120,
        minimum_font_size=100,
        maximum_font_size=140,
        max_lines_per_block=1,
    )

    with pytest.raises(ValueError, match="cannot fit"):
        TypographyPlanner(config=config).plan(
            copy=(CopyBlock("THIS IS AN EXTREMELY LONG THUMBNAIL HEADLINE", CopyRole.PAYOFF),),
            composition=_composition(BoundingBox(0.0, 0.0, 0.30, 0.40)),
            target=_target(),
        )


def test_invalid_typography_config_is_rejected() -> None:
    with pytest.raises(ValueError):
        TypographyPlannerConfig(
            minimum_font_size=100,
            base_font_size=80,
        )


def test_shorts_portrait_uses_large_attention_grabbing_headline_scale() -> None:
    config = TypographyPlannerConfig(
        base_font_size=100,
        maximum_font_size=100,
        minimum_font_size=42,
    )
    plan = TypographyPlanner(config=config).plan(
        copy=(CopyBlock("FREE DISNEY TRIP", CopyRole.HOOK),),
        composition=_composition(BoundingBox(0.0, 0.0, 0.60, 0.90)),
        target=_target(),
    )

    # The requested 1.05 initial scale is retained, while the fit loop
    # may reduce it to keep longer headlines inside the available region.
    assert plan.blocks[0].font_size >= 80
    assert plan.blocks[0].text_bounds.top >= 0.04
    assert plan.blocks[0].text_bounds.bottom <= 0.30
    assert plan.blocks[0].color == "#FFD700"
