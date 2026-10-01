from __future__ import annotations

from src.thumbnails.domain.assets import AssetProvenance, VisualAsset
from src.thumbnails.domain.concepts import CopyBlock, CopyRole
from src.thumbnails.domain.geometry import BoundingBox, Point, Size
from src.thumbnails.domain.plans import (
    CompositionPlan,
    ThumbnailRenderPlan,
    TypographyBlockPlan,
    TypographyPlan,
    VisualPlacement,
    VisualTreatmentPlan,
)
from src.thumbnails.domain.target import ThumbnailTarget, UIOcclusionRegion
from src.thumbnails.layout.negotiation import (
    LayoutCandidate,
    LayoutEvaluator,
    LayoutNegotiator,
    LayoutNegotiatorConfig,
)


def make_target(*, ui: bool = False) -> ThumbnailTarget:
    occlusions = ()
    if ui:
        occlusions = (
            UIOcclusionRegion(
                region=__import__("src.thumbnails.domain.geometry", fromlist=["Region"]).Region(
                    name="duration",
                    bounds=BoundingBox(0.80, 0.85, 1.0, 1.0),
                ),
                weight=1.0,
                reason="platform duration badge",
            ),
        )
    return ThumbnailTarget(
        target_id="shorts-1080x1920",
        platform="youtube_shorts",
        size=Size(1080, 1920),
        safe_regions=(
            __import__("src.thumbnails.domain.geometry", fromlist=["Region"]).Region(
                name="safe",
                bounds=BoundingBox(0.05, 0.05, 0.95, 0.90),
            ),
        ),
        ui_occlusion_regions=occlusions,
    )


def make_plan(
    *,
    text_bounds: BoundingBox = BoundingBox(0.05, 0.30, 0.45, 0.60),
    focal_point: Point = Point(0.75, 0.50),
    font_size: int = 100,
) -> ThumbnailRenderPlan:
    asset = VisualAsset(
        asset_id="frame-01",
        provenance=AssetProvenance.SOURCE_FRAME,
        path="frame.jpg",
    )
    composition = CompositionPlan(
        visual_placements=(
            VisualPlacement(
                asset_id=asset.asset_id,
                focal_point=focal_point,
            ),
        ),
        negative_space_regions=(text_bounds,),
    )
    copy = CopyBlock(
        text="THIS JET COSTS",
        role=CopyRole.HOOK,
    )
    typography = TypographyPlan(
        blocks=(
            TypographyBlockPlan(
                copy=copy,
                font_name="Arial",
                font_size=font_size,
                weight="bold",
                alignment="left",
                color="#FFFFFF",
                position=Point(text_bounds.left, text_bounds.center.y),
                text_bounds=text_bounds,
                rendered_text=copy.text,
                stroke_color="#000000",
                stroke_width=4,
            ),
        ),
    )
    return ThumbnailRenderPlan(
        canvas_width=1080,
        canvas_height=1920,
        composition=composition,
        typography=typography,
        visual_treatment=VisualTreatmentPlan(),
    )


def test_evaluator_accepts_clean_layout() -> None:
    evaluation = LayoutEvaluator().evaluate(
        candidate=LayoutCandidate("clean", make_plan()),
        target=make_target(),
    )

    assert evaluation.accepted
    assert evaluation.score >= 70
    assert not evaluation.hard_failures


def test_evaluator_rejects_ui_collision() -> None:
    evaluation = LayoutEvaluator().evaluate(
        candidate=LayoutCandidate(
            "ui-collision",
            make_plan(text_bounds=BoundingBox(0.80, 0.84, 0.96, 0.96)),
        ),
        target=make_target(ui=True),
    )

    assert not evaluation.accepted
    assert "protected UI" in " ".join(evaluation.hard_failures)


def test_evaluator_rejects_text_covering_focal_point() -> None:
    evaluation = LayoutEvaluator().evaluate(
        candidate=LayoutCandidate(
            "subject-collision",
            make_plan(
                text_bounds=BoundingBox(0.65, 0.35, 0.90, 0.65),
                focal_point=Point(0.75, 0.50),
            ),
        ),
        target=make_target(),
    )

    assert not evaluation.accepted
    assert "primary focal point" in " ".join(evaluation.hard_failures)


def test_evaluator_flags_long_copy_without_hard_failure() -> None:
    plan = make_plan()
    block = plan.typography.blocks[0]
    long_copy = CopyBlock(
        text="THIS JET COSTS FIVE HUNDRED THOUSAND DOLLARS",
        role=CopyRole.HOOK,
    )
    long_typography = TypographyPlan(
        blocks=(
            TypographyBlockPlan(
                copy=long_copy,
                font_name=block.font_name,
                font_size=block.font_size,
                weight=block.weight,
                alignment=block.alignment,
                color=block.color,
                position=block.position,
                text_bounds=block.text_bounds,
                rendered_text=long_copy.text,
                stroke_color=block.stroke_color,
                stroke_width=block.stroke_width,
            ),
        )
    )
    long_plan = ThumbnailRenderPlan(
        canvas_width=plan.canvas_width,
        canvas_height=plan.canvas_height,
        composition=plan.composition,
        typography=long_typography,
        visual_treatment=plan.visual_treatment,
    )

    evaluation = LayoutEvaluator().evaluate(
        candidate=LayoutCandidate("long-copy", long_plan),
        target=make_target(),
    )

    assert "thumbnail copy is too long" in evaluation.soft_failures


def test_negotiator_selects_best_acceptable_candidate() -> None:
    candidates = (
        LayoutCandidate("small", make_plan(font_size=60)),
        LayoutCandidate("strong", make_plan(font_size=120)),
    )

    result = LayoutNegotiator().negotiate(
        candidates=candidates,
        target=make_target(),
    )

    assert result.selected_candidate is not None
    assert result.selected_candidate.candidate_id == "strong"
    assert len(result.evaluations) == 2


def test_negotiator_returns_failure_when_every_candidate_is_invalid() -> None:
    result = LayoutNegotiator().negotiate(
        candidates=(
            LayoutCandidate(
                "bad",
                make_plan(
                    text_bounds=BoundingBox(0.70, 0.35, 0.95, 0.65),
                    focal_point=Point(0.80, 0.50),
                ),
            ),
        ),
        target=make_target(),
    )

    assert result.selected_candidate is None
    assert result.failure_reason
