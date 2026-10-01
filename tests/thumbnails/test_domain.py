from __future__ import annotations

import pytest

from src.thumbnails.domain.assets import (
    AssetProvenance,
    FrameCandidate,
    VisualAsset,
)
from src.thumbnails.domain.attempts import (
    AttemptStatus,
    ThumbnailAttempt,
)
from src.thumbnails.domain.evaluation import ThumbnailEvaluation
from src.thumbnails.domain.brief import CreativeBrief
from src.thumbnails.domain.concepts import (
    CopyBlock,
    CopyConcept,
    CopyRole,
    ThumbnailConcept,
    VisualStrategy,
)
from src.thumbnails.domain.geometry import (
    BoundingBox,
    Point,
    Region,
    Size,
)
from src.thumbnails.domain.plans import (
    CompositionPlan,
    ThumbnailRenderPlan,
    TypographyBlockPlan,
    TypographyPlan,
    VisualPlacement,
    VisualTreatmentPlan,
)
from src.thumbnails.domain.results import (
    ThumbnailResult,
    ThumbnailResultStatus,
)
from src.thumbnails.domain.target import (
    ThumbnailTarget,
    UIOcclusionRegion,
)


def make_copy_concept() -> CopyConcept:
    return CopyConcept(
        concept_id="copy-01",
        blocks=(
            CopyBlock(
                text="THIS JET COST",
                role=CopyRole.HOOK,
            ),
            CopyBlock(
                text="$500,000",
                role=CopyRole.PAYOFF,
                emphasis=1.4,
            ),
        ),
    )


def make_thumbnail_concept() -> ThumbnailConcept:
    return ThumbnailConcept(
        concept_id="concept-01",
        visual_strategy=VisualStrategy.SOURCE_FRAME,
        copy=make_copy_concept(),
    )


def make_asset() -> VisualAsset:
    return VisualAsset(
        asset_id="frame-01",
        provenance=AssetProvenance.SOURCE_FRAME,
        path="frame.jpg",
        source_timestamp=30.0,
    )


def make_render_plan() -> ThumbnailRenderPlan:
    asset = make_asset()

    placement = VisualPlacement(
        asset_id=asset.asset_id,
        focal_point=Point(0.65, 0.45),
        scale=1.2,
        center=Point(0.55, 0.5),
    )

    composition = CompositionPlan(
        visual_placements=(placement,),
        negative_space_regions=(
            BoundingBox(
                left=0.05,
                top=0.05,
                right=0.45,
                bottom=0.35,
            ),
        ),
    )

    typography = TypographyPlan(
        blocks=(
            TypographyBlockPlan(
                copy=CopyBlock(
                    text="THIS JET COST",
                    role=CopyRole.HOOK,
                ),
                font_name="Arial",
                font_size=72,
                weight="bold",
                alignment="left",
                color="#FFFFFF",
                position=Point(0.1, 0.1),
                text_bounds=BoundingBox(
                    left=0.05,
                    top=0.05,
                    right=0.45,
                    bottom=0.15,
                ),
                stroke_color="#000000",
                stroke_width=4,
            ),
        ),
    )

    treatment = VisualTreatmentPlan(
        contrast=1.1,
        saturation=1.1,
        sharpness=1.05,
        vignette=0.2,
        overlay_opacity=0.1,
    )

    return ThumbnailRenderPlan(
        canvas_width=1080,
        canvas_height=1920,
        composition=composition,
        typography=typography,
        visual_treatment=treatment,
        background_asset=asset,
    )


class TestGeometry:
    def test_point_accepts_normalized_coordinates(self):
        point = Point(0.5, 0.25)

        assert point.x == 0.5
        assert point.y == 0.25

    @pytest.mark.parametrize(
        "x,y",
        [
            (-0.01, 0.5),
            (1.01, 0.5),
            (0.5, -0.01),
            (0.5, 1.01),
        ],
    )
    def test_point_rejects_coordinates_outside_normalized_space(
        self,
        x,
        y,
    ):
        with pytest.raises(ValueError):
            Point(x, y)

    def test_bounding_box_calculates_geometry(self):
        box = BoundingBox(
            left=0.1,
            top=0.2,
            right=0.7,
            bottom=0.8,
        )

        assert box.width == pytest.approx(0.6)
        assert box.height == pytest.approx(0.6)
        assert box.center.x == pytest.approx(0.4)
        assert box.center.y == pytest.approx(0.5)

    def test_bounding_box_rejects_inverted_bounds(self):
        with pytest.raises(ValueError):
            BoundingBox(
                left=0.8,
                top=0.2,
                right=0.7,
                bottom=0.8,
            )

    def test_region_requires_name(self):
        with pytest.raises(ValueError):
            Region(
                name="",
                bounds=BoundingBox(
                    0.0,
                    0.0,
                    0.5,
                    0.5,
                ),
            )

    def test_size_requires_positive_dimensions(self):
        with pytest.raises(ValueError):
            Size(0, 1920)

        with pytest.raises(ValueError):
            Size(1080, 0)


class TestTarget:
    def test_target_accepts_safe_and_ui_regions(self):
        safe_region = Region(
            name="top_text",
            bounds=BoundingBox(
                0.05,
                0.05,
                0.95,
                0.35,
            ),
        )

        ui_region = UIOcclusionRegion(
            region=Region(
                name="bottom_ui",
                bounds=BoundingBox(
                    0.0,
                    0.8,
                    1.0,
                    1.0,
                ),
            ),
            weight=0.8,
            reason="Platform controls may cover this area.",
        )

        target = ThumbnailTarget(
            target_id="youtube-shorts",
            platform="youtube",
            size=Size(1080, 1920),
            safe_regions=(safe_region,),
            ui_occlusion_regions=(ui_region,),
        )

        assert target.size.width == 1080
        assert target.size.height == 1920
        assert target.safe_regions == (safe_region,)
        assert target.ui_occlusion_regions == (ui_region,)

    def test_ui_occlusion_weight_is_normalized(self):
        with pytest.raises(ValueError):
            UIOcclusionRegion(
                region=Region(
                    name="ui",
                    bounds=BoundingBox(
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                    ),
                ),
                weight=1.5,
                reason="test",
            )


class TestCreativeDomain:
    def test_creative_brief_requires_core_fields(self):
        brief = CreativeBrief(
            core_hook="This jet costs $500,000.",
            subject="Private jet",
            promise="Show what the jet is like inside.",
            curiosity_angle="What does a $500,000 jet actually look like?",
        )

        assert brief.subject == "Private jet"

    def test_copy_concept_preserves_copy_hierarchy(self):
        concept = make_copy_concept()

        assert concept.blocks[0].role is CopyRole.HOOK
        assert concept.blocks[1].role is CopyRole.PAYOFF
        assert concept.blocks[1].emphasis > concept.blocks[0].emphasis

    def test_thumbnail_concept_contains_strategy_and_copy(self):
        concept = make_thumbnail_concept()

        assert concept.visual_strategy is VisualStrategy.SOURCE_FRAME
        assert concept.copy.concept_id == "copy-01"


class TestAssets:
    def test_frame_candidate_references_visual_asset(self):
        asset = make_asset()

        candidate = FrameCandidate(
            candidate_id="candidate-01",
            timestamp=30.0,
            asset=asset,
        )

        assert candidate.asset.asset_id == "frame-01"
        assert candidate.timestamp == 30.0

    def test_visual_asset_rejects_negative_timestamp(self):
        with pytest.raises(ValueError):
            VisualAsset(
                asset_id="frame-01",
                provenance=AssetProvenance.SOURCE_FRAME,
                path="frame.jpg",
                source_timestamp=-1.0,
            )


class TestPlans:
    def test_render_plan_is_complete(self):
        plan = make_render_plan()

        assert plan.canvas_width == 1080
        assert plan.canvas_height == 1920
        assert len(plan.composition.visual_placements) == 1
        assert len(plan.typography.blocks) == 1

    def test_visual_placement_requires_positive_scale(self):
        with pytest.raises(ValueError):
            VisualPlacement(
                asset_id="frame-01",
                focal_point=Point(0.5, 0.5),
                scale=0,
            )

    def test_typography_rejects_invalid_alignment(self):
        with pytest.raises(ValueError):
            TypographyBlockPlan(
                copy=CopyBlock(
                    text="TEST",
                    role=CopyRole.HOOK,
                ),
                font_name="Arial",
                font_size=72,
                weight="bold",
                alignment="diagonal",
                color="#FFFFFF",
                position=Point(0.5, 0.5),
                text_bounds=BoundingBox(
                    left=0.25,
                    top=0.45,
                    right=0.75,
                    bottom=0.55,
                ),
            )

    def test_visual_treatment_rejects_invalid_opacity(self):
        with pytest.raises(ValueError):
            VisualTreatmentPlan(
                overlay_opacity=1.5,
            )


class TestAttempts:
    def test_planned_attempt_does_not_require_evaluation(self):
        attempt = ThumbnailAttempt(
            attempt_id="attempt-01",
            concept=make_thumbnail_concept(),
            render_plan=make_render_plan(),
            status=AttemptStatus.PLANNED,
        )

        assert attempt.evaluation is None

    def test_accepted_attempt_requires_evaluation(self):
        with pytest.raises(ValueError):
            ThumbnailAttempt(
                attempt_id="attempt-01",
                concept=make_thumbnail_concept(),
                render_plan=make_render_plan(),
                status=AttemptStatus.ACCEPTED,
            )

    def test_rejected_attempt_can_record_failure_reasons(self):
        evaluation = ThumbnailEvaluation(
            accepted=False,
            hard_failures=("face_cropped",),
            soft_failures=("weak_text_hierarchy",),
            score=0.41,
            reason="The selected crop damages the primary subject.",
        )

        attempt = ThumbnailAttempt(
            attempt_id="attempt-02",
            concept=make_thumbnail_concept(),
            render_plan=make_render_plan(),
            status=AttemptStatus.REJECTED,
            evaluation=evaluation,
        )

        assert attempt.evaluation is evaluation
        assert "face_cropped" in evaluation.hard_failures


class TestResults:
    def test_success_requires_output_and_attempt(self):
        result = ThumbnailResult(
            status=ThumbnailResultStatus.SUCCESS,
            output_path="thumbnail.jpg",
            selected_attempt_id="attempt-01",
        )

        assert result.output_path == "thumbnail.jpg"

    def test_success_requires_output_path(self):
        with pytest.raises(ValueError):
            ThumbnailResult(
                status=ThumbnailResultStatus.SUCCESS,
                selected_attempt_id="attempt-01",
            )

    def test_no_acceptable_result_requires_reason(self):
        with pytest.raises(ValueError):
            ThumbnailResult(
                status=ThumbnailResultStatus.NO_ACCEPTABLE_RESULT,
            )

    def test_no_acceptable_result_records_reason(self):
        result = ThumbnailResult(
            status=ThumbnailResultStatus.NO_ACCEPTABLE_RESULT,
            failure_reason="No candidate survived the quality gate.",
        )

        assert result.failure_reason == (
            "No candidate survived the quality gate."
        )
