from pathlib import Path

from PIL import Image

from src.thumbnails.domain.geometry import BoundingBox, Point, Region, Size
from src.thumbnails.domain.plans import (
    CompositionPlan,
    ThumbnailRenderPlan,
    TypographyPlan,
    VisualPlacement,
    VisualTreatmentPlan,
)
from src.thumbnails.domain.target import ThumbnailTarget
from src.thumbnails.evaluation.deterministic import DeterministicThumbnailEvaluator


def _target() -> ThumbnailTarget:
    return ThumbnailTarget(
        target_id="youtube",
        platform="youtube",
        size=Size(width=1280, height=720),
        safe_regions=(
            Region(
                name="safe",
                bounds=BoundingBox(0.0, 0.0, 1.0, 1.0),
            ),
        ),
    )


def _plan() -> ThumbnailRenderPlan:
    return ThumbnailRenderPlan(
        canvas_width=1280,
        canvas_height=720,
        composition=CompositionPlan(
            visual_placements=(
                VisualPlacement(
                    asset_id="frame-001",
                    focal_point=Point(0.5, 0.5),
                    scale=1.0,
                    center=Point(0.5, 0.5),
                    crop_bounds=BoundingBox(0.0, 0.0, 1.0, 1.0),
                ),
            ),
            negative_space_regions=(
                BoundingBox(0.0, 0.0, 0.4, 1.0),
            ),
        ),
        typography=TypographyPlan(blocks=()),
        visual_treatment=VisualTreatmentPlan(),
    )


def test_deterministic_evaluator_accepts_valid_render(tmp_path: Path) -> None:
    output = tmp_path / "thumbnail.jpg"
    Image.new("RGB", (1280, 720), (20, 20, 20)).save(output)

    evaluation = DeterministicThumbnailEvaluator().evaluate(
        output_path=output,
        plan=_plan(),
        target=_target(),
    )

    assert evaluation.accepted
    assert evaluation.technical_score == 1.0
    assert evaluation.hard_failures == ()


def test_deterministic_evaluator_rejects_wrong_dimensions(tmp_path: Path) -> None:
    output = tmp_path / "thumbnail.jpg"
    Image.new("RGB", (720, 1280), (20, 20, 20)).save(output)

    evaluation = DeterministicThumbnailEvaluator().evaluate(
        output_path=output,
        plan=_plan(),
        target=_target(),
    )

    assert not evaluation.accepted
    assert evaluation.technical_score == 0.0
    assert evaluation.hard_failures
