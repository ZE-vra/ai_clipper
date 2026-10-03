from pathlib import Path

import pytest

from src.thumbnails.domain.assets import (
    AssetMatch,
    AssetProvenance,
    AssetProvenanceKind,
    FrameCandidate,
    VisualAsset,
)
from src.thumbnails.domain.concepts import (
    CopyBlock,
    CopyConcept,
    CopyRole,
    ThumbnailConcept,
    VisualStrategy,
)
from src.thumbnails.domain.operations import AssetOperationKind
from src.thumbnails.intelligence.strategy_planner import StrategyPlanner


def _concept(
    strategies: tuple[VisualStrategy, ...] = (),
    *,
    visual_strategy: VisualStrategy | None = None,
) -> ThumbnailConcept:
    return ThumbnailConcept(
        concept_id="test-concept",
        title="Test concept",
        visual_idea="A clear test visual",
        curiosity_mechanism="curiosity",
        emotional_direction="surprise",
        copy=CopyConcept(
            concept_id="test-copy",
            blocks=(CopyBlock("Test hook", CopyRole.HOOK),),
        ),
        candidate_strategies=strategies,
        visual_strategy=visual_strategy,
    )


def _candidate() -> FrameCandidate:
    return FrameCandidate(
        candidate_id="candidate-frame-1",
        timestamp=1.0,
        asset=VisualAsset(
            asset_id="frame-1",
            provenance=AssetProvenance(kind=AssetProvenanceKind.SOURCE_FRAME),
            path=str(Path("frame-1.jpg")),
        ),
    )


def _match() -> AssetMatch:
    return AssetMatch(
        asset_id="frame-1",
        concept_id="test-concept",
        suitability_score=0.8,
    )


@pytest.mark.parametrize(
    ("strategy", "expected_operation"),
    (
        (VisualStrategy.ENHANCED_FRAME, AssetOperationKind.ENHANCE),
        (VisualStrategy.SUBJECT_CUTOUT, AssetOperationKind.SEGMENT),
    ),
)
def test_planner_emits_operation_for_selected_strategy(
    strategy: VisualStrategy,
    expected_operation: AssetOperationKind,
) -> None:
    plan = StrategyPlanner().plan(
        concept=_concept((strategy,)),
        match=_match(),
        candidates=(_candidate(),),
    )

    assert plan.strategy is strategy
    assert tuple(operation.kind for operation in plan.operations) == (
        AssetOperationKind.EXTRACT,
        expected_operation,
    )


def test_planner_honors_explicit_legacy_strategy() -> None:
    plan = StrategyPlanner().plan(
        concept=_concept(visual_strategy=VisualStrategy.ENHANCED_FRAME),
        match=_match(),
        candidates=(_candidate(),),
    )

    assert plan.strategy is VisualStrategy.ENHANCED_FRAME
    assert plan.operations[-1].kind is AssetOperationKind.ENHANCE


def test_planner_rejects_strategy_without_executable_operations() -> None:
    with pytest.raises(ValueError, match="no executable candidate strategy"):
        StrategyPlanner().plan(
            concept=_concept((VisualStrategy.HYBRID,)),
            match=_match(),
            candidates=(_candidate(),),
        )
