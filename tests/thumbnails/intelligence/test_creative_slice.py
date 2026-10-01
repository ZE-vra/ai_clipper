from pathlib import Path
from types import SimpleNamespace

from src.thumbnails.domain.assets import FrameCandidate, VisualAsset, AssetProvenance, AssetProvenanceKind
from src.thumbnails.domain.brief import ThumbnailBrief
from src.thumbnails.domain.content import ContentEvent, ContentUnderstanding
from src.thumbnails.domain.concepts import VisualStrategy
from src.thumbnails.intelligence.asset_matcher import AssetMatcher
from src.thumbnails.intelligence.creative_director import RuleBasedCreativeDirector
from src.thumbnails.intelligence.strategy_planner import StrategyPlanner


def _candidate(asset_id: str, score: float = 0.8) -> FrameCandidate:
    perception = SimpleNamespace(
        quality=SimpleNamespace(overall_quality=score),
        crop=SimpleNamespace(score=score),
        focal=SimpleNamespace(
            regions=(SimpleNamespace(strength=score),)
        ),
        subjects=SimpleNamespace(subjects=(object(),)),
    )
    asset = VisualAsset(
        asset_id=asset_id,
        provenance=AssetProvenance(
            kind=AssetProvenanceKind.SOURCE_FRAME,
        ),
        path=str(Path(f"{asset_id}.jpg")),
    )
    return FrameCandidate(
        candidate_id=f"candidate-{asset_id}",
        timestamp=10.0,
        asset=asset,
        perception=perception,
    )


def _brief() -> ThumbnailBrief:
    return ThumbnailBrief(
        core_hook="$2M CARPET ON A PLANE?!",
        subject="private jet",
        promise="A $2 million carpet is revealed.",
        curiosity_angle="Why is the carpet worth $2 million?",
        emotional_direction="surprise",
        important_entities=("private jet",),
        important_objects=("carpet",),
        visual_evidence=("luxury jet interior",),
    )


def test_creative_director_does_not_select_a_frame() -> None:
    understanding = ContentUnderstanding(
        entities=(),
        events=(
            ContentEvent(
                event_id="event-1",
                start_time=10.0,
                end_time=20.0,
                description="The carpet price is revealed.",
                importance=1.0,
                confidence=0.9,
            ),
        ),
        confidence=0.9,
    )

    concepts = RuleBasedCreativeDirector().create(
        brief=_brief(),
        understanding=understanding,
        max_concepts=3,
    )

    assert concepts
    assert all(concept.visual_strategy is None for concept in concepts)
    assert all(concept.candidate_strategies for concept in concepts)


def test_asset_matcher_ranks_candidates_without_discarding_the_pool() -> None:
    concepts = RuleBasedCreativeDirector().create(
        brief=_brief(),
        understanding=ContentUnderstanding(entities=(), events=()),
        max_concepts=1,
    )
    candidates = (_candidate("frame-a", 0.4), _candidate("frame-b", 0.9))

    matches = AssetMatcher().match(
        concept=concepts[0],
        candidates=candidates,
    )

    assert len(matches) == 2
    assert matches[0].asset_id == "frame-b"


def test_strategy_planner_creates_an_executable_source_plan() -> None:
    concepts = RuleBasedCreativeDirector().create(
        brief=_brief(),
        understanding=ContentUnderstanding(entities=(), events=()),
        max_concepts=1,
    )
    candidate = _candidate("frame-a")
    match = AssetMatcher().match(
        concept=concepts[0],
        candidates=(candidate,),
    )[0]

    plan = StrategyPlanner().plan(
        concept=concepts[0],
        match=match,
        candidates=(candidate,),
    )

    assert plan.source_asset_ids == ("frame-a",)
    assert plan.strategy is VisualStrategy.SOURCE_FRAME
    assert plan.plan_fingerprint
