from pathlib import Path
from types import SimpleNamespace

from src.thumbnails.domain.assets import FrameCandidate, VisualAsset, AssetProvenance, AssetProvenanceKind
from src.thumbnails.domain.brief import ThumbnailBrief
from src.thumbnails.domain.content import ContentEntity, ContentEvent, ContentUnderstanding
from src.thumbnails.domain.concepts import VisualStrategy, ThumbnailConcept, CopyConcept, CopyBlock, CopyRole
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


def _candidate_with_subjects(asset_id: str, subjects: list[any], score: float = 0.8) -> FrameCandidate:
    perception = SimpleNamespace(
        quality=SimpleNamespace(overall_quality=score),
        crop=SimpleNamespace(score=score),
        focal=SimpleNamespace(
            regions=(SimpleNamespace(strength=score),)
        ),
        subjects=SimpleNamespace(subjects=tuple(subjects)),
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


def test_creative_director_grounds_event_description_and_entities() -> None:
    understanding = ContentUnderstanding(
        entities=(
            ContentEntity(entity_id="ent-carpet", label="Golden Carpet", kind="object", confidence=0.95),
            ContentEntity(entity_id="ent-jet", label="Gulfstream G650", kind="vehicle", confidence=0.9),
        ),
        events=(
            ContentEvent(
                event_id="event-1",
                start_time=10.0,
                end_time=20.0,
                description="The carpet price is revealed.",
                entity_ids=("ent-carpet", "ent-jet", "missing-entity-id"),
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

    # Find the event-reveal concept
    event_concept = next(c for c in concepts if c.concept_id == "event-reveal")

    # Verify event description grounding
    assert "The carpet price is revealed." in event_concept.title
    assert "The carpet price is revealed." in event_concept.visual_idea

    # Verify entity label resolution and missing entity handling
    assert "Golden Carpet" in event_concept.visual_idea
    assert "Gulfstream G650" in event_concept.visual_idea
    assert "missing-entity-id" not in event_concept.visual_idea


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


def test_asset_matcher_ranks_candidate_with_matching_evidence_higher() -> None:
    copy_concept = CopyConcept(
        concept_id="test-copy",
        blocks=(CopyBlock(text="test text", role=CopyRole.HOOK),),
    )
    concept = ThumbnailConcept(
        concept_id="test-concept",
        title="Test Concept",
        visual_idea="Test visual idea",
        curiosity_mechanism="test curiosity",
        emotional_direction="test emotion",
        copy=copy_concept,
        required_visual_evidence=("luxury jet interior",),
        preferred_entities=("private jet",),
        candidate_strategies=(VisualStrategy.SOURCE_FRAME,),
    )

    # Candidate A has matching evidence
    subject_a1 = SimpleNamespace(label="luxury jet interior")
    subject_a2 = SimpleNamespace(entity_id="private jet")
    candidate_a = _candidate_with_subjects("frame-a", [subject_a1, subject_a2], score=0.8)

    # Candidate B has no matching evidence but same base score
    subject_b = SimpleNamespace(label="unrelated")
    candidate_b = _candidate_with_subjects("frame-b", [subject_b], score=0.8)

    matches = AssetMatcher().match(
        concept=concept,
        candidates=(candidate_b, candidate_a),
    )

    assert len(matches) == 2
    assert matches[0].asset_id == "frame-a"
    assert matches[1].asset_id == "frame-b"
    assert matches[0].suitability_score > matches[1].suitability_score


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
