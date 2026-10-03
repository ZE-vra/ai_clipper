from pathlib import Path
from types import SimpleNamespace

from src.thumbnails.domain.assets import AssetProvenance, AssetProvenanceKind, FrameCandidate, VisualAsset
from src.thumbnails.domain.concepts import (
    CopyBlock,
    CopyConcept,
    CopyRole,
    ThumbnailConcept,
    VisualStrategy,
)
from src.thumbnails.intelligence.asset_matcher import AssetMatcher


def _candidate(asset_id: str, timestamp: float, score: float = 0.8) -> FrameCandidate:
    perception = SimpleNamespace(
        quality=SimpleNamespace(overall_quality=score),
        crop=SimpleNamespace(score=score),
        focal=SimpleNamespace(regions=(SimpleNamespace(strength=score),)),
        subjects=SimpleNamespace(subjects=(SimpleNamespace(kind="person"),)),
    )
    return FrameCandidate(
        candidate_id=f"candidate-{asset_id}",
        timestamp=timestamp,
        asset=VisualAsset(
            asset_id=asset_id,
            provenance=AssetProvenance(kind=AssetProvenanceKind.SOURCE_FRAME),
            path=str(Path(f"{asset_id}.jpg")),
        ),
        perception=perception,
    )


def _copy() -> CopyConcept:
    return CopyConcept(
        concept_id="event-reveal-copy",
        blocks=(CopyBlock(text="What happens next?", role=CopyRole.HOOK),),
    )


def _event_concept() -> ThumbnailConcept:
    return ThumbnailConcept(
        concept_id="event-reveal",
        title="The prize is revealed",
        visual_idea="Show the moment the prize appears",
        copy=_copy(),
        curiosity_mechanism="reveal",
        emotional_direction="surprise",
        candidate_strategies=(VisualStrategy.SOURCE_FRAME,),
        preferred_time_range=(10.0, 15.0),
    )


def test_asset_matcher_prefers_frame_inside_the_title_relevant_event() -> None:
    unrelated_but_sharp = _candidate("sharp", timestamp=3.0, score=1.0)
    event_frame = _candidate("event", timestamp=12.0, score=0.7)

    matches = AssetMatcher().match(
        concept=_event_concept(),
        candidates=(unrelated_but_sharp, event_frame),
    )

    assert matches[0].asset_id == "event"
    assert "event_time_match=1.00" in matches[0].reasons


def test_asset_matcher_allows_nearby_reaction_frames_to_compete() -> None:
    inside = _candidate("inside", timestamp=12.0, score=0.7)
    nearby = _candidate("nearby", timestamp=16.0, score=0.9)

    matches = AssetMatcher().match(
        concept=_event_concept(),
        candidates=(nearby, inside),
    )

    assert matches[0].asset_id == "inside"
    assert any(reason.startswith("event_time_match=") for match in matches for reason in match.reasons)


def test_asset_matcher_keeps_quality_order_when_no_event_window_exists() -> None:
    lower_quality = _candidate("lower", timestamp=1.0, score=0.4)
    higher_quality = _candidate("higher", timestamp=30.0, score=0.9)
    concept = ThumbnailConcept(
        concept_id="general",
        title="The prize is revealed",
        visual_idea="Show a relevant moment",
        copy=_copy(),
        curiosity_mechanism="reveal",
        emotional_direction="surprise",
        candidate_strategies=(VisualStrategy.SOURCE_FRAME,),
    )

    matches = AssetMatcher().match(concept=concept, candidates=(lower_quality, higher_quality))
    assert matches[0].asset_id == "higher"
