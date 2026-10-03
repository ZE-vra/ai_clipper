from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from src.thumbnails.domain.assets import (
    AssetProvenance,
    AssetProvenanceKind,
    FrameCandidate,
    VisualAsset,
)
from src.thumbnails.domain.brief import ThumbnailBrief
from src.thumbnails.domain.content import ContentUnderstanding
from src.thumbnails.domain.geometry import BoundingBox, Point, Region, Size
from src.thumbnails.domain.target import ThumbnailTarget
from src.thumbnails.intelligence.creative_director import RuleBasedCreativeDirector
from src.thumbnails.intelligence.asset_matcher import AssetMatcher
from src.thumbnails.intelligence.strategy_planner import StrategyPlanner
from src.thumbnails.layout.composition import CompositionPlanner
from src.thumbnails.layout.negotiation import LayoutNegotiator
from src.thumbnails.layout.typography import TypographyPlanner
from src.thumbnails.orchestration.thumbnail_orchestrator import ThumbnailOrchestrator


class FakeFrameDiscovery:
    def __init__(self, candidate: FrameCandidate) -> None:
        self.candidate = candidate

    def discover(
        self,
        *,
        video_path: Path,
        output_dir: Path,
    ) -> tuple[FrameCandidate, ...]:
        return (self.candidate,)


def _candidate(image_path: Path) -> FrameCandidate:
    perception = SimpleNamespace(
        frame_path=image_path,
        quality=SimpleNamespace(overall_quality=0.9),
        crop=SimpleNamespace(score=0.9),
        focal=SimpleNamespace(
            regions=(SimpleNamespace(strength=0.8),),
            primary_region=SimpleNamespace(strength=0.8),
        ),
        subjects=SimpleNamespace(
            subjects=(object(),),
            primary_subject=SimpleNamespace(
                prominence=0.9,
                focal_point=Point(x=0.72, y=0.5),
            ),
        ),
    )
    asset = VisualAsset(
        asset_id=f"frame-{image_path.stem}",
        provenance=AssetProvenance(
            kind=AssetProvenanceKind.SOURCE_FRAME,
            source_timestamps=(12.0,),
        ),
        path=str(image_path),
        source_timestamp=12.0,
    )
    return FrameCandidate(
        candidate_id=f"candidate-{image_path.stem}",
        timestamp=12.0,
        asset=asset,
        perception=perception,
    )


def _target() -> ThumbnailTarget:
    return ThumbnailTarget(
        target_id="youtube-test",
        platform="youtube",
        size=Size(width=1280, height=720),
        safe_regions=(
            Region(
                name="safe",
                bounds=BoundingBox(
                    left=0.02,
                    top=0.02,
                    right=0.98,
                    bottom=0.98,
                ),
            ),
        ),
    )


def test_v2_orchestrator_renders_real_first_vertical_slice(tmp_path: Path) -> None:
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"placeholder")

    frame = tmp_path / "frame.jpg"
    Image.new("RGB", (1920, 1080), (30, 80, 120)).save(frame)

    orchestrator = ThumbnailOrchestrator(
        frame_discovery=FakeFrameDiscovery(_candidate(frame)),
        creative_director=RuleBasedCreativeDirector(),
        asset_matcher=AssetMatcher(),
        strategy_planner=StrategyPlanner(),
        composition_planner=CompositionPlanner(),
        typography_planner=TypographyPlanner(),
        layout_negotiator=LayoutNegotiator(),
    )

    output = tmp_path / "thumbnail.jpg"

    result = orchestrator.generate(
        source_video_path=source_video,
        output_path=output,
        candidates_dir=tmp_path / "candidates",
        brief=ThumbnailBrief(
            core_hook="$2M CARPET ON A PLANE?!",
            subject="private jet",
            promise="A $2 million carpet is revealed.",
            curiosity_angle="Why is the carpet worth $2 million?",
            emotional_direction="surprise",
            important_objects=("carpet",),
            visual_evidence=("luxury jet interior",),
        ),
        understanding=ContentUnderstanding(
            entities=(),
            events=(),
            confidence=1.0,
        ),
        target=_target(),
    )

    assert result.output_path == str(output)
    assert result.selected_attempt_id
    assert output.is_file()

    with Image.open(output) as image:
        assert image.size == (1280, 720)


class MultiFrameDiscovery:
    def __init__(self, candidates: tuple[FrameCandidate, ...]) -> None:
        self.candidates = candidates

    def discover(
        self,
        *,
        video_path: Path,
        output_dir: Path,
    ) -> tuple[FrameCandidate, ...]:
        return self.candidates


def test_v2_orchestrator_skips_broken_candidate_and_uses_next_valid_one(
    tmp_path: Path,
) -> None:
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"placeholder")

    broken = _candidate(tmp_path / "missing.jpg")
    valid_frame = tmp_path / "valid.jpg"
    Image.new("RGB", (1920, 1080), (30, 80, 120)).save(valid_frame)
    valid = _candidate(valid_frame)

    orchestrator = ThumbnailOrchestrator(
        frame_discovery=MultiFrameDiscovery((broken, valid)),
        creative_director=RuleBasedCreativeDirector(),
        asset_matcher=AssetMatcher(),
        strategy_planner=StrategyPlanner(),
        composition_planner=CompositionPlanner(),
        typography_planner=TypographyPlanner(),
        layout_negotiator=LayoutNegotiator(),
        max_concepts=1,
        max_matches_per_concept=2,
    )

    output = tmp_path / "thumbnail.jpg"

    result = orchestrator.generate(
        source_video_path=source_video,
        output_path=output,
        candidates_dir=tmp_path / "candidates",
        brief=ThumbnailBrief(
            core_hook="$2M CARPET ON A PLANE?!",
            subject="private jet",
            promise="A $2 million carpet is revealed.",
            curiosity_angle="Why is the carpet worth $2 million?",
            emotional_direction="surprise",
            important_objects=("carpet",),
            visual_evidence=("luxury jet interior",),
        ),
        understanding=ContentUnderstanding(
            entities=(),
            events=(),
            confidence=1.0,
        ),
        target=_target(),
    )

    assert result.status.value == "success"
    assert result.output_path == str(output)
    assert output.is_file()
