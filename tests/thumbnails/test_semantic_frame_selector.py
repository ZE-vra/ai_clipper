from pathlib import Path
from types import SimpleNamespace

from src.thumbnails.domain.geometry import BoundingBox, Point
from src.thumbnails.perception.semantic_frame_selector import SemanticFrameSelector
from src.thumbnails.perception.visual_intelligence import (
    SemanticSubject,
    VisualIntelligenceResult,
)


def _semantic(frame: Path, bounds: BoundingBox) -> VisualIntelligenceResult:
    return VisualIntelligenceResult(
        frame_path=frame,
        subjects=(
            SemanticSubject(
                subject_id="person_01",
                kind="person",
                confidence=0.95,
                bounds=bounds,
                mask_path=frame,
                area_ratio=0.25,
                importance=0.9,
            ),
        ),
    )


def _perception(frame: Path, overall: float):
    return SimpleNamespace(
        frame_path=frame,
        quality=SimpleNamespace(overall_quality=overall),
        subjects=SimpleNamespace(
            primary_subject=SimpleNamespace(prominence=overall)
        ),
        focal=SimpleNamespace(
            primary_region=SimpleNamespace(strength=overall)
        ),
        crop=SimpleNamespace(score=overall),
    )


def test_semantic_selector_prefers_frame_with_head_clearance() -> None:
    selector = SemanticFrameSelector(semantic_weight=0.20)

    bad = Path("bad.jpg")
    good = Path("good.jpg")

    candidates = [
        _perception(bad, 0.90),
        _perception(good, 0.86),
    ]
    semantic = {
        bad: _semantic(
            bad,
            BoundingBox(0.30, 0.0, 0.70, 0.70),
        ),
        good: _semantic(
            good,
            BoundingBox(0.30, 0.10, 0.70, 0.80),
        ),
    }

    selected = selector.select(candidates, semantic)

    assert selected.perception.frame_path == good


def test_semantic_selector_prefers_real_side_negative_space() -> None:
    selector = SemanticFrameSelector(semantic_weight=0.35)

    centered = Path("centered.jpg")
    staged = Path("staged.jpg")

    candidates = [
        _perception(centered, 0.92),
        _perception(staged, 0.88),
    ]
    semantic = {
        centered: _semantic(
            centered,
            BoundingBox(0.18, 0.10, 0.82, 0.90),
        ),
        staged: _semantic(
            staged,
            BoundingBox(0.52, 0.10, 0.84, 0.90),
        ),
    }

    selected = selector.select(candidates, semantic)

    assert selected.perception.frame_path == staged


def test_semantic_selector_preserves_base_selection_without_semantic_evidence() -> None:
    selector = SemanticFrameSelector(semantic_weight=0.20)

    first = Path("first.jpg")
    second = Path("second.jpg")

    selected = selector.select(
        [_perception(first, 0.80), _perception(second, 0.90)],
        {},
    )

    assert selected.perception.frame_path == second
