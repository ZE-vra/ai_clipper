from types import SimpleNamespace

from src.thumbnails.domain.assets import AssetProvenance, VisualAsset
from src.thumbnails.domain.geometry import BoundingBox, Point, Region, Size
from src.thumbnails.domain.target import ThumbnailTarget
from src.thumbnails.layout.v11_composition import V11CompositionPlanner
from src.thumbnails.perception.subjects import SubjectEvidence, SubjectKind


def _target() -> ThumbnailTarget:
    return ThumbnailTarget(
        target_id="test", platform="test", size=Size(width=1080, height=1920),
        safe_regions=(Region(name="safe", bounds=BoundingBox(0.0, 0.0, 1.0, 1.0)),),
        ui_occlusion_regions=(),
    )


def _subject(bounds: BoundingBox, focal: Point) -> SubjectEvidence:
    return SubjectEvidence(
        subject_id="person-1", kind=SubjectKind.PERSON, confidence=0.95,
        bounds=bounds, focal_point=focal, prominence=0.9, occluded=False,
    )


def _selected(subject: SubjectEvidence):
    return SimpleNamespace(perception=SimpleNamespace(subjects=SimpleNamespace(primary_subject=subject)))


def _asset() -> VisualAsset:
    return VisualAsset(asset_id="frame", provenance=AssetProvenance.SOURCE_FRAME, path="frame.jpg")


def test_crop_never_slices_subject_when_subject_near_top_and_bottom() -> None:
    subject = _subject(BoundingBox(0.35, 0.04, 0.65, 0.96), Point(0.5, 0.50))
    crop = V11CompositionPlanner()._crop_window(
        source_aspect_ratio=16 / 9, target_aspect_ratio=9 / 16,
        subject=subject.bounds, center_x=subject.focal_point.x,
    )
    assert crop.top == 0.0
    assert crop.bottom == 1.0


def test_crop_contains_subject_with_clearance_when_zoom_is_possible() -> None:
    subject = _subject(BoundingBox(0.30, 0.10, 0.60, 0.70), Point(0.45, 0.40))
    crop = V11CompositionPlanner()._crop_window(
        source_aspect_ratio=16 / 9, target_aspect_ratio=9 / 16,
        subject=subject.bounds, center_x=subject.focal_point.x,
    )
    assert crop.top <= subject.bounds.top - 0.04
    assert crop.bottom >= subject.bounds.bottom + 0.04


def test_text_band_avoids_primary_subject_when_alternate_band_is_available() -> None:
    subject = _subject(BoundingBox(0.30, 0.02, 0.70, 0.45), Point(0.5, 0.22))
    composition = V11CompositionPlanner().plan(
        selected_frame=_selected(subject), asset=_asset(), target=_target(), source_aspect_ratio=16 / 9,
    )
    region = composition.negative_space_regions[0]
    assert region.top > subject.bounds.bottom


def test_subject_can_be_staged_opposite_the_text() -> None:
    subject = _subject(
        BoundingBox(0.30, 0.12, 0.62, 0.78),
        Point(0.46, 0.45),
    )
    composition = V11CompositionPlanner(
        __import__(
            "src.thumbnails.layout.v11_composition",
            fromlist=["V11CompositionConfig"],
        ).V11CompositionConfig(
            preferred_text_side="top",
            preferred_subject_side="right",
        )
    ).plan(
        selected_frame=_selected(subject),
        asset=_asset(),
        target=_target(),
        source_aspect_ratio=16 / 9,
    )

    placement = composition.visual_placements[0]
    region = composition.negative_space_regions[0]

    transformed_focal = placement.focal_point
    assert transformed_focal.x > 0.55
    assert region.right < transformed_focal.x
    assert region.bottom < 0.45


def test_centered_subject_remains_the_fallback_layout() -> None:
    subject = _subject(
        BoundingBox(0.30, 0.10, 0.60, 0.70),
        Point(0.45, 0.40),
    )
    composition = V11CompositionPlanner().plan(
        selected_frame=_selected(subject),
        asset=_asset(),
        target=_target(),
        source_aspect_ratio=16 / 9,
    )

    assert 0.35 <= composition.visual_placements[0].focal_point.x <= 0.55
