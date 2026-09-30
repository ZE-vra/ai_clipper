from __future__ import annotations

import pytest

from src.thumbnails.domain.assets import AssetProvenance, VisualAsset
from src.thumbnails.domain.brief import CreativeBrief
from src.thumbnails.domain.concepts import CopyBlock, CopyConcept, CopyRole, ThumbnailConcept, VisualStrategy
from src.thumbnails.domain.geometry import BoundingBox, Point, Size
from src.thumbnails.domain.target import ThumbnailTarget, UIOcclusionRegion
from src.thumbnails.layout.composition import CompositionPlanner, CompositionPlannerConfig
from src.thumbnails.perception.crop import CropSuitabilityEvidence
from src.thumbnails.perception.focal import FocalAnalysisEvidence
from src.thumbnails.perception.frame_perception import FramePerception
from src.thumbnails.perception.quality import FrameQualityEvidence
from src.thumbnails.perception.scoring import FrameCandidateScore
from src.thumbnails.perception.subjects import SubjectAnalysisEvidence, SubjectEvidence, SubjectKind
from src.thumbnails.perception.frame_selector import SelectedFrame


def _selected_frame(subject: SubjectEvidence | None) -> SelectedFrame:
    perception = FramePerception(
        frame_path="frame.jpg",
        quality=FrameQualityEvidence(
            sharpness=1.0,
            brightness=0.8,
            contrast=0.8,
            saturation=0.7,
            motion_blur=0.0,
            noise=0.1,
            overall_quality=0.9,
        ),
        focal=FocalAnalysisEvidence(regions=()),
        subjects=SubjectAnalysisEvidence(
            subjects=() if subject is None else (subject,)
        ),
        crop=CropSuitabilityEvidence(
            score=1.0,
            retained_subject_ratio=1.0,
            primary_subject_retention=1.0,
            focal_retention=1.0,
            subject_count=0 if subject is None else 1,
        ),
    )
    return SelectedFrame(
        perception=perception,
        score=FrameCandidateScore(
            quality_score=0.9,
            subject_score=0.8,
            focal_score=0.0,
            crop_score=1.0,
            overall_score=0.9,
        ),
    )


def _target(*, occlusions=()) -> ThumbnailTarget:
    return ThumbnailTarget(
        target_id="youtube-short-thumbnail",
        platform="youtube",
        size=Size(width=1080, height=1920),
        ui_occlusion_regions=occlusions,
    )


def _asset() -> VisualAsset:
    return VisualAsset(
        asset_id="frame-15",
        provenance=AssetProvenance.SOURCE_FRAME,
        path="frame.jpg",
        source_timestamp=15.0,
    )


def test_planner_centers_vertical_crop_on_primary_subject() -> None:
    subject = SubjectEvidence(
        subject_id="person_01",
        kind=SubjectKind.PERSON,
        confidence=0.95,
        bounds=BoundingBox(0.65, 0.10, 0.90, 0.90),
        focal_point=Point(0.775, 0.50),
        prominence=0.90,
    )

    plan = CompositionPlanner().plan(
        selected_frame=_selected_frame(subject),
        asset=_asset(),
        target=_target(),
        source_aspect_ratio=16 / 9,
    )

    placement = plan.visual_placements[0]
    assert placement.asset_id == "frame-15"
    assert placement.focal_point == Point(0.775, 0.50)
    assert placement.crop_bounds is not None
    assert placement.crop_bounds.left == pytest.approx(0.616796875)
    assert placement.crop_bounds.right == pytest.approx(0.933203125)


def test_planner_places_negative_space_opposite_subject() -> None:
    subject = SubjectEvidence(
        subject_id="person_01",
        kind=SubjectKind.PERSON,
        confidence=0.95,
        bounds=BoundingBox(0.65, 0.10, 0.90, 0.90),
        focal_point=Point(0.775, 0.50),
        prominence=0.90,
    )

    plan = CompositionPlanner().plan(
        selected_frame=_selected_frame(subject),
        asset=_asset(),
        target=_target(),
        source_aspect_ratio=16 / 9,
    )

    assert plan.negative_space_regions[0] == BoundingBox(
        0.0, 0.10, 0.42, 0.90
    )


def test_planner_uses_alternate_side_when_preferred_side_is_ui_occluded() -> None:
    occlusion = UIOcclusionRegion(
        region=__import__("src.thumbnails.domain.geometry", fromlist=["Region"]).Region(
            name="right_ui",
            bounds=BoundingBox(0.55, 0.0, 1.0, 1.0),
        ),
        weight=1.0,
        reason="platform controls",
    )
    subject = SubjectEvidence(
        subject_id="person_01",
        kind=SubjectKind.PERSON,
        confidence=0.95,
        bounds=BoundingBox(0.05, 0.10, 0.30, 0.90),
        focal_point=Point(0.20, 0.50),
        prominence=0.90,
    )

    plan = CompositionPlanner().plan(
        selected_frame=_selected_frame(subject),
        asset=_asset(),
        target=_target(occlusions=(occlusion,)),
        source_aspect_ratio=16 / 9,
    )

    assert plan.negative_space_regions[0] == BoundingBox(
        0.58, 0.10, 1.0, 0.90
    )


def test_planner_handles_frames_without_detected_subjects() -> None:
    plan = CompositionPlanner().plan(
        selected_frame=_selected_frame(None),
        asset=_asset(),
        target=_target(),
        source_aspect_ratio=16 / 9,
    )

    placement = plan.visual_placements[0]
    assert placement.focal_point == Point(0.5, 0.5)
    assert placement.crop_bounds is not None
    assert plan.negative_space_regions


def test_planner_keeps_full_frame_when_source_is_already_vertical() -> None:
    plan = CompositionPlanner().plan(
        selected_frame=_selected_frame(None),
        asset=_asset(),
        target=_target(),
        source_aspect_ratio=9 / 16,
    )

    assert plan.visual_placements[0].crop_bounds == BoundingBox(
        0.0, 0.0, 1.0, 1.0
    )


def test_config_rejects_invalid_negative_space_range() -> None:
    with pytest.raises(ValueError):
        CompositionPlannerConfig(
            negative_space_top=0.9,
            negative_space_bottom=0.2,
        )
