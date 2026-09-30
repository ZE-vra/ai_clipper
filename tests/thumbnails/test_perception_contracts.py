from __future__ import annotations

import pytest

from src.thumbnails.domain.assets import (
    AssetProvenance,
    FrameCandidate,
    VisualAsset,
)
from src.thumbnails.domain.geometry import (
    BoundingBox,
    Point,
)
from src.thumbnails.perception.candidates import (
    FrameVisualEvidence,
)
from src.thumbnails.perception.quality import (
    FrameQualityEvidence,
)
from src.thumbnails.perception.subjects import (
    SubjectEvidence,
    SubjectKind,
)


def make_candidate() -> FrameCandidate:
    asset = VisualAsset(
        asset_id="frame-01",
        provenance=AssetProvenance.SOURCE_FRAME,
        path="frame.jpg",
        source_timestamp=30.0,
    )

    return FrameCandidate(
        candidate_id="candidate-01",
        timestamp=30.0,
        asset=asset,
    )


def make_quality() -> FrameQualityEvidence:
    return FrameQualityEvidence(
        sharpness=0.85,
        brightness=0.60,
        contrast=0.72,
        saturation=0.58,
        motion_blur=0.12,
        noise=0.10,
        overall_quality=0.78,
    )


def make_subject() -> SubjectEvidence:
    return SubjectEvidence(
        subject_id="subject-01",
        kind=SubjectKind.PERSON,
        confidence=0.94,
        bounds=BoundingBox(
            left=0.55,
            top=0.18,
            right=0.88,
            bottom=0.82,
        ),
        focal_point=Point(0.715, 0.50),
        prominence=0.86,
    )


class TestFrameQualityEvidence:
    def test_accepts_normalized_quality_signals(self):
        evidence = make_quality()

        assert evidence.sharpness == 0.85
        assert evidence.motion_blur == 0.12
        assert evidence.overall_quality == 0.78

    @pytest.mark.parametrize(
        "field",
        [
            "sharpness",
            "brightness",
            "contrast",
            "saturation",
            "motion_blur",
            "noise",
            "overall_quality",
        ],
    )
    def test_rejects_scores_below_zero(self, field):
        values = {
            "sharpness": 0.85,
            "brightness": 0.60,
            "contrast": 0.72,
            "saturation": 0.58,
            "motion_blur": 0.12,
            "noise": 0.10,
            "overall_quality": 0.78,
        }

        values[field] = -0.01

        with pytest.raises(ValueError):
            FrameQualityEvidence(**values)

    @pytest.mark.parametrize(
        "field",
        [
            "sharpness",
            "brightness",
            "contrast",
            "saturation",
            "motion_blur",
            "noise",
            "overall_quality",
        ],
    )
    def test_rejects_scores_above_one(self, field):
        values = {
            "sharpness": 0.85,
            "brightness": 0.60,
            "contrast": 0.72,
            "saturation": 0.58,
            "motion_blur": 0.12,
            "noise": 0.10,
            "overall_quality": 0.78,
        }

        values[field] = 1.01

        with pytest.raises(ValueError):
            FrameQualityEvidence(**values)


class TestSubjectEvidence:
    def test_preserves_subject_information(self):
        subject = make_subject()

        assert subject.subject_id == "subject-01"
        assert subject.kind is SubjectKind.PERSON
        assert subject.confidence == pytest.approx(0.94)
        assert subject.prominence == pytest.approx(0.86)
        assert subject.occluded is False

    def test_rejects_blank_subject_id(self):
        with pytest.raises(ValueError):
            SubjectEvidence(
                subject_id="",
                kind=SubjectKind.PERSON,
                confidence=0.9,
                bounds=BoundingBox(
                    0.2,
                    0.2,
                    0.8,
                    0.8,
                ),
                focal_point=Point(0.5, 0.5),
                prominence=0.8,
            )

    @pytest.mark.parametrize(
        "confidence",
        [-0.01, 1.01],
    )
    def test_rejects_invalid_confidence(self, confidence):
        with pytest.raises(ValueError):
            SubjectEvidence(
                subject_id="subject-01",
                kind=SubjectKind.PERSON,
                confidence=confidence,
                bounds=BoundingBox(
                    0.2,
                    0.2,
                    0.8,
                    0.8,
                ),
                focal_point=Point(0.5, 0.5),
                prominence=0.8,
            )

    @pytest.mark.parametrize(
        "prominence",
        [-0.01, 1.01],
    )
    def test_rejects_invalid_prominence(self, prominence):
        with pytest.raises(ValueError):
            SubjectEvidence(
                subject_id="subject-01",
                kind=SubjectKind.PERSON,
                confidence=0.9,
                bounds=BoundingBox(
                    0.2,
                    0.2,
                    0.8,
                    0.8,
                ),
                focal_point=Point(0.5, 0.5),
                prominence=prominence,
            )


class TestFrameVisualEvidence:
    def test_combines_candidate_quality_and_subjects(self):
        subject = make_subject()

        evidence = FrameVisualEvidence(
            candidate=make_candidate(),
            quality=make_quality(),
            subjects=(subject,),
            focal_regions=(
                BoundingBox(
                    0.50,
                    0.10,
                    0.92,
                    0.88,
                ),
            ),
        )

        assert evidence.candidate.candidate_id == "candidate-01"
        assert evidence.quality.overall_quality == 0.78
        assert len(evidence.subjects) == 1
        assert len(evidence.focal_regions) == 1

    def test_primary_focal_point_comes_from_most_prominent_subject(self):
        subject_a = make_subject()

        subject_b = SubjectEvidence(
            subject_id="subject-02",
            kind=SubjectKind.OBJECT,
            confidence=0.88,
            bounds=BoundingBox(
                0.05,
                0.20,
                0.40,
                0.70,
            ),
            focal_point=Point(0.225, 0.45),
            prominence=0.55,
        )

        evidence = FrameVisualEvidence(
            candidate=make_candidate(),
            quality=make_quality(),
            subjects=(subject_a, subject_b),
        )

        assert evidence.primary_focal_point == subject_a.focal_point

    def test_primary_focal_point_is_none_without_subjects(self):
        evidence = FrameVisualEvidence(
            candidate=make_candidate(),
            quality=make_quality(),
        )

        assert evidence.primary_focal_point is None

    def test_analysis_version_is_persisted(self):
        evidence = FrameVisualEvidence(
            candidate=make_candidate(),
            quality=make_quality(),
            analysis_version="2026-09-v1",
        )

        assert evidence.analysis_version == "2026-09-v1"

    def test_analysis_version_must_not_be_blank(self):
        with pytest.raises(ValueError):
            FrameVisualEvidence(
                candidate=make_candidate(),
                quality=make_quality(),
                analysis_version="",
            )
