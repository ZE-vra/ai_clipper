from __future__ import annotations

import pytest

from src.thumbnails.domain.geometry import BoundingBox, Point
from src.thumbnails.perception.subjects import (
    SubjectAnalysisEvidence,
    SubjectEvidence,
    SubjectKind,
)


def make_subject(
    subject_id: str = "subject_01",
    kind: SubjectKind = SubjectKind.PERSON,
    prominence: float = 0.8,
) -> SubjectEvidence:
    return SubjectEvidence(
        subject_id=subject_id,
        kind=kind,
        confidence=0.95,
        bounds=BoundingBox(
            left=0.2,
            top=0.1,
            right=0.6,
            bottom=0.8,
        ),
        focal_point=Point(
            x=0.4,
            y=0.45,
        ),
        prominence=prominence,
    )


def test_subject_analysis_accepts_empty_subjects() -> None:
    evidence = SubjectAnalysisEvidence()

    assert evidence.subjects == ()
    assert evidence.primary_subject is None
    assert evidence.people == ()
    assert evidence.faces == ()


def test_subject_analysis_preserves_subjects() -> None:
    person = make_subject()

    evidence = SubjectAnalysisEvidence(
        subjects=(person,),
    )

    assert evidence.subjects == (person,)


def test_people_returns_only_person_subjects() -> None:
    person = make_subject(
        subject_id="person_01",
        kind=SubjectKind.PERSON,
    )
    face = make_subject(
        subject_id="face_01",
        kind=SubjectKind.FACE,
    )

    evidence = SubjectAnalysisEvidence(
        subjects=(person, face),
    )

    assert evidence.people == (person,)


def test_faces_returns_only_face_subjects() -> None:
    person = make_subject(
        subject_id="person_01",
        kind=SubjectKind.PERSON,
    )
    face = make_subject(
        subject_id="face_01",
        kind=SubjectKind.FACE,
    )

    evidence = SubjectAnalysisEvidence(
        subjects=(person, face),
    )

    assert evidence.faces == (face,)


def test_primary_subject_returns_most_prominent_subject() -> None:
    weak = make_subject(
        subject_id="weak",
        prominence=0.4,
    )
    strong = make_subject(
        subject_id="strong",
        prominence=0.9,
    )

    evidence = SubjectAnalysisEvidence(
        subjects=(weak, strong),
    )

    assert evidence.primary_subject == strong


def test_subject_ids_must_be_unique() -> None:
    first = make_subject(subject_id="duplicate")
    second = make_subject(subject_id="duplicate")

    with pytest.raises(
        ValueError,
        match="subject_ids must be unique",
    ):
        SubjectAnalysisEvidence(
            subjects=(first, second),
        )


def test_analysis_version_must_not_be_blank() -> None:
    with pytest.raises(
        ValueError,
        match="analysis_version must not be blank",
    ):
        SubjectAnalysisEvidence(
            analysis_version="   ",
        )


def test_subject_rejects_blank_id() -> None:
    with pytest.raises(
        ValueError,
        match="subject_id must not be blank",
    ):
        make_subject(subject_id="   ")


def test_subject_rejects_invalid_confidence() -> None:
    with pytest.raises(
        ValueError,
        match="confidence must be between 0 and 1",
    ):
        SubjectEvidence(
            subject_id="subject_01",
            kind=SubjectKind.PERSON,
            confidence=1.1,
            bounds=BoundingBox(
                left=0.2,
                top=0.1,
                right=0.6,
                bottom=0.8,
            ),
            focal_point=Point(
                x=0.4,
                y=0.45,
            ),
            prominence=0.8,
        )


def test_subject_rejects_invalid_prominence() -> None:
    with pytest.raises(
        ValueError,
        match="prominence must be between 0 and 1",
    ):
        SubjectEvidence(
            subject_id="subject_01",
            kind=SubjectKind.PERSON,
            confidence=0.95,
            bounds=BoundingBox(
                left=0.2,
                top=0.1,
                right=0.6,
                bottom=0.8,
            ),
            focal_point=Point(
                x=0.4,
                y=0.45,
            ),
            prominence=1.1,
        )