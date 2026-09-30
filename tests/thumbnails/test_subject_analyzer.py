from __future__ import annotations

from pathlib import Path

from src.thumbnails.domain.geometry import BoundingBox, Point
from src.thumbnails.perception.subject_analyzer import SubjectAnalyzer
from src.thumbnails.perception.subjects import (
    SubjectAnalysisEvidence,
    SubjectEvidence,
    SubjectKind,
)


class FakeSubjectAnalyzer:
    """Minimal implementation used to verify the analyzer contract."""

    def analyze(
        self,
        image_path: Path,
    ) -> SubjectAnalysisEvidence:
        return SubjectAnalysisEvidence(
            subjects=(
                SubjectEvidence(
                    subject_id="person_01",
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
                    prominence=0.9,
                ),
            ),
        )


def test_subject_analyzer_contract_accepts_implementation() -> None:
    analyzer: SubjectAnalyzer = FakeSubjectAnalyzer()

    result = analyzer.analyze(
        Path("frame.jpg")
    )

    assert isinstance(
        result,
        SubjectAnalysisEvidence,
    )


def test_subject_analyzer_returns_detected_subjects() -> None:
    analyzer: SubjectAnalyzer = FakeSubjectAnalyzer()

    result = analyzer.analyze(
        Path("frame.jpg")
    )

    assert len(result.subjects) == 1

    subject = result.subjects[0]

    assert subject.subject_id == "person_01"
    assert subject.kind is SubjectKind.PERSON
    assert subject.confidence == 0.95
    assert subject.prominence == 0.9