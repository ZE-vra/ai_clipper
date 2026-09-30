from pathlib import Path

import pytest

from src.thumbnails.perception.local_subject_analyzer import (
    LocalSubjectAnalyzer,
)
from src.thumbnails.perception.subjects import SubjectKind


class FakeBox:
    def __init__(
        self,
        coordinates: list[float],
        confidence: float,
    ) -> None:
        self.xyxy = [FakeTensor(coordinates)]
        self.conf = [FakeTensor([confidence])]


class FakeTensor:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values

    def __getitem__(self, index: int) -> "FakeTensor":
        return self

    def __float__(self) -> float:
        return float(self._values[0])


class FakeBoxes:
    def __init__(self, boxes: list[FakeBox]) -> None:
        self._boxes = boxes

    def __len__(self) -> int:
        return len(self._boxes)

    def __iter__(self):
        return iter(self._boxes)


class FakeResult:
    def __init__(
        self,
        boxes: FakeBoxes,
        shape: tuple[int, int],
    ) -> None:
        self.boxes = boxes
        self.orig_shape = shape


class FakeModel:
    def __init__(self, results: list[FakeResult]) -> None:
        self.results = results
        self.calls: list[dict] = []

    def __call__(
        self,
        image_path: str,
        *,
        classes: list[int],
        verbose: bool,
    ) -> list[FakeResult]:
        self.calls.append(
            {
                "image_path": image_path,
                "classes": classes,
                "verbose": verbose,
            }
        )

        return self.results


def build_analyzer(
    results: list[FakeResult],
) -> tuple[LocalSubjectAnalyzer, FakeModel]:
    analyzer = LocalSubjectAnalyzer.__new__(
        LocalSubjectAnalyzer
    )

    fake_model = FakeModel(results)
    analyzer._model = fake_model

    return analyzer, fake_model


def test_analyze_converts_person_detection_to_subject_evidence(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.touch()

    result = FakeResult(
        boxes=FakeBoxes(
            [
                FakeBox(
                    coordinates=[100.0, 200.0, 500.0, 800.0],
                    confidence=0.90,
                )
            ]
        ),
        shape=(1000, 2000),
    )

    analyzer, fake_model = build_analyzer([result])

    analysis = analyzer.analyze(image_path)

    assert len(analysis.subjects) == 1

    subject = analysis.subjects[0]

    assert subject.subject_id == "person_01"
    assert subject.kind is SubjectKind.PERSON
    assert subject.confidence == pytest.approx(0.90)

    assert subject.bounds.left == pytest.approx(0.05)
    assert subject.bounds.top == pytest.approx(0.20)
    assert subject.bounds.width == pytest.approx(0.20)
    assert subject.bounds.height == pytest.approx(0.60)

    assert subject.focal_point.x == pytest.approx(0.15)
    assert subject.focal_point.y == pytest.approx(0.50)

    assert 0.0 <= subject.prominence <= 1.0
    assert subject.occluded is False

    assert fake_model.calls == [
        {
            "image_path": str(image_path),
            "classes": [0],
            "verbose": False,
        }
    ]


def test_analyze_marks_edge_touching_detection_as_occluded(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.touch()

    result = FakeResult(
        boxes=FakeBoxes(
            [
                FakeBox(
                    coordinates=[0.0, 100.0, 400.0, 800.0],
                    confidence=0.80,
                )
            ]
        ),
        shape=(1000, 2000),
    )

    analyzer, _ = build_analyzer([result])

    analysis = analyzer.analyze(image_path)

    assert analysis.subjects[0].occluded is True


def test_analyze_returns_empty_evidence_when_no_boxes(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.touch()

    result = FakeResult(
        boxes=FakeBoxes([]),
        shape=(1000, 2000),
    )

    analyzer, _ = build_analyzer([result])

    analysis = analyzer.analyze(image_path)

    assert analysis.subjects == ()
    assert analysis.analysis_version == "1"


def test_analyze_rejects_missing_image() -> None:
    analyzer, _ = build_analyzer([])

    with pytest.raises(FileNotFoundError):
        analyzer.analyze(
            Path("does_not_exist.jpg")
        )