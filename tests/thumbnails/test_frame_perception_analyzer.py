from pathlib import Path

import pytest

from src.thumbnails.domain.geometry import BoundingBox, Point
from src.thumbnails.perception.focal import (
    FocalAnalysisEvidence,
    FocalRegionEvidence,
)
from src.thumbnails.perception.frame_perception_analyzer import (
    FramePerceptionAnalyzer,
)
from src.thumbnails.perception.quality import (
    FrameQualityEvidence,
)
from src.thumbnails.perception.subjects import (
    SubjectAnalysisEvidence,
)


class FakeQualityAnalyzer:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def analyze(
        self,
        image_path: str | Path,
    ) -> FrameQualityEvidence:
        path = Path(image_path)
        self.calls.append(path)

        return FrameQualityEvidence(
            sharpness=0.8,
            brightness=0.5,
            contrast=0.7,
            saturation=0.6,
            motion_blur=0.1,
            noise=0.1,
            overall_quality=0.75,
        )


class FakeFocalAnalyzer:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def analyze(
        self,
        image_path: str | Path,
    ) -> FocalAnalysisEvidence:
        path = Path(image_path)
        self.calls.append(path)

        return FocalAnalysisEvidence(
            regions=(
                FocalRegionEvidence(
                    region_id="focal_01",
                    bounds=BoundingBox(
                        left=0.2,
                        top=0.2,
                        right=0.6,
                        bottom=0.8,
                    ),
                    focal_point=Point(
                        x=0.4,
                        y=0.5,
                    ),
                    strength=0.8,
                    area_ratio=0.24,
                ),
            ),
        )


class FakeSubjectAnalyzer:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def analyze(
        self,
        image_path: Path,
    ) -> SubjectAnalysisEvidence:
        self.calls.append(image_path)

        return SubjectAnalysisEvidence(
            subjects=(),
        )


def test_analyzer_combines_all_perception_evidence(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.touch()

    quality = FakeQualityAnalyzer()
    focal = FakeFocalAnalyzer()
    subjects = FakeSubjectAnalyzer()

    analyzer = FramePerceptionAnalyzer(
        quality_analyzer=quality,
        focal_analyzer=focal,
        subject_analyzer=subjects,
    )

    result = analyzer.analyze(image_path)

    assert result.frame_path == image_path

    assert result.quality.overall_quality == pytest.approx(
        0.75
    )

    assert result.focal.primary_region is not None

    assert result.subjects.subjects == ()

    assert quality.calls == [image_path]
    assert focal.calls == [image_path]
    assert subjects.calls == [image_path]


def test_analyzer_requires_subject_analyzer(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.touch()

    analyzer = FramePerceptionAnalyzer(
        quality_analyzer=FakeQualityAnalyzer(),
        focal_analyzer=FakeFocalAnalyzer(),
    )

    with pytest.raises(RuntimeError):
        analyzer.analyze(image_path)