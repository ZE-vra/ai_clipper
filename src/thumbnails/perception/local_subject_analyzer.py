from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from src.thumbnails.domain.geometry import BoundingBox, Point
from src.thumbnails.perception.subjects import (
    SubjectAnalysisEvidence,
    SubjectEvidence,
    SubjectKind,
)


class LocalSubjectAnalyzer:
    """
    Local semantic subject analyzer backed by YOLO.

    V1 detects people only.

    This class adapts model output into domain perception evidence.
    It does not make thumbnail-selection or composition decisions.
    """

    def __init__(
        self,
        model_path: str | Path = "yolo26n.pt",
    ) -> None:
        self._model = YOLO(str(model_path))

    def analyze(
        self,
        image_path: Path,
    ) -> SubjectAnalysisEvidence:
        """
        Detect people in one image and return normalized subject evidence.
        """
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image file does not exist: {image_path}"
            )

        results = self._model(
            str(image_path),
            classes=[0],
            verbose=False,
        )

        if not results:
            return SubjectAnalysisEvidence(
                subjects=(),
                analysis_version="1",
            )

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return SubjectAnalysisEvidence(
                subjects=(),
                analysis_version="1",
            )

        image_height, image_width = result.orig_shape

        subjects: list[SubjectEvidence] = []

        for index, box in enumerate(result.boxes):
            x1, y1, x2, y2 = (
                float(value)
                for value in box.xyxy[0].tolist()
            )

            confidence = float(box.conf[0])

            bounds = BoundingBox(
                left=x1 / image_width,
                top=y1 / image_height,
                right=x2 / image_width,
                bottom=y2 / image_height,
            )

            focal_point = Point(
                x=(x1 + x2) / 2 / image_width,
                y=(y1 + y2) / 2 / image_height,
            )

            prominence = self._calculate_prominence(
                bounds=bounds,
                confidence=confidence,
            )

            subjects.append(
                SubjectEvidence(
                    subject_id=f"person_{index + 1:02d}",
                    kind=SubjectKind.PERSON,
                    confidence=confidence,
                    bounds=bounds,
                    focal_point=focal_point,
                    prominence=prominence,
                    occluded=self._is_edge_occluded(bounds),
                )
            )

        return SubjectAnalysisEvidence(
            subjects=tuple(subjects),
            analysis_version="1",
        )

    @staticmethod
    def _calculate_prominence(
        *,
        bounds: BoundingBox,
        confidence: float,
    ) -> float:
        """
        Calculate deterministic visual prominence for a detected person.

        Prominence combines model confidence with the person's
        relative frame area.

        This is evidence about the detection, not a thumbnail
        selection decision.
        """
        area_ratio = bounds.width * bounds.height

        area_score = min(
            1.0,
            area_ratio / 0.50,
        )

        prominence = (
            0.70 * confidence
            + 0.30 * area_score
        )

        return max(0.0, min(1.0, prominence))

    @staticmethod
    def _is_edge_occluded(
        bounds: BoundingBox,
    ) -> bool:
        """
        Mark detections touching a frame edge as potentially occluded.
        """
        edge_tolerance = 0.01

        return (
            bounds.left <= edge_tolerance
            or bounds.top <= edge_tolerance
            or bounds.right >= 1.0 - edge_tolerance
            or bounds.bottom >= 1.0 - edge_tolerance
        )