from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from ultralytics import YOLO

from src.thumbnails.domain.geometry import BoundingBox
from src.thumbnails.perception.visual_intelligence import (
    SemanticSubject,
    VisualIntelligenceResult,
)


class LocalVisualIntelligenceAnalyzer:
    """
    Local semantic visual intelligence backed by YOLO26 instance segmentation.

    V1.2 deliberately starts with person instance segmentation. The analyzer
    produces a pixel mask and semantic subject evidence; it does not decide
    copy, layout, or creative strategy.
    """

    def __init__(
        self,
        model_path: str | Path = "yolo26n-seg.pt",
        *,
        confidence_threshold: float = 0.35,
        model: Any | None = None,
    ) -> None:
        if not 0.0 < confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")

        self._model = model if model is not None else YOLO(str(model_path))
        self._confidence_threshold = confidence_threshold

    def analyze(
        self,
        image_path: str | Path,
    ) -> VisualIntelligenceResult:
        image = Path(image_path)

        if not image.is_file():
            raise FileNotFoundError(f"Image file does not exist: {image}")

        results = self._model(
            str(image),
            classes=[0],
            conf=self._confidence_threshold,
            imgsz=1024,
            verbose=False,
        )

        if not results or results[0].masks is None:
            return VisualIntelligenceResult(
                frame_path=image,
                subjects=(),
                analysis_version="1",
            )

        result = results[0]
        masks = result.masks.data
        boxes = result.boxes

        if len(masks) == 0 or boxes is None:
            return VisualIntelligenceResult(
                frame_path=image,
                subjects=(),
                analysis_version="1",
            )

        with Image.open(image) as source:
            width, height = source.size

        mask_output_dir = image.parent / "semantic_masks"
        mask_output_dir.mkdir(parents=True, exist_ok=True)

        subjects: list[SemanticSubject] = []

        for index in range(len(masks)):
            confidence = float(boxes.conf[index].item())

            mask = masks[index].detach().cpu().numpy()
            mask_image = Image.fromarray(
                (mask > 0.5).astype("uint8") * 255,
                mode="L",
            ).resize(
                (width, height),
                Image.Resampling.LANCZOS,
            )

            mask_path = mask_output_dir / (
                f"{image.stem}_person_{index + 1:02d}.png"
            )
            mask_image.save(mask_path)

            x1, y1, x2, y2 = (
                float(value)
                for value in boxes.xyxy[index].tolist()
            )

            bounds = BoundingBox(
                left=max(0.0, min(1.0, x1 / width)),
                top=max(0.0, min(1.0, y1 / height)),
                right=max(0.0, min(1.0, x2 / width)),
                bottom=max(0.0, min(1.0, y2 / height)),
            )

            area_ratio = sum(mask_image.getdata()) / (
                255.0 * width * height
            )

            importance = min(
                1.0,
                0.65 * confidence
                + 0.35 * min(1.0, area_ratio / 0.45),
            )

            subjects.append(
                SemanticSubject(
                    subject_id=f"person_{index + 1:02d}",
                    kind="person",
                    confidence=confidence,
                    bounds=bounds,
                    mask_path=mask_path,
                    area_ratio=max(1e-9, area_ratio),
                    importance=importance,
                )
            )

        return VisualIntelligenceResult(
            frame_path=image,
            subjects=tuple(subjects),
            analysis_version="1",
        )
