from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from src.thumbnails.domain.geometry import BoundingBox


class UltralyticsSubjectMaskProvider:
    """Create person masks using an injected Ultralytics segmentation model.

    Inject the model explicitly (for example, a YOLO segmentation checkpoint)
    so callers control model weights and downloads. The detected person's
    normalized bounding box is matched to model detections using IoU; the
    corresponding polygon is rasterized at the original image dimensions.
    """

    def __init__(
        self,
        model: Any,
        *,
        person_class_id: int = 0,
        minimum_iou: float = 0.25,
    ) -> None:
        if not 0.0 <= minimum_iou <= 1.0:
            raise ValueError("minimum_iou must be between 0 and 1.")
        self._model = model
        self._person_class_id = person_class_id
        self._minimum_iou = minimum_iou

    def create_mask(
        self,
        *,
        image_path: str | Path,
        subject_bounds: BoundingBox,
        output_path: str | Path,
    ) -> Path:
        source_path = Path(image_path)
        output = Path(output_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"source image does not exist: {source_path}")

        with Image.open(source_path) as image:
            width, height = image.size

        results = self._model(
            str(source_path),
            classes=[self._person_class_id],
            verbose=False,
        )
        if not results:
            raise ValueError("segmentation model returned no results.")

        result = results[0]
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        polygons = getattr(masks, "xy", None) if masks is not None else None
        if boxes is None or masks is None or polygons is None or len(polygons) == 0:
            raise ValueError("segmentation model produced no subject masks.")

        target_box = (
            subject_bounds.left * width,
            subject_bounds.top * height,
            subject_bounds.right * width,
            subject_bounds.bottom * height,
        )
        best_index = -1
        best_iou = -1.0
        for index, box in enumerate(boxes.xyxy):
            class_ids = getattr(boxes, "cls", None)
            if class_ids is not None and int(class_ids[index]) != self._person_class_id:
                continue
            coordinates = tuple(float(value) for value in box.tolist())
            iou = self._intersection_over_union(target_box, coordinates)
            if iou > best_iou and index < len(polygons):
                best_index, best_iou = index, iou

        if best_index < 0 or best_iou < self._minimum_iou:
            raise ValueError(
                "no segmentation mask matched the requested subject "
                f"(best IoU: {max(best_iou, 0.0):.3f}, "
                f"minimum: {self._minimum_iou:.3f})."
            )

        mask = Image.new("L", (width, height), 0)
        polygon = polygons[best_index]
        points = [(float(point[0]), float(point[1])) for point in polygon]
        if len(points) < 3:
            raise ValueError("matched subject mask polygon has fewer than 3 points.")
        ImageDraw.Draw(mask).polygon(points, fill=255)

        output.parent.mkdir(parents=True, exist_ok=True)
        mask.save(output, format="PNG")
        return output

    @staticmethod
    def _intersection_over_union(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> float:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0
