from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from src.thumbnails.domain.geometry import BoundingBox
from src.thumbnails.perception.ultralytics_subject_mask_provider import (
    UltralyticsSubjectMaskProvider,
)


class FakeModel:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return [self.result]


def _result(boxes, polygons):
    return SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=boxes,
            cls=[0] * len(boxes),
        ),
        masks=SimpleNamespace(xy=polygons),
    )


def test_provider_writes_original_size_mask_for_matching_subject(
    tmp_path: Path,
) -> None:
    source = tmp_path / "frame.png"
    Image.new("RGB", (100, 80), (10, 20, 30)).save(source)
    model = FakeModel(
        _result(
            boxes=[
                # A different person on the left.
                SimpleNamespace(tolist=lambda: [2, 5, 30, 70]),
                # Requested person on the right.
                SimpleNamespace(tolist=lambda: [60, 5, 95, 75]),
            ],
            polygons=[
                [(2, 5), (30, 5), (30, 70), (2, 70)],
                [(60, 5), (95, 5), (95, 75), (60, 75)],
            ],
        )
    )
    output = tmp_path / "masks" / "person.png"
    provider = UltralyticsSubjectMaskProvider(model)

    result = provider.create_mask(
        image_path=source,
        subject_bounds=BoundingBox(left=0.60, top=0.05, right=0.95, bottom=0.9375),
        output_path=output,
    )

    assert result == output
    assert model.calls[0][1] == {"classes": [0], "verbose": False}
    with Image.open(output) as mask:
        assert mask.mode == "L"
        assert mask.size == (100, 80)
        assert mask.getpixel((75, 30)) == 255
        assert mask.getpixel((10, 30)) == 0


def test_provider_rejects_when_no_detection_matches_subject(
    tmp_path: Path,
) -> None:
    source = tmp_path / "frame.png"
    Image.new("RGB", (100, 80)).save(source)
    model = FakeModel(
        _result(
            boxes=[SimpleNamespace(tolist=lambda: [2, 5, 30, 70])],
            polygons=[[(2, 5), (30, 5), (30, 70), (2, 70)]],
        )
    )
    provider = UltralyticsSubjectMaskProvider(model)

    with pytest.raises(ValueError, match="no segmentation mask matched"):
        provider.create_mask(
            image_path=source,
            subject_bounds=BoundingBox(left=0.60, top=0.05, right=0.95, bottom=0.90),
            output_path=tmp_path / "mask.png",
        )


def test_provider_rejects_empty_segmentation_results(tmp_path: Path) -> None:
    source = tmp_path / "frame.png"
    Image.new("RGB", (20, 20)).save(source)
    model = FakeModel(SimpleNamespace(boxes=None, masks=None))

    with pytest.raises(ValueError, match="no subject masks"):
        UltralyticsSubjectMaskProvider(model).create_mask(
            image_path=source,
            subject_bounds=BoundingBox(left=0.1, top=0.1, right=0.9, bottom=0.9),
            output_path=tmp_path / "mask.png",
        )
