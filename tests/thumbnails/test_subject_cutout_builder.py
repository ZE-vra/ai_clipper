from pathlib import Path

import pytest
from PIL import Image

from src.thumbnails.asset_builders.subject_cutout import (
    MaskedSubjectCutoutBuilder,
)
from src.thumbnails.domain.assets import (
    AssetProvenance,
    AssetProvenanceKind,
    VisualAsset,
)


def _source(path: Path) -> VisualAsset:
    return VisualAsset(
        asset_id="frame-1",
        provenance=AssetProvenance(
            kind=AssetProvenanceKind.SOURCE_FRAME,
        ),
        path=str(path),
        source_timestamp=12.5,
    )


def test_builder_materializes_mask_as_png_alpha_and_preserves_lineage(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.jpg"
    source_image = Image.new("RGB", (3, 2), (200, 100, 50))
    source_image.save(source_path)

    mask_path = tmp_path / "mask.png"
    mask = Image.new("L", (3, 2), 0)
    mask.putpixel((1, 0), 128)
    mask.putpixel((2, 1), 255)
    mask.save(mask_path)

    output_path = tmp_path / "derived" / "cutout.png"
    result = MaskedSubjectCutoutBuilder().build(
        source_asset=_source(source_path),
        mask_path=mask_path,
        output_path=output_path,
    )

    assert output_path.is_file()
    assert result.asset_id == "frame-1-subject-cutout"
    assert result.provenance.kind is AssetProvenanceKind.SUBJECT_CUTOUT
    assert result.provenance.source_asset_ids == ("frame-1",)
    assert result.provenance.source_timestamps == (12.5,)
    assert result.subject_mask_path == str(mask_path)

    with Image.open(output_path) as cutout:
        assert cutout.mode == "RGBA"
        assert cutout.getpixel((0, 0))[3] == 0
        assert cutout.getpixel((1, 0))[3] == 128
        assert cutout.getpixel((2, 1))[3] == 255


def test_builder_rejects_misaligned_mask_without_resizing(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.png"
    Image.new("RGB", (10, 8), (200, 100, 50)).save(source_path)
    mask_path = tmp_path / "mask.png"
    Image.new("L", (5, 4), 255).save(mask_path)

    with pytest.raises(ValueError, match="dimensions must match"):
        MaskedSubjectCutoutBuilder().build(
            source_asset=_source(source_path),
            mask_path=mask_path,
            output_path=tmp_path / "cutout.png",
        )


def test_builder_reports_missing_mask(tmp_path: Path) -> None:
    source_path = tmp_path / "source.png"
    Image.new("RGB", (10, 8), (200, 100, 50)).save(source_path)

    with pytest.raises(FileNotFoundError, match="segmentation mask"):
        MaskedSubjectCutoutBuilder().build(
            source_asset=_source(source_path),
            mask_path=tmp_path / "missing-mask.png",
            output_path=tmp_path / "cutout.png",
        )
