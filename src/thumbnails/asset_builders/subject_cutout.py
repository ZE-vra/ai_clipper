from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.thumbnails.domain.assets import (
    AssetProvenance,
    AssetProvenanceKind,
    VisualAsset,
)


class MaskedSubjectCutoutBuilder:
    """Create a transparent subject asset from a source image and segmentation mask.

    The mask is supplied by a segmentation provider; this class does not
    infer a mask or make creative decisions. White mask pixels are opaque,
    black pixels are transparent, and intermediate grayscale values become
    partial alpha. Source and mask dimensions must match to prevent silent
    spatial misalignment.
    """

    def build(
        self,
        *,
        source_asset: VisualAsset,
        mask_path: str | Path,
        output_path: str | Path,
    ) -> VisualAsset:
        source_path = Path(source_asset.path)
        mask_source = Path(mask_path)
        output = Path(output_path)

        if not source_path.is_file():
            raise FileNotFoundError(
                f"source asset does not exist: {source_path}"
            )
        if not mask_source.is_file():
            raise FileNotFoundError(
                f"subject segmentation mask does not exist: {mask_source}"
            )

        with Image.open(source_path) as source_image:
            source = source_image.convert("RGBA")
        with Image.open(mask_source) as mask_image:
            mask = mask_image.convert("L")

        if source.size != mask.size:
            raise ValueError(
                "subject mask dimensions must match source asset dimensions: "
                f"{mask.size} != {source.size}"
            )

        source.putalpha(mask)
        output.parent.mkdir(parents=True, exist_ok=True)
        source.save(output, format="PNG")

        return VisualAsset(
            asset_id=f"{source_asset.asset_id}-subject-cutout",
            provenance=AssetProvenance(
                kind=AssetProvenanceKind.SUBJECT_CUTOUT,
                source_asset_ids=(source_asset.asset_id,),
                source_timestamps=(
                    (source_asset.source_timestamp,)
                    if source_asset.source_timestamp is not None
                    else ()
                ),
            ),
            path=str(output),
            source_timestamp=source_asset.source_timestamp,
            bounds=source_asset.bounds,
            subject_mask_path=str(mask_source),
        )
