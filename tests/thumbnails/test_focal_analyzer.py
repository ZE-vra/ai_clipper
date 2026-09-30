from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.thumbnails.perception.focal_analyzer import (
    FocalAnalyzerConfig,
    FocalRegionAnalyzer,
)


def _save_image(
    path: Path,
    array: np.ndarray,
) -> None:
    Image.fromarray(
        array.astype(np.uint8),
        mode="RGB",
    ).save(path)


def _uniform_image(
    value: int = 128,
    size: int = 256,
) -> np.ndarray:
    return np.full(
        (size, size, 3),
        value,
        dtype=np.uint8,
    )


def _bright_block_image(
    size: int = 256,
) -> np.ndarray:
    image = _uniform_image(
        value=30,
        size=size,
    )

    image[
        80:176,
        144:240,
    ] = 255

    return image


def _color_block_image(
    size: int = 256,
) -> np.ndarray:
    image = _uniform_image(
        value=40,
        size=size,
    )

    image[
        64:192,
        144:240,
    ] = [240, 30, 30]

    return image


def test_analyzer_rejects_missing_file() -> None:
    analyzer = FocalRegionAnalyzer()

    with pytest.raises(
        FileNotFoundError
    ):
        analyzer.analyze(
            "does_not_exist.png"
        )


def test_analyzer_rejects_directory(
    tmp_path: Path,
) -> None:
    analyzer = FocalRegionAnalyzer()

    with pytest.raises(ValueError):
        analyzer.analyze(tmp_path)


def test_uniform_image_has_no_focal_regions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "uniform.png"

    _save_image(
        path,
        _uniform_image(),
    )

    evidence = FocalRegionAnalyzer().analyze(
        path
    )

    assert evidence.regions == ()


def test_bright_region_is_detected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bright.png"

    _save_image(
        path,
        _bright_block_image(),
    )

    evidence = FocalRegionAnalyzer().analyze(
        path
    )

    assert evidence.regions

    primary = evidence.primary_region

    assert primary is not None
    assert primary.strength > 0.20

    assert primary.focal_point.x > 0.50
    assert primary.focal_point.y > 0.25
    assert primary.focal_point.y < 0.75


def test_colorful_region_is_detected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "colorful.png"

    _save_image(
        path,
        _color_block_image(),
    )

    evidence = FocalRegionAnalyzer().analyze(
        path
    )

    assert evidence.primary_region is not None

    primary = evidence.primary_region

    assert primary.focal_point.x > 0.50


def test_region_bounds_are_not_grid_locked(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bright.png"

    image = _uniform_image(
        value=30,
    )

    image[
        55:181,
        137:239,
    ] = 255

    _save_image(
        path,
        image,
    )

    evidence = FocalRegionAnalyzer().analyze(
        path
    )

    primary = evidence.primary_region

    assert primary is not None

    bounds = primary.bounds

    assert bounds.left < 0.60
    assert bounds.right > 0.80
    assert bounds.top < 0.30
    assert bounds.bottom > 0.60


def test_region_bounds_are_normalized(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bright.png"

    _save_image(
        path,
        _bright_block_image(),
    )

    evidence = FocalRegionAnalyzer().analyze(
        path
    )

    for region in evidence.regions:
        bounds = region.bounds

        assert 0.0 <= bounds.left <= 1.0
        assert 0.0 <= bounds.top <= 1.0
        assert 0.0 <= bounds.right <= 1.0
        assert 0.0 <= bounds.bottom <= 1.0


def test_max_regions_is_respected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bright.png"

    image = _uniform_image(
        value=30,
    )

    image[
        20:70,
        20:70,
    ] = 255

    image[
        20:70,
        180:230,
    ] = 255

    image[
        180:230,
        20:70,
    ] = 255

    image[
        180:230,
        180:230,
    ] = 255

    _save_image(
        path,
        image,
    )

    analyzer = FocalRegionAnalyzer(
        FocalAnalyzerConfig(
            max_regions=2,
        )
    )

    evidence = analyzer.analyze(path)

    assert len(evidence.regions) <= 2


def test_regions_have_unique_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bright.png"

    _save_image(
        path,
        _bright_block_image(),
    )

    evidence = FocalRegionAnalyzer().analyze(
        path
    )

    ids = [
        region.region_id
        for region in evidence.regions
    ]

    assert len(ids) == len(set(ids))


def test_strength_is_absolute_not_rank_normalized(
    tmp_path: Path,
) -> None:
    weak_path = tmp_path / "weak.png"
    strong_path = tmp_path / "strong.png"

    weak = _uniform_image(
        value=100,
    )

    weak[
        80:176,
        144:240,
    ] = 130

    strong = _uniform_image(
        value=30,
    )

    strong[
        80:176,
        144:240,
    ] = 255

    _save_image(
        weak_path,
        weak,
    )

    _save_image(
        strong_path,
        strong,
    )

    analyzer = FocalRegionAnalyzer()

    weak_evidence = analyzer.analyze(
        weak_path
    )

    strong_evidence = analyzer.analyze(
        strong_path
    )

    assert weak_evidence.primary_region is not None
    assert strong_evidence.primary_region is not None

    assert (
        strong_evidence.primary_region.strength
        > weak_evidence.primary_region.strength
    )


def test_analysis_version_is_present(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bright.png"

    _save_image(
        path,
        _bright_block_image(),
    )

    evidence = FocalRegionAnalyzer().analyze(
        path
    )

    assert evidence.analysis_version == "2"