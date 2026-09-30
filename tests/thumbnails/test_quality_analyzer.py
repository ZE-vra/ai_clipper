from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

from src.thumbnails.perception.quality_analyzer import (
    FrameQualityAnalyzer,
)


def _save_image(
    path: Path,
    array: np.ndarray,
) -> None:
    Image.fromarray(
        array.astype(np.uint8),
        mode="RGB",
    ).save(path)


def _sharp_pattern(
    size: int = 256,
) -> np.ndarray:
    image = np.zeros(
        (size, size, 3),
        dtype=np.uint8,
    )

    block = 32

    for row in range(
        0,
        size,
        block,
    ):
        for column in range(
            0,
            size,
            block,
        ):
            if (
                row // block
                + column // block
            ) % 2 == 0:
                image[
                    row : row + block,
                    column : column + block,
                ] = 255

    return image


def _gray_image(
    value: int,
    size: int = 256,
) -> np.ndarray:
    return np.full(
        (size, size, 3),
        value,
        dtype=np.uint8,
    )


def _colorful_image(
    size: int = 256,
) -> np.ndarray:
    image = np.zeros(
        (size, size, 3),
        dtype=np.uint8,
    )

    image[
        :,
        : size // 3,
    ] = [220, 30, 30]

    image[
        :,
        size // 3 : 2 * size // 3,
    ] = [30, 220, 30]

    image[
        :,
        2 * size // 3 :,
    ] = [30, 30, 220]

    return image


def test_analyzer_rejects_missing_file() -> None:
    analyzer = FrameQualityAnalyzer()

    missing_path = Path(
        "tests/thumbnails/does_not_exist.png"
    )

    with pytest.raises(
        FileNotFoundError
    ):
        analyzer.analyze(missing_path)


def test_analyzer_rejects_directory(
    tmp_path: Path,
) -> None:
    analyzer = FrameQualityAnalyzer()

    with pytest.raises(ValueError):
        analyzer.analyze(tmp_path)


def test_black_image_has_zero_brightness(
    tmp_path: Path,
) -> None:
    path = tmp_path / "black.png"

    _save_image(
        path,
        _gray_image(0),
    )

    evidence = FrameQualityAnalyzer().analyze(
        path
    )

    assert evidence.brightness == 0.0


def test_white_image_has_maximum_brightness(
    tmp_path: Path,
) -> None:
    path = tmp_path / "white.png"

    _save_image(
        path,
        _gray_image(255),
    )

    evidence = FrameQualityAnalyzer().analyze(
        path
    )

    assert evidence.brightness == 1.0


def test_gray_image_has_midrange_brightness(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gray.png"

    _save_image(
        path,
        _gray_image(128),
    )

    evidence = FrameQualityAnalyzer().analyze(
        path
    )

    assert evidence.brightness == pytest.approx(
        128 / 255,
        abs=0.01,
    )


def test_uniform_image_has_no_contrast(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gray.png"

    _save_image(
        path,
        _gray_image(128),
    )

    evidence = FrameQualityAnalyzer().analyze(
        path
    )

    assert evidence.contrast == 0.0


def test_high_contrast_image_has_high_contrast_measurement(
    tmp_path: Path,
) -> None:
    path = tmp_path / "high_contrast.png"

    _save_image(
        path,
        _sharp_pattern(),
    )

    evidence = FrameQualityAnalyzer().analyze(
        path
    )

    assert evidence.contrast > 0.9


def test_medium_contrast_image_is_below_high_contrast(
    tmp_path: Path,
) -> None:
    path = tmp_path / "medium_contrast.png"

    image = np.zeros(
        (256, 256, 3),
        dtype=np.uint8,
    )

    image[:, :128] = 80
    image[:, 128:] = 180

    _save_image(
        path,
        image,
    )

    evidence = FrameQualityAnalyzer().analyze(
        path
    )

    assert 0.3 < evidence.contrast < 0.5


def test_sharp_pattern_has_more_sharpness_than_blurred_pattern(
    tmp_path: Path,
) -> None:
    sharp_path = tmp_path / "sharp.png"
    blurred_path = tmp_path / "blurred.png"

    sharp = Image.fromarray(
        _sharp_pattern(),
        mode="RGB",
    )

    sharp.save(sharp_path)

    sharp.filter(
        ImageFilter.GaussianBlur(
            radius=8
        )
    ).save(blurred_path)

    analyzer = FrameQualityAnalyzer()

    sharp_evidence = analyzer.analyze(
        sharp_path
    )

    blurred_evidence = analyzer.analyze(
        blurred_path
    )

    assert (
        sharp_evidence.sharpness
        > blurred_evidence.sharpness
    )


def test_colorful_image_has_more_saturation_than_gray(
    tmp_path: Path,
) -> None:
    colorful_path = (
        tmp_path / "colorful.png"
    )

    gray_path = (
        tmp_path / "gray.png"
    )

    _save_image(
        colorful_path,
        _colorful_image(),
    )

    _save_image(
        gray_path,
        _gray_image(128),
    )

    analyzer = FrameQualityAnalyzer()

    colorful = analyzer.analyze(
        colorful_path
    )

    gray = analyzer.analyze(
        gray_path
    )

    assert (
        colorful.saturation
        > gray.saturation
    )


def test_all_measurements_are_normalized(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pattern.png"

    _save_image(
        path,
        _sharp_pattern(),
    )

    evidence = FrameQualityAnalyzer().analyze(
        path
    )

    values = (
        evidence.sharpness,
        evidence.brightness,
        evidence.contrast,
        evidence.saturation,
        evidence.motion_blur,
        evidence.noise,
        evidence.overall_quality,
    )

    assert all(
        0.0 <= value <= 1.0
        for value in values
    )