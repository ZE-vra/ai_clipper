from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from src.thumbnails.perception.quality import (
    FrameQualityEvidence,
)


class FrameQualityAnalyzer:
    """
    Deterministic image-quality analyzer.

    The analyzer observes an image and returns normalized quality
    evidence. It does not decide whether the image is suitable
    for a thumbnail.
    """

    def analyze(
        self,
        image_path: str | Path,
    ) -> FrameQualityEvidence:
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Image does not exist: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Image path is not a file: {path}"
            )

        with Image.open(path) as image:
            rgb = image.convert("RGB")
            array = np.asarray(
                rgb,
                dtype=np.float32,
            )

        brightness = self._brightness(array)
        contrast = self._contrast(array)
        saturation = self._saturation(array)
        sharpness = self._sharpness(array)
        motion_blur = self._motion_blur(array)
        noise = self._noise(array)

        overall_quality = self._overall_quality(
            sharpness=sharpness,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            motion_blur=motion_blur,
            noise=noise,
        )

        return FrameQualityEvidence(
            sharpness=sharpness,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            motion_blur=motion_blur,
            noise=noise,
            overall_quality=overall_quality,
        )

    @staticmethod
    def _luminance(
        array: np.ndarray,
    ) -> np.ndarray:
        return (
            0.2126 * array[:, :, 0]
            + 0.7152 * array[:, :, 1]
            + 0.0722 * array[:, :, 2]
        )

    def _brightness(
        self,
        array: np.ndarray,
    ) -> float:
        luminance = self._luminance(array)

        return float(
            np.clip(
                np.mean(luminance) / 255.0,
                0.0,
                1.0,
            )
        )

    def _contrast(
        self,
        array: np.ndarray,
    ) -> float:
        """
        Estimate luminance contrast using percentile spread.

        The 5th-to-95th percentile range measures the usable tonal
        range while ignoring a small number of extreme pixels.
        """
        luminance = self._luminance(array)

        low = float(
            np.percentile(
                luminance,
                5,
            )
        )

        high = float(
            np.percentile(
                luminance,
                95,
            )
        )

        spread = max(
            high - low,
            0.0,
        )

        return float(
            np.clip(
                spread / 255.0,
                0.0,
                1.0,
            )
        )

    def _saturation(
        self,
        array: np.ndarray,
    ) -> float:
        rgb = array / 255.0

        maximum = np.max(
            rgb,
            axis=2,
        )

        minimum = np.min(
            rgb,
            axis=2,
        )

        delta = maximum - minimum

        saturation = np.zeros_like(maximum)

        nonzero = maximum > 0

        saturation[nonzero] = (
            delta[nonzero]
            / maximum[nonzero]
        )

        return float(
            np.clip(
                np.mean(saturation),
                0.0,
                1.0,
            )
        )

    def _sharpness(
        self,
        array: np.ndarray,
    ) -> float:
        luminance = self._luminance(array)

        padded = np.pad(
            luminance,
            1,
            mode="reflect",
        )

        laplacian = (
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            - 4.0
            * padded[1:-1, 1:-1]
        )

        variance = float(
            np.var(laplacian)
        )

        return float(
            np.clip(
                variance / 500.0,
                0.0,
                1.0,
            )
        )

    def _motion_blur(
        self,
        array: np.ndarray,
    ) -> float:
        """
        Estimate blur using loss of high-frequency detail.

        This is a V1 blur proxy, not optical-flow motion
        estimation. Higher values indicate stronger evidence
        of blur.

        The implementation remains replaceable so a future
        CV-based blur detector can use the same evidence
        contract.
        """
        luminance = self._luminance(array)

        padded = np.pad(
            luminance,
            1,
            mode="reflect",
        )

        laplacian = (
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            - 4.0
            * padded[1:-1, 1:-1]
        )

        high_frequency_energy = float(
            np.mean(
                np.abs(laplacian)
            )
        )

        return float(
            np.clip(
                1.0
                - (
                    high_frequency_energy
                    / 32.0
                ),
                0.0,
                1.0,
            )
        )

    def _noise(
        self,
        array: np.ndarray,
    ) -> float:
        luminance = self._luminance(array)

        image = Image.fromarray(
            np.clip(
                luminance,
                0,
                255,
            ).astype(np.uint8)
        )

        smoothed = image.filter(
            ImageFilter.GaussianBlur(
                radius=1.0
            )
        )

        smooth_array = np.asarray(
            smoothed,
            dtype=np.float32,
        )

        residual = (
            luminance
            - smooth_array
        )

        noise_level = float(
            np.std(residual)
        )

        return float(
            np.clip(
                noise_level / 32.0,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _overall_quality(
        *,
        sharpness: float,
        brightness: float,
        contrast: float,
        saturation: float,
        motion_blur: float,
        noise: float,
    ) -> float:
        brightness_quality = (
            1.0
            - abs(
                brightness - 0.5
            )
            * 2.0
        )

        saturation_quality = min(
            saturation / 0.6,
            1.0,
        )

        quality = (
            0.30 * sharpness
            + 0.20 * contrast
            + 0.15
            * max(
                brightness_quality,
                0.0,
            )
            + 0.10
            * saturation_quality
            + 0.15
            * (1.0 - motion_blur)
            + 0.10
            * (1.0 - noise)
        )

        return float(
            np.clip(
                quality,
                0.0,
                1.0,
            )
        )