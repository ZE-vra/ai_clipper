from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from src.thumbnails.domain.geometry import (
    BoundingBox,
    Point,
)
from src.thumbnails.perception.focal import (
    FocalAnalysisEvidence,
    FocalRegionEvidence,
)


@dataclass(frozen=True)
class FocalAnalyzerConfig:
    """Configuration for deterministic focal-region detection."""

    blur_radius: float = 3.0

    # Minimum adaptive prominence required for a region.
    threshold: float = 0.20

    minimum_region_strength: float = 0.10

    minimum_area_ratio: float = 0.005
    maximum_area_ratio: float = 0.60

    minimum_region_distance: float = 0.04

    max_regions: int = 4

    edge_weight: float = 0.15
    contrast_weight: float = 0.40
    saturation_weight: float = 0.20
    luminance_weight: float = 0.25

    # Only the visually strongest portion of a real frame
    # becomes candidate focal evidence.
    candidate_percentile: float = 85.0

    def __post_init__(self) -> None:
        if self.blur_radius < 0.0:
            raise ValueError(
                "blur_radius must not be negative."
            )

        for name, value in (
            ("threshold", self.threshold),
            (
                "minimum_region_strength",
                self.minimum_region_strength,
            ),
            ("minimum_area_ratio", self.minimum_area_ratio),
            ("maximum_area_ratio", self.maximum_area_ratio),
            (
                "minimum_region_distance",
                self.minimum_region_distance,
            ),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1."
                )

        if not 0.0 <= self.candidate_percentile <= 100.0:
            raise ValueError(
                "candidate_percentile must be between 0 and 100."
            )

        if (
            self.minimum_area_ratio
            > self.maximum_area_ratio
        ):
            raise ValueError(
                "minimum_area_ratio must not exceed "
                "maximum_area_ratio."
            )

        if self.max_regions <= 0:
            raise ValueError(
                "max_regions must be greater than 0."
            )

        weights = (
            self.edge_weight,
            self.contrast_weight,
            self.saturation_weight,
            self.luminance_weight,
        )

        if any(weight < 0.0 for weight in weights):
            raise ValueError(
                "feature weights must not be negative."
            )

        if sum(weights) <= 0.0:
            raise ValueError(
                "at least one feature weight must be greater than 0."
            )


class FocalRegionAnalyzer:
    """
    Deterministic analyzer for visually significant image regions.

    The analyzer combines several visual signals and converts them
    into adaptive, frame-relative focal evidence.

    It does not identify semantic subjects and does not decide
    whether a region is suitable for a thumbnail.
    """

    def __init__(
        self,
        config: FocalAnalyzerConfig | None = None,
    ) -> None:
        self.config = (
            config
            or FocalAnalyzerConfig()
        )

    def analyze(
        self,
        image_path: str | Path,
    ) -> FocalAnalysisEvidence:
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

        energy = self._visual_energy(array)

        if self.config.blur_radius > 0:
            energy = self._smooth_energy(
                energy
            )

        prominence = self._adaptive_prominence(
            energy
        )

        regions = self._extract_regions(
            prominence,
            energy,
        )

        return FocalAnalysisEvidence(
            regions=tuple(regions),
            analysis_version="2",
        )

    def _visual_energy(
        self,
        array: np.ndarray,
    ) -> np.ndarray:
        luminance = self._luminance(array)
        saturation = self._saturation(array)

        edges = self._edge_energy(
            luminance
        )

        contrast = self._regional_contrast_map(
            luminance
        )

        luminance_interest = (
            self._luminance_interest_map(
                luminance
            )
        )

        weights = (
            self.config.edge_weight,
            self.config.contrast_weight,
            self.config.saturation_weight,
            self.config.luminance_weight,
        )

        weight_sum = sum(weights)

        energy = (
            self.config.edge_weight * edges
            + self.config.contrast_weight * contrast
            + self.config.saturation_weight * saturation
            + self.config.luminance_weight
            * luminance_interest
        ) / weight_sum

        return np.clip(
            energy,
            0.0,
            1.0,
        )

    def _smooth_energy(
        self,
        energy: np.ndarray,
    ) -> np.ndarray:
        image = Image.fromarray(
            np.clip(
                energy * 255.0,
                0,
                255,
            ).astype(np.uint8)
        )

        smoothed = image.filter(
            ImageFilter.GaussianBlur(
                radius=self.config.blur_radius
            )
        )

        return (
            np.asarray(
                smoothed,
                dtype=np.float32,
            )
            / 255.0
        )

    @staticmethod
    def _adaptive_prominence(
        energy: np.ndarray,
    ) -> np.ndarray:
        """
        Convert absolute visual energy into frame-relative
        prominence.

        A real video frame can have substantial visual energy
        everywhere. What matters here is which areas are
        unusually strong relative to that particular frame.

        The median provides the frame's ordinary baseline.
        The 95th percentile provides a robust high-energy
        reference without being controlled by a single pixel.
        """
        median = float(
            np.percentile(
                energy,
                50,
            )
        )

        high_reference = float(
            np.percentile(
                energy,
                95,
            )
        )

        spread = high_reference - median

        if spread <= 1e-6:
            return np.zeros_like(
                energy
            )

        prominence = (
            energy - median
        ) / spread

        return np.clip(
            prominence,
            0.0,
            1.0,
        )

    def _extract_regions(
        self,
        prominence: np.ndarray,
        energy: np.ndarray,
    ) -> list[FocalRegionEvidence]:
        if float(np.max(prominence)) <= 0.0:
            return []

        percentile_cutoff = float(
            np.percentile(
                prominence,
                self.config.candidate_percentile,
            )
        )

        threshold = max(
            self.config.threshold,
            percentile_cutoff,
        )

        mask = prominence >= threshold

        components = self._connected_components(
            mask
        )

        candidates: list[
            FocalRegionEvidence
        ] = []

        for component in components:
            region = self._component_to_region(
                component,
                prominence,
                energy,
            )

            if region is None:
                continue

            candidates.append(region)

        candidates.sort(
            key=lambda region: region.strength,
            reverse=True,
        )

        selected: list[
            FocalRegionEvidence
        ] = []

        for region in candidates:
            if self._too_close_to_selected(
                region,
                selected,
            ):
                continue

            selected.append(region)

            if (
                len(selected)
                >= self.config.max_regions
            ):
                break

        return [
            FocalRegionEvidence(
                region_id=(
                    f"focal_{index:02d}"
                ),
                bounds=region.bounds,
                focal_point=region.focal_point,
                strength=region.strength,
                area_ratio=region.area_ratio,
            )
            for index, region in enumerate(
                selected,
                start=1,
            )
        ]

    def _component_to_region(
        self,
        component: list[tuple[int, int]],
        prominence: np.ndarray,
        energy: np.ndarray,
    ) -> FocalRegionEvidence | None:
        height, width = prominence.shape

        area_ratio = (
            len(component)
            / float(height * width)
        )

        if (
            area_ratio
            < self.config.minimum_area_ratio
        ):
            return None

        if (
            area_ratio
            > self.config.maximum_area_ratio
        ):
            return None

        rows = np.asarray(
            [point[0] for point in component],
            dtype=np.int32,
        )

        columns = np.asarray(
            [point[1] for point in component],
            dtype=np.int32,
        )

        prominence_values = prominence[
            rows,
            columns,
        ]

        energy_values = energy[
            rows,
            columns,
        ]

        strength = self._region_strength(
            prominence_values,
            energy_values,
            area_ratio,
        )

        if (
            strength
            < self.config.minimum_region_strength
        ):
            return None

        bounds = BoundingBox(
            left=float(
                np.min(columns) / width
            ),
            top=float(
                np.min(rows) / height
            ),
            right=float(
                (np.max(columns) + 1) / width
            ),
            bottom=float(
                (np.max(rows) + 1) / height
            ),
        )

        weights = np.maximum(
            prominence_values,
            1e-6,
        )

        focal_x = float(
            np.average(
                columns / max(width - 1, 1),
                weights=weights,
            )
        )

        focal_y = float(
            np.average(
                rows / max(height - 1, 1),
                weights=weights,
            )
        )

        return FocalRegionEvidence(
            region_id="pending",
            bounds=bounds,
            focal_point=Point(
                x=float(
                    np.clip(
                        focal_x,
                        0.0,
                        1.0,
                    )
                ),
                y=float(
                    np.clip(
                        focal_y,
                        0.0,
                        1.0,
                    )
                ),
            ),
            strength=strength,
            area_ratio=area_ratio,
        )

    @staticmethod
    def _region_strength(
        prominence_values: np.ndarray,
        energy_values: np.ndarray,
        area_ratio: float,
    ) -> float:
        mean_prominence = float(
            np.mean(prominence_values)
        )

        peak_prominence = float(
            np.percentile(
                prominence_values,
                90,
            )
        )

        mean_energy = float(
            np.mean(energy_values)
        )

        concentration = min(
            area_ratio / 0.10,
            1.0,
        )

        strength = (
            0.50 * mean_prominence
            + 0.25 * peak_prominence
            + 0.15 * mean_energy
            + 0.10 * concentration
        )

        return float(
            np.clip(
                strength,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _connected_components(
        mask: np.ndarray,
    ) -> list[list[tuple[int, int]]]:
        height, width = mask.shape

        visited = np.zeros(
            mask.shape,
            dtype=bool,
        )

        components: list[
            list[tuple[int, int]]
        ] = []

        neighbors = (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )

        for row in range(height):
            for column in range(width):
                if (
                    not mask[row, column]
                    or visited[row, column]
                ):
                    continue

                stack = [
                    (row, column)
                ]

                visited[
                    row,
                    column,
                ] = True

                component: list[
                    tuple[int, int]
                ] = []

                while stack:
                    current_row, current_column = (
                        stack.pop()
                    )

                    component.append(
                        (
                            current_row,
                            current_column,
                        )
                    )

                    for (
                        row_offset,
                        column_offset,
                    ) in neighbors:
                        neighbor_row = (
                            current_row
                            + row_offset
                        )

                        neighbor_column = (
                            current_column
                            + column_offset
                        )

                        if not (
                            0 <= neighbor_row < height
                            and 0 <= neighbor_column < width
                        ):
                            continue

                        if (
                            visited[
                                neighbor_row,
                                neighbor_column,
                            ]
                        ):
                            continue

                        if not mask[
                            neighbor_row,
                            neighbor_column,
                        ]:
                            continue

                        visited[
                            neighbor_row,
                            neighbor_column,
                        ] = True

                        stack.append(
                            (
                                neighbor_row,
                                neighbor_column,
                            )
                        )

                components.append(
                    component
                )

        return components

    def _too_close_to_selected(
        self,
        region: FocalRegionEvidence,
        selected: list[FocalRegionEvidence],
    ) -> bool:
        for other in selected:
            distance = self._point_distance(
                region.focal_point,
                other.focal_point,
            )

            if (
                distance
                < self.config.minimum_region_distance
            ):
                return True

        return False

    @staticmethod
    def _point_distance(
        first: Point,
        second: Point,
    ) -> float:
        return float(
            np.sqrt(
                (first.x - second.x) ** 2
                + (first.y - second.y) ** 2
            )
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

    @staticmethod
    def _saturation(
        array: np.ndarray,
    ) -> np.ndarray:
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

        saturation = np.zeros_like(
            maximum
        )

        nonzero = maximum > 0

        saturation[nonzero] = (
            delta[nonzero]
            / maximum[nonzero]
        )

        return saturation

    @staticmethod
    def _edge_energy(
        luminance: np.ndarray,
    ) -> np.ndarray:
        padded = np.pad(
            luminance,
            1,
            mode="reflect",
        )

        horizontal = (
            padded[1:-1, 2:]
            - padded[1:-1, :-2]
        )

        vertical = (
            padded[2:, 1:-1]
            - padded[:-2, 1:-1]
        )

        magnitude = np.sqrt(
            horizontal * horizontal
            + vertical * vertical
        )

        return 1.0 - np.exp(
            -magnitude / 48.0
        )

    @staticmethod
    def _regional_contrast_map(
        luminance: np.ndarray,
    ) -> np.ndarray:
        """
        Measure how different each pixel is from its
        larger-scale surroundings.
        """
        image = Image.fromarray(
            np.clip(
                luminance,
                0,
                255,
            ).astype(np.uint8)
        )

        blurred = image.filter(
            ImageFilter.GaussianBlur(
                radius=24.0
            )
        )

        surrounding_luminance = np.asarray(
            blurred,
            dtype=np.float32,
        )

        difference = np.abs(
            luminance
            - surrounding_luminance
        )

        return 1.0 - np.exp(
            -difference / 32.0
        )

    @staticmethod
    def _luminance_interest_map(
        luminance: np.ndarray,
    ) -> np.ndarray:
        mean = float(
            np.mean(luminance)
        )

        std = float(
            np.std(luminance)
        )

        if std <= 1e-6:
            return np.zeros_like(
                luminance
            )

        deviation = np.abs(
            luminance - mean
        )

        return 1.0 - np.exp(
            -deviation / (
                std * 1.5
            )
        )