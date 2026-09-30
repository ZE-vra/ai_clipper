from __future__ import annotations

from dataclasses import dataclass


def _validate_score(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")


@dataclass(frozen=True)
class FrameQualityEvidence:
    """
    Deterministic visual-quality measurements for one frame.

    These values describe the frame. They do not determine whether
    the frame should become a thumbnail.
    """

    sharpness: float
    brightness: float
    contrast: float
    saturation: float
    motion_blur: float
    noise: float
    overall_quality: float

    def __post_init__(self) -> None:
        for name, value in (
            ("sharpness", self.sharpness),
            ("brightness", self.brightness),
            ("contrast", self.contrast),
            ("saturation", self.saturation),
            ("motion_blur", self.motion_blur),
            ("noise", self.noise),
            ("overall_quality", self.overall_quality),
        ):
            _validate_score(value, name)
