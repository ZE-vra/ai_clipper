from __future__ import annotations

from dataclasses import dataclass


def _validate_normalized(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be between 0 and 1."
        )


@dataclass(frozen=True)
class Point:
    """A normalized point on a thumbnail canvas."""

    x: float
    y: float

    def __post_init__(self) -> None:
        _validate_normalized(self.x, "x")
        _validate_normalized(self.y, "y")


@dataclass(frozen=True)
class Size:
    """A positive pixel size."""

    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError(
                "width must be greater than 0."
            )

        if self.height <= 0:
            raise ValueError(
                "height must be greater than 0."
            )


@dataclass(frozen=True)
class BoundingBox:
    """A normalized rectangular region."""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        for name, value in (
            ("left", self.left),
            ("top", self.top),
            ("right", self.right),
            ("bottom", self.bottom),
        ):
            _validate_normalized(value, name)

        if self.left >= self.right:
            raise ValueError(
                "left must be less than right."
            )

        if self.top >= self.bottom:
            raise ValueError(
                "top must be less than bottom."
            )

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def center(self) -> Point:
        return Point(
            x=(self.left + self.right) / 2,
            y=(self.top + self.bottom) / 2,
        )


@dataclass(frozen=True)
class Region:
    """A named normalized region of the thumbnail."""

    name: str
    bounds: BoundingBox

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError(
                "name must not be blank."
            )