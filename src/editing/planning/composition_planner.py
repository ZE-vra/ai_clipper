from __future__ import annotations

from src.editing.models import (
    BackgroundPlan,
    CanvasPlan,
    CompositionPlan,
    ForegroundPlan,
)


class CompositionPlanner:
    """
    Creates a deterministic vertical-video composition plan.

    V1 uses the original video twice:
    - a strongly blurred/dimmed copy fills the vertical canvas
    - a clear copy remains in the foreground with its aspect ratio preserved

    The background exists to fill the canvas without competing
    with the foreground video.

    No face detection, tracking, or AI is required.
    """

    def __init__(
        self,
        *,
        canvas_width: int = 1080,
        canvas_height: int = 1920,
        blur_radius: float = 32.0,
        background_brightness: float = 0.58,
        foreground_scale: float = 1.0,
    ) -> None:
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.blur_radius = blur_radius
        self.background_brightness = background_brightness
        self.foreground_scale = foreground_scale

    def create_plan(self, clip_id: str) -> CompositionPlan:
        if not clip_id.strip():
            raise ValueError("clip_id must not be empty.")

        if self.canvas_width <= 0:
            raise ValueError("canvas_width must be greater than zero.")

        if self.canvas_height <= 0:
            raise ValueError("canvas_height must be greater than zero.")

        if self.blur_radius < 0:
            raise ValueError("blur_radius must not be negative.")

        if not 0.0 < self.background_brightness <= 1.0:
            raise ValueError(
                "background_brightness must be greater than 0 "
                "and less than or equal to 1."
            )

        if self.foreground_scale <= 0:
            raise ValueError(
                "foreground_scale must be greater than zero."
            )

        return CompositionPlan(
            canvas=CanvasPlan(
                width=self.canvas_width,
                height=self.canvas_height,
            ),
            background=BackgroundPlan(
                source="same_video",
                blur_radius=self.blur_radius,
                brightness=self.background_brightness,
            ),
            foreground=ForegroundPlan(
                preserve_aspect_ratio=True,
                scale=self.foreground_scale,
            ),
        )