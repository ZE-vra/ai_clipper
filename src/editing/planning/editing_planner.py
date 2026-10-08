from __future__ import annotations

from src.editing.models import EditingPlan, RenderConfig
from src.editing.planning.caption_planner import CaptionPlanner
from src.editing.planning.composition_planner import CompositionPlanner
from src.rendering.music.selector import MusicPlanner
from src.schemas import Transcript


class EditingPlanner:
    """
    Assembles the deterministic editing plan for a rendered clip.

    The planner combines:
    - composition decisions
    - clip-relative caption decisions
    - render configuration

    It does not render media and does not call AI services.
    """

    def __init__(
        self,
        *,
        composition_planner: CompositionPlanner | None = None,
        caption_planner: CaptionPlanner | None = None,
        render_config: RenderConfig | None = None,
        music_planner: MusicPlanner | None = None,
    ) -> None:
        self.composition_planner = (
            composition_planner
            or CompositionPlanner()
        )

        self.caption_planner = (
            caption_planner
            or CaptionPlanner()
        )

        self.render_config = (
            render_config
            or RenderConfig()
        )

        self.music_planner = (
            music_planner
            or MusicPlanner()
        )

    def create_plan(
        self,
        *,
        clip_id: str,
        transcript: Transcript,
        clip_start_time: float = 0.0,
        clip_end_time: float | None = None,
        content_reason: str = "",
        title: str | None = None,
    ) -> EditingPlan:
        if not clip_id.strip():
            raise ValueError(
                "clip_id must not be empty."
            )

        if clip_start_time < 0:
            raise ValueError(
                "clip_start_time must not be negative."
            )

        if (
            clip_end_time is not None
            and clip_end_time <= clip_start_time
        ):
            raise ValueError(
                "clip_end_time must be greater than "
                "clip_start_time."
            )

        composition = (
            self.composition_planner.create_plan(
                clip_id
            )
        )

        captions = (
            self.caption_planner.create_plan(
                transcript,
                clip_start_time=clip_start_time,
                clip_end_time=clip_end_time,
            )
        )

        music = self.music_planner.create_plan(
            title=title,
            reason=content_reason,
        )

        return EditingPlan(
            clip_id=clip_id,
            composition=composition,
            captions=captions,
            render_config=self.render_config,
            audio=music,
            clip_duration=(
                clip_end_time - clip_start_time
                if clip_end_time is not None
                else None
            ),
        )