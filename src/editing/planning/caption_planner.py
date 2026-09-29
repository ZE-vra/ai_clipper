from __future__ import annotations

from src.editing.models import CaptionPlan, CaptionSegment, CaptionStyle
from src.schemas import Transcript


class CaptionPlanner:
    """
    Converts transcript segments into a social-media-friendly
    clip-relative caption plan.

    Transcript timestamps are source-video timestamps.

    When a clip range is supplied, timestamps are converted so
    the selected clip begins at 0.0 seconds.

    Caption content and timing remain deterministic and are derived
    directly from the Whisper transcript.
    """

    def __init__(
        self,
        *,
        max_words: int = 6,
        max_characters: int = 32,
        max_lines: int = 2,
    ) -> None:
        self.max_words = max_words
        self.max_characters = max_characters
        self.max_lines = max_lines

    def create_plan(
        self,
        transcript: Transcript,
        *,
        clip_start_time: float = 0.0,
        clip_end_time: float | None = None,
    ) -> CaptionPlan:
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

        segments: list[CaptionSegment] = []

        for segment in transcript.segments:
            if clip_end_time is not None:
                if segment.end <= clip_start_time:
                    continue

                if segment.start >= clip_end_time:
                    continue

            effective_start = max(
                segment.start,
                clip_start_time,
            )

            effective_end = segment.end

            if clip_end_time is not None:
                effective_end = min(
                    effective_end,
                    clip_end_time,
                )

            if effective_end <= effective_start:
                continue

            text = segment.text.strip()

            if not text:
                continue

            relative_start = (
                effective_start - clip_start_time
            )

            relative_end = (
                effective_end - clip_start_time
            )

            for caption in self._split_segment(
                text=text,
                start_time=relative_start,
                end_time=relative_end,
            ):
                segments.append(caption)

        return CaptionPlan(
            segments=segments,
            style=CaptionStyle(
                font_size=54,
                font_name="Arial",
                font_weight="bold",
                horizontal_alignment="center",
                vertical_position="lower_middle",
                horizontal_margin=80,
                vertical_margin=500,
                max_lines=self.max_lines,
                outline_width=4,
                shadow=True,
            ),
            enabled=True,
        )

    def _split_segment(
        self,
        *,
        text: str,
        start_time: float,
        end_time: float,
    ) -> list[CaptionSegment]:
        words = text.split()

        if not words:
            return []

        chunks: list[list[str]] = []
        current: list[str] = []

        for word in words:
            candidate = " ".join(
                current + [word]
            )

            if (
                current
                and (
                    len(current) >= self.max_words
                    or len(candidate) > self.max_characters
                )
            ):
                chunks.append(current)
                current = [word]
            else:
                current.append(word)

        if current:
            chunks.append(current)

        duration = end_time - start_time

        if duration <= 0:
            return []

        total_words = sum(
            len(chunk)
            for chunk in chunks
        )

        captions: list[CaptionSegment] = []
        elapsed_words = 0

        for chunk in chunks:
            word_count = len(chunk)

            chunk_start = (
                start_time
                + duration
                * elapsed_words
                / total_words
            )

            elapsed_words += word_count

            chunk_end = (
                start_time
                + duration
                * elapsed_words
                / total_words
            )

            captions.append(
                CaptionSegment(
                    start_time=chunk_start,
                    end_time=chunk_end,
                    text=" ".join(chunk),
                )
            )

        return captions