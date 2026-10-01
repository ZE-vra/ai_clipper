from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ThumbnailBrief:
    """Creative brief derived from content understanding and packaging."""

    core_hook: str
    subject: str
    promise: str
    curiosity_angle: str
    emotional_direction: str = ""
    important_entities: tuple[str, ...] = ()
    important_objects: tuple[str, ...] = ()
    important_locations: tuple[str, ...] = ()
    visual_evidence: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    forbidden_misrepresentations: tuple[str, ...] = ()
    content_understanding_id: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("core_hook", self.core_hook),
            ("subject", self.subject),
            ("promise", self.promise),
            ("curiosity_angle", self.curiosity_angle),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be blank.")


# Backwards-compatible name while the new orchestration path migrates.
CreativeBrief = ThumbnailBrief
