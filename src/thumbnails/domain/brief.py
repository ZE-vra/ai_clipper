from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class CreativeBrief:
    """
    Creative direction derived from the underlying clip.
    """

    core_hook: str
    subject: str
    promise: str
    curiosity_angle: str

    emotional_direction: str = ""

    important_entities: Tuple[str, ...] = field(
        default_factory=tuple
    )

    visual_evidence: Tuple[str, ...] = field(
        default_factory=tuple
    )

    constraints: Tuple[str, ...] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        required = (
            ("core_hook", self.core_hook),
            ("subject", self.subject),
            ("promise", self.promise),
            (
                "curiosity_angle",
                self.curiosity_angle,
            ),
        )

        for name, value in required:
            if not value.strip():
                raise ValueError(
                    f"{name} must not be blank."
                )