from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ThumbnailResultStatus(str, Enum):
    SUCCESS = "success"
    FALLBACK_SUCCESS = "fallback_success"
    NO_ACCEPTABLE_RESULT = "no_acceptable_result"
    FAILED = "failed"


@dataclass(frozen=True)
class ThumbnailResult:
    """Final outcome of thumbnail production."""

    status: ThumbnailResultStatus

    output_path: Optional[str] = None
    selected_attempt_id: Optional[str] = None

    failure_reason: Optional[str] = None

    def __post_init__(self) -> None:
        successful = {
            ThumbnailResultStatus.SUCCESS,
            ThumbnailResultStatus.FALLBACK_SUCCESS,
        }

        if self.status in successful:
            if not self.output_path:
                raise ValueError(
                    "successful results must have "
                    "an output_path."
                )

            if not self.selected_attempt_id:
                raise ValueError(
                    "successful results must have "
                    "a selected_attempt_id."
                )

        if (
            self.status
            == ThumbnailResultStatus.NO_ACCEPTABLE_RESULT
            and not self.failure_reason
        ):
            raise ValueError(
                "no acceptable result must have "
                "a failure_reason."
            )