"""Deterministically adapt existing clip packaging into V2 thumbnail inputs.

This bridge deliberately performs no model calls. Packaging is already produced
by the clipper's per-clip Gemini request, so V2 should reuse those outputs
instead of asking Gemini to summarize the same clip a second time.
"""

from __future__ import annotations

from src.packaging.models import ClipPackaging
from src.thumbnails.domain.brief import ThumbnailBrief
from src.thumbnails.domain.content import ContentUnderstanding


def thumbnail_inputs_from_packaging(
    packaging: ClipPackaging,
) -> tuple[ThumbnailBrief, ContentUnderstanding]:
    """Build V2 inputs from saved packaging without inventing visual evidence.

    ClipPackaging contains publish copy, not entity/event extraction. The
    adapter therefore leaves entities and events empty rather than fabricating
    structured facts. V2 can still use the existing hook, title, thumbnail
    text, and content angle to guide creative concepts.
    """
    title = packaging.title.strip()
    hook = packaging.hook.strip()
    thumbnail_text = packaging.thumbnail_text.strip()
    content_angle = packaging.content_angle.strip()

    core_hook = thumbnail_text or hook or title
    subject = content_angle or title or hook
    promise = title or content_angle or hook
    curiosity_angle = hook or content_angle or title

    if not all((core_hook, subject, promise, curiosity_angle)):
        raise ValueError(
            "Clip packaging must contain at least one usable title, hook, "
            "thumbnail_text, or content_angle value to create thumbnail inputs."
        )

    brief = ThumbnailBrief(
        core_hook=core_hook,
        subject=subject,
        promise=promise,
        curiosity_angle=curiosity_angle,
    )
    understanding = ContentUnderstanding(
        entities=(),
        events=(),
        themes=(content_angle,) if content_angle else (),
        claims=(),
        confidence=0.0,
    )
    return brief, understanding
