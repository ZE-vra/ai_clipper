from src.packaging.models import ClipPackaging
from src.thumbnails.domain.content import ContentUnderstanding
from src.thumbnails.intelligence.packaging_adapter import (
    thumbnail_inputs_from_packaging,
)


def _packaging(**overrides):
    values = {
        "clip_id": 3,
        "title": "Luxury Plane Reveal",
        "hook": "THE $2M CARPET",
        "caption": "A look inside an unusual luxury jet.",
        "description": "A clip about a luxury jet interior.",
        "thumbnail_text": "Inside the most expensive private jet carpet ever made",
        "content_angle": "An unusually expensive carpet inside a private jet",
        "hashtags": ["#luxury"],
    }
    values.update(overrides)
    return ClipPackaging(**values)


def test_adapter_reuses_packaging_without_generating_content():
    brief, understanding = thumbnail_inputs_from_packaging(_packaging())

    # The adapter prefers existing concise packaging copy over a second model call.
    assert brief.core_hook == "THE $2M CARPET"
    assert brief.subject == "An unusually expensive carpet inside a private jet"
    assert brief.promise == "Luxury Plane Reveal"
    assert brief.curiosity_angle == "THE $2M CARPET"

    # Packaging does not contain entity/event evidence; do not fabricate it.
    assert isinstance(understanding, ContentUnderstanding)
    assert understanding.entities == ()
    assert understanding.events == ()
    assert understanding.themes == (
        "An unusually expensive carpet inside a private jet",
    )


def test_adapter_falls_back_to_shortest_existing_copy_when_no_concise_phrase():
    packaging = _packaging(
        title="A",
        hook="An extended hook explaining the entire story",
        thumbnail_text="A very long thumbnail phrase that will not fit",
        content_angle="",
    )

    brief, understanding = thumbnail_inputs_from_packaging(packaging)

    assert brief.core_hook == "A"
    assert brief.subject == "A"
    assert brief.promise == "A"
    assert brief.curiosity_angle == "An extended hook explaining the entire story"
    assert understanding.themes == ()
