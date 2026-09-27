from src.packaging.gemini_packager import (
    GeminiPackager,
    PackagingError,
)
from src.schemas import ClipDecision


def test_gemini_packager_falls_back_after_primary_failure(
    monkeypatch,
):
    packager = GeminiPackager(
        api_key="test-api-key",
        model="gemini-3.8-flash",
        fallback_model="gemini-3.5-flash",
        max_retries=1,
    )

    calls = []

    expected = {
        "title": "Fallback Title",
        "hook": "Fallback Hook",
        "caption": "Fallback Caption",
        "description": "Fallback Description",
        "thumbnail_text": "FALLBACK",
        "content_angle": "surprise",
        "hashtags": [
            "#test",
            "#fallback",
        ],
    }

    def fake_generate_with_model(
        model_name,
        prompt,
        clip_id,
    ):
        calls.append(model_name)

        if model_name == "gemini-3.8-flash":
            raise PackagingError(
                "Model gemini-3.8-flash failed "
                "for clip 1: 503 UNAVAILABLE"
            )

        assert model_name == "gemini-3.5-flash"

        return expected

    monkeypatch.setattr(
        packager,
        "_generate_with_model",
        fake_generate_with_model,
    )

    clip = ClipDecision(
        clip_id=1,
        candidate_id=1,
        snapped_start_time=10.0,
        snapped_end_time=50.0,
        final_score=9.0,
        reason="Strong test clip.",
        title="Test Clip",
    )

    result = packager.package_clip(
        clip=clip,
        transcript_text="This is test transcript context.",
        source_title="Test Source",
    )

    assert calls == [
        "gemini-3.8-flash",
        "gemini-3.5-flash",
    ]

    assert result == expected