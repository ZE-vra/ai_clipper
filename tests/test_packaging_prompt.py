from src.packaging.gemini_packager import GeminiPackager
from src.schemas import ClipDecision


def test_packaging_prompt_prioritizes_specific_curiosity_without_fabrication():
    clip = ClipDecision(
        clip_id=1,
        candidate_id=1,
        snapped_start_time=12.0,
        snapped_end_time=30.0,
        final_score=9.0,
        reason="A surprising reveal",
        title="A test clip",
    )

    prompt = GeminiPackager._build_prompt(
        clip=clip,
        transcript_text="She gave away the entire prize after one question.",
        source_title="A test video",
    )

    assert "CURIOSITY-FIRST, HIGH-CLICK-THROUGH PACKAGING" in prompt
    assert "Do not invent stakes, numbers" in prompt
    assert "This is the thumbnail's main click trigger" in prompt
    assert "2-5 punchy words" in prompt
    assert "do not merely repeat the title" in prompt
