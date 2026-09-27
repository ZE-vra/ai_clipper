from src.candidates.window_generator import generate_candidate_windows
from src.config import ProjectWorkspace
from src.schemas import (
    Transcript,
    TranscriptSegment,
    VideoSource,
)


def test_candidate_generation(tmp_path):
    workspace = ProjectWorkspace(
        project_id="candidate_test",
        root_dir=tmp_path,
        source_dir=tmp_path / "source",
        audio_dir=tmp_path / "audio",
        transcript_dir=tmp_path / "transcript",
        candidates_dir=tmp_path / "candidates",
        evaluations_dir=tmp_path / "evaluations",
        clips_dir=tmp_path / "clips",
        packaging_dir=tmp_path / "packaging",
        logs_dir=tmp_path / "logs",
    )

    workspace.initialize()

    source = VideoSource(
        source_type="local",
        location="test_source.mp4",
        title="Candidate Generation Test",
    )

    segments = []

    start = 0.0

    for index in range(10):
        end = start + 8.0

        segments.append(
            TranscriptSegment(
                id=index + 1,
                start=start,
                end=end,
                text=f"Test transcript segment {index + 1}.",
            )
        )

        start = end + 1.0

    transcript = Transcript(
        source=source,
        duration=80.0,
        language="en",
        segments=segments,
    )

    manifest = generate_candidate_windows(
        transcript=transcript,
        workspace=workspace,
    )

    assert manifest.total_candidates > 0
    assert len(manifest.candidates) > 0
    assert (
        workspace.candidates_dir / "candidates.json"
    ).exists()

    for candidate in manifest.candidates:
        assert candidate.start_time >= 0
        assert candidate.end_time > candidate.start_time
        assert candidate.duration >= 30.0
        assert candidate.duration <= 90.0
        assert candidate.transcript_text.strip()