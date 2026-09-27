"""Run the real discovery pipeline against a YouTube video."""

from src.candidates.window_generator import generate_candidate_windows
from src.config import Config
from src.ingestion.downloader import ingest_audio
from src.persistence.manifests import (
    load_candidate_manifest,
    load_source,
    load_transcript,
    save_source,
)
from src.perception.whisper_engine import transcribe_audio


VIDEO_URL = (
    "https://youtu.be/vp5sSqyZ5Go?si=Q5urbvUO-p64nCnD"
)


def main() -> None:
    print("=== REAL DISCOVERY PIPELINE TEST ===")
    print()

    workspace, existed = Config.get_or_create_workspace(
        VIDEO_URL
    )

    print(f"Project: {workspace.project_id}")
    print(f"Existing project: {existed}")
    print()

    # ---------------------------------------------------------
    # 1. INGESTION
    # ---------------------------------------------------------

    print("[1/3] REAL INGESTION")
    print("-" * 40)

    audio_path = workspace.audio_dir / "audio.mp3"

    if audio_path.exists():
        print("✓ Existing audio found:")
        print(f"  {audio_path}")

        source_path = (
            workspace.source_dir / "source.json"
        )

        if source_path.exists():
            source = load_source(workspace)
        else:
            source, _ = ingest_audio(
                source_location=VIDEO_URL,
                workspace=workspace,
            )
            save_source(
                source,
                workspace,
            )

    else:
        source, audio_path = ingest_audio(
            source_location=VIDEO_URL,
            workspace=workspace,
        )

        save_source(
            source,
            workspace,
        )

        print("✓ Audio downloaded:")
        print(f"  {audio_path}")

    print(f"Title: {source.title}")
    print()

    # ---------------------------------------------------------
    # 2. WHISPER
    # ---------------------------------------------------------

    print("[2/3] REAL WHISPER")
    print("-" * 40)

    transcript_path = (
        workspace.transcript_dir
        / "transcript.json"
    )

    if transcript_path.exists():
        transcript = load_transcript(workspace)

        print("✓ Existing transcript reused.")

    else:
        transcript = transcribe_audio(
            audio_path=audio_path,
            source=source,
            workspace=workspace,
            model_name="base",
        )

        print("✓ Whisper transcription completed.")

    print(f"Language: {transcript.language}")
    print(f"Duration: {transcript.duration:.2f}s")
    print(f"Segments: {len(transcript.segments)}")
    print()

    # ---------------------------------------------------------
    # 3. CANDIDATE GENERATION
    # ---------------------------------------------------------

    print("[3/3] REAL CANDIDATE GENERATION")
    print("-" * 40)

    candidates_path = (
        workspace.candidates_dir
        / "candidates.json"
    )

    if candidates_path.exists():
        candidate_manifest = (
            load_candidate_manifest(
                workspace
            )
        )

        print("✓ Existing candidates reused.")

    else:
        candidate_manifest = (
            generate_candidate_windows(
                transcript=transcript,
                workspace=workspace,
            )
        )

        print(
            "✓ Candidate generation completed."
        )

    print(
        f"Candidates: "
        f"{len(candidate_manifest.candidates)}"
    )

    print()

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------

    print("=" * 60)
    print("DISCOVERY RESULTS")
    print("=" * 60)

    for candidate in candidate_manifest.candidates:
        print()

        print(
            f"Candidate "
            f"{candidate.candidate_id:02d}"
        )

        print(
            f"Time: "
            f"{candidate.start_time:.2f}s"
            f" -> "
            f"{candidate.end_time:.2f}s"
        )

        print(
            f"Duration: "
            f"{candidate.duration:.2f}s"
        )

        print("Text:")
        print(
            f"  {candidate.transcript_text}"
        )

    print()
    print("=" * 60)
    print("REAL DISCOVERY TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()