# test_candidates.py
from src.candidates.window_generator import generate_candidate_windows
from src.config import Config
from src.ingestion.downloader import ingest_audio
from src.perception.whisper_engine import transcribe_audio

# 1. Workspace Lifecycle
workspace = Config.create_workspace("candidate_discovery_test")
workspace.initialize()
print(f"Project directory: {workspace.root_dir}")

# 2. Ingestion
test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
source, audio_path = ingest_audio(test_url, workspace)

# 3. Perception (Whisper)
transcript = transcribe_audio(
    audio_path=audio_path,
    source=source,
    workspace=workspace,
    model_name="base",
)

# 4. Candidate Discovery
print("\nDiscovering natural candidate windows...")
candidate_manifest = generate_candidate_windows(
    transcript=transcript,
    workspace=workspace,
    min_seconds=30.0,
    max_seconds=90.0,
)

print("\n✅ Candidate Discovery Successful!")
print(f"Total Discovered Candidates: {candidate_manifest.total_candidates}")

# Display first 2 candidates
for cand in candidate_manifest.candidates[:2]:
    print(f"\n--- Candidate #{cand.candidate_id} ---")
    print(f"Time Range: {cand.start_time:.1f}s -> {cand.end_time:.1f}s (Duration: {cand.duration:.1f}s)")
    print(f"Text Snippet: {cand.transcript_text[:120]}...")

candidates_file = workspace.candidates_dir / "candidates.json"
print(f"\nManifest saved to: {candidates_file}")