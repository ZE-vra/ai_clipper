# test_perception.py
from pathlib import Path
from src.config import Config
from src.ingestion.downloader import ingest_audio
from src.perception.whisper_engine import transcribe_audio

# Setup project run workspace
workspace = Config.create_workspace("whisper_test")
workspace.initialize()

print(f"Project directory: {workspace.root_dir}")

# 1. Ingest Audio
test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
source, audio_path = ingest_audio(test_url, workspace)
print(f"Downloaded audio to: {audio_path}")

# 2. Transcribe Audio
print("\nRunning Whisper transcription...")
transcript = transcribe_audio(
    audio_path=audio_path,
    source=source,
    workspace=workspace,
    model_name="base",  # Fast lightweight model for local testing
)

print("\n✅ Transcription Successful!")
print(f"Language: {transcript.language}")
print(f"Duration: {transcript.duration} seconds")
print(f"Total Segments: {len(transcript.segments)}")

# Print first 3 segments
print("\nFirst 3 Segments:")
for seg in transcript.segments[:3]:
    print(f" [{seg.start:.1f}s -> {seg.end:.1f}s] {seg.text}")

transcript_json_path = workspace.transcript_dir / "transcript.json"
print(f"\nTranscript saved to: {transcript_json_path}")