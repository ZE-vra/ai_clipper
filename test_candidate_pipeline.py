from src.pipeline.orchestrator import PipelineOrchestrator


SOURCE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


orchestrator = PipelineOrchestrator()

workspace, source, transcript, candidates = (
    orchestrator.prepare_candidates(
        source_location=SOURCE_URL,
        whisper_model="base",
    )
)

print()
print("=== DISCOVERY COMPLETE ===")
print(f"Project:    {workspace.project_id}")
print(f"Title:      {source.title}")
print(f"Language:   {transcript.language}")
print(f"Duration:   {transcript.duration:.2f}s")
print(f"Segments:   {len(transcript.segments)}")
print(f"Candidates: {candidates.total_candidates}")
print()
print(f"Transcript: {workspace.transcript_dir / 'transcript.json'}")
print(f"Candidates: {workspace.candidates_dir / 'candidates.json'}")