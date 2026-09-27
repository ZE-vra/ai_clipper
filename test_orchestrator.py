from pathlib import Path

from src.pipeline.orchestrator import PipelineOrchestrator
from src.schemas import (
    ClipDecision,
    ClipManifest,
    VideoSource,
)


SOURCE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


source = VideoSource(
    source_type="youtube",
    location=SOURCE_URL,
    title="Orchestrator Test",
)


decision = ClipDecision(
    clip_id=1,
    candidate_id=1,
    snapped_start_time=136.60,
    snapped_end_time=146.60,
    final_score=9.0,
    reason="Test clip for the rendering pipeline.",
    title="Orchestrator Test Clip",
)


manifest = ClipManifest(
    source=source,
    selected_clips=[decision],
)


orchestrator = PipelineOrchestrator()


output_dir = Path("output") / "orchestrator_test"


rendered = orchestrator.render_clips(
    clip_manifest=manifest,
    output_dir=output_dir,
)


for clip in rendered:
    print(f"Created: {clip.file_path}")
    print(f"Title:   {clip.title}")
    print(f"Exists:  {clip.exists}")