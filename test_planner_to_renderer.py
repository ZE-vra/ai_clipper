from pathlib import Path

from src.pipeline.orchestrator import PipelineOrchestrator
from src.planning.clip_planner import ClipPlanner
from src.schemas import (
    CandidateManifest,
    CandidateWindow,
    CandidateEvaluation,
    EvaluationManifest,
    TranscriptSegment,
    VideoSource,
)


SOURCE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


# ---------------------------------------------------------
# 1. Create a fake candidate manifest
# ---------------------------------------------------------

source = VideoSource(
    source_type="youtube",
    location=SOURCE_URL,
    title="Planner to Renderer Test",
)


candidate = CandidateWindow(
    candidate_id=1,
    start_time=130.00,
    end_time=150.00,
    transcript_text="This is a test candidate.",
    segments=[
        TranscriptSegment(
            id=1,
            start=130.00,
            end=150.00,
            text="This is a test candidate.",
        )
    ],
)


candidate_manifest = CandidateManifest(
    source=source,
    total_candidates=1,
    candidates=[candidate],
)


# ---------------------------------------------------------
# 2. Create a fake AI evaluation
# ---------------------------------------------------------

evaluation = CandidateEvaluation(
    candidate_id=1,
    score=9.0,
    reason="Strong test candidate.",
    suggested_title="Planner Test Clip",
)


evaluation_manifest = EvaluationManifest(
    source=source,
    evaluations=[evaluation],
)


# ---------------------------------------------------------
# 3. Let the REAL ClipPlanner make the decision
# ---------------------------------------------------------

planner = ClipPlanner()

clip_manifest = planner.plan(
    candidate_manifest=candidate_manifest,
    evaluation_manifest=evaluation_manifest,
)


print(f"Planner selected {len(clip_manifest.selected_clips)} clip(s).")


# ---------------------------------------------------------
# 4. Let the REAL orchestrator render the decision
# ---------------------------------------------------------

orchestrator = PipelineOrchestrator()

output_dir = Path("output") / "planner_to_renderer_test"

rendered_clips = orchestrator.render_clips(
    clip_manifest=clip_manifest,
    output_dir=output_dir,
)


# ---------------------------------------------------------
# 5. Verify the result
# ---------------------------------------------------------

for clip in rendered_clips:
    print(f"Created: {clip.file_path}")
    print(f"Title:   {clip.title}")
    print(f"Exists:  {clip.exists}")