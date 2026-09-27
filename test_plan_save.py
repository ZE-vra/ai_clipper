from src.schemas import (
    VideoSource,
    ClipDecision,
    ClipManifest,
)


source = VideoSource(
    source_type="youtube",
    location="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    title="Test Video",
)

clips = [
    ClipDecision(
        clip_id=1,
        candidate_id=69,
        snapped_start_time=136.60,
        snapped_end_time=177.96,
        final_score=9.5,
        reason="Strong standalone moment.",
        title="We Know the Game",
    ),
    ClipDecision(
        clip_id=2,
        candidate_id=27,
        snapped_start_time=64.76,
        snapped_end_time=101.72,
        final_score=8.5,
        reason="Strong buildup and payoff.",
        title="Never Gonna Let You Down",
    ),
]

manifest = ClipManifest(
    source=source,
    selected_clips=clips,
)

plan_file = "clip_plan.json"

with open(plan_file, "w", encoding="utf-8") as f:
    f.write(manifest.to_json())

print(f"✅ Clip plan saved to: {plan_file}")