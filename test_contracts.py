# test_contracts.py
import json
from src.schemas import VideoSource, TranscriptSegment, CandidateWindow, CandidateManifest

source = VideoSource(source_type="youtube", location="https://youtube.com/watch?v=demo", title="Demo Video")
seg1 = TranscriptSegment(id=0, start=0.0, end=5.2, text="Hello world.")
seg2 = TranscriptSegment(id=1, start=5.2, end=12.4, text="This is a test segment.")

cand = CandidateWindow(
    candidate_id=1,
    start_time=seg1.start,
    end_time=seg2.end,
    transcript_text="Hello world. This is a test segment.",
    segments=[seg1, seg2]
)

manifest = CandidateManifest(source=source, total_candidates=1, candidates=[cand])

# Check JSON conversion
json_output = manifest.to_json()
print(json_output)

# Assert property calculations hold
assert cand.duration == 12.4
print("\n✅ Data contracts & serialization verified successfully!")