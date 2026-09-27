import json

path = r".\projects\2026-09-26_17-20-29_outube_com_watch_v_dQw4w9WgXcQ\candidates\candidates.json"

with open(path, encoding="utf-8") as f:
    data = json.load(f)

print(f"Total candidates: {len(data['candidates'])}")
print()

for candidate in data["candidates"]:
    print(
        f"{candidate['candidate_id']}: "
        f"{candidate['start_time']:.1f}s -> "
        f"{candidate['end_time']:.1f}s | "
        f"{candidate['transcript_text'][:120]}..."
    )