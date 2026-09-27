import json

path = r".\projects\youtube_vp5sSqyZ5Go\transcript\transcript.json"

with open(path, encoding="utf-8") as file:
    data = json.load(file)

for segment in data["segments"]:
    if segment["start"] >= 486:
        print(
            f'{segment["start"]:.2f} -> '
            f'{segment["end"]:.2f}: '
            f'{segment["text"]}'
        )