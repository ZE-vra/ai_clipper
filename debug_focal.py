from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.thumbnails.perception.focal_analyzer import FocalRegionAnalyzer


INPUT_DIR = Path("debug/quality")
OUTPUT_DIR = Path("debug/focal")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

analyzer = FocalRegionAnalyzer()

for path in sorted(INPUT_DIR.glob("frame_*.jpg")):
    with Image.open(path) as image:
        frame = image.convert("RGB")

    evidence = analyzer.analyze(path)

    draw = ImageDraw.Draw(frame)

    width, height = frame.size

    for region in evidence.regions:
        box = region.bounds

        left = int(box.left * width)
        top = int(box.top * height)
        right = int(box.right * width)
        bottom = int(box.bottom * height)

        draw.rectangle(
            (left, top, right, bottom),
            outline="red",
            width=6,
        )

        label = (
            f"{region.region_id} "
            f"s={region.strength:.2f}"
        )

        text_y = max(
            5,
            top - 25,
        )

        draw.text(
            (left + 5, text_y),
            label,
            fill="red",
        )

        focal_x = int(
            region.focal_point.x * width
        )

        focal_y = int(
            region.focal_point.y * height
        )

        radius = 8

        draw.ellipse(
            (
                focal_x - radius,
                focal_y - radius,
                focal_x + radius,
                focal_y + radius,
            ),
            fill="red",
        )

    output = (
        OUTPUT_DIR
        / f"{path.stem}_regions.jpg"
    )

    frame.save(
        output,
        quality=95,
    )

    print(
        f"saved {output} "
        f"({len(evidence.regions)} regions)"
    )
