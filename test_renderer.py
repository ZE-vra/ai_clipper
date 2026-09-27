from pathlib import Path

from src.rendering.ffmpeg_renderer import FFmpegRenderer


renderer = FFmpegRenderer()

output = renderer.render_clip(
    source_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    output_path=Path("test_renderer_clip_2.mp4"),
    start_time=136.60,
    end_time=146.60,
)

print(f"\nCreated: {output}")