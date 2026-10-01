from __future__ import annotations

from pathlib import Path

from src.thumbnails.pipeline_v11 import ThumbnailPipelineV11


class ThumbnailPipelineV12(ThumbnailPipelineV11):
    """
    V1.2 thumbnail pipeline.

    V1.2 keeps the proven V1.1 composition pipeline and adds semantic
    instance-segmentation intelligence plus a layered subject foreground.
    """

    def __init__(
        self,
        *,
        ffmpeg_binary: str = "ffmpeg",
        ffprobe_binary: str = "ffprobe",
        yolo_model_path: str | Path = "yolo26n.pt",
        segmentation_model_path: str | Path = "yolo26n-seg.pt",
    ) -> None:
        super().__init__(
            ffmpeg_binary=ffmpeg_binary,
            ffprobe_binary=ffprobe_binary,
            yolo_model_path=yolo_model_path,
            segmentation_model_path=segmentation_model_path,
        )
