from pathlib import Path

import pytest

from src.thumbnails.perception.video_frame_sampler import (
    FFprobeVideoDurationReader,
    ThumbnailFrameSampler,
)


def test_duration_reader_rejects_missing_video(tmp_path: Path) -> None:
    reader = FFprobeVideoDurationReader()

    with pytest.raises(FileNotFoundError):
        reader.read(tmp_path / "missing.mp4")


def test_sampler_creates_expected_number_of_samples(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")

    calls: list[float] = []

    class FakeExtractor:
        def extract(self, *, source_path, timestamp, output_path):
            calls.append(timestamp)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"frame")
            return output_path

    sampler = ThumbnailFrameSampler(
        extractor=FakeExtractor(),
        duration_reader=lambda _: 100.0,
        sample_count=5,
    )

    samples = sampler.sample(
        video_path=source,
        output_dir=tmp_path / "frames",
    )

    assert len(samples) == 5
    assert len(calls) == 5
    assert samples[0].timestamp == pytest.approx(8.0)
    assert samples[-1].timestamp == pytest.approx(92.0)
