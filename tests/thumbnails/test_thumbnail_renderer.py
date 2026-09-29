from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.thumbnails.models import (
    ThumbnailComposition,
    ThumbnailPlan,
    ThumbnailStyle,
)
from src.thumbnails.thumbnail_renderer import ThumbnailRenderer


def make_plan(
    *,
    text: str = "FLYING FOR $1",
) -> ThumbnailPlan:
    return ThumbnailPlan(
        clip_id=1,
        frame_timestamp=55.0,
        text=text,
        composition=ThumbnailComposition(),
        style=ThumbnailStyle(),
    )


def create_source_frame(
    path: Path,
) -> None:
    image = Image.new(
        "RGB",
        (640, 360),
        "#336699",
    )

    image.save(
        path,
        format="JPEG",
    )


def test_missing_source_frame_is_rejected(
    tmp_path: Path,
) -> None:
    renderer = ThumbnailRenderer()

    with pytest.raises(
        FileNotFoundError,
        match="Source frame does not exist",
    ):
        renderer.render(
            plan=make_plan(),
            source_frame_path=tmp_path / "missing.jpg",
            output_path=tmp_path / "thumbnail.jpg",
        )


def test_renders_jpeg_with_requested_dimensions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jpg"
    output = tmp_path / "thumbnail.jpg"

    create_source_frame(source)

    renderer = ThumbnailRenderer()

    result = renderer.render(
        plan=make_plan(),
        source_frame_path=source,
        output_path=output,
    )

    assert result == output
    assert output.exists()
    assert output.stat().st_size > 0

    with Image.open(output) as image:
        assert image.format == "JPEG"
        assert image.size == (1080, 1920)


def test_output_directory_is_created(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jpg"
    output = (
        tmp_path
        / "nested"
        / "thumbnails"
        / "thumbnail.jpg"
    )

    create_source_frame(source)

    renderer = ThumbnailRenderer()

    renderer.render(
        plan=make_plan(),
        source_frame_path=source,
        output_path=output,
    )

    assert output.exists()


def test_solid_background_is_supported(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jpg"
    output = tmp_path / "thumbnail.jpg"

    create_source_frame(source)

    plan = ThumbnailPlan(
        clip_id=1,
        frame_timestamp=55.0,
        text="FLYING FOR $1",
        composition=ThumbnailComposition(),
        style=ThumbnailStyle(
            background_mode="solid",
            background_color="#112233",
        ),
    )

    renderer = ThumbnailRenderer()

    renderer.render(
        plan=plan,
        source_frame_path=source,
        output_path=output,
    )

    with Image.open(output) as image:
        assert image.format == "JPEG"
        assert image.size == (1080, 1920)


def test_three_line_text_is_rejected(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jpg"
    output = tmp_path / "thumbnail.jpg"

    create_source_frame(source)

    plan = make_plan(
        text=(
            "THIS TEXT SHOULD "
            "NEED MORE THAN TWO LINES "
            "ON THE THUMBNAIL"
        ),
    )

    renderer = ThumbnailRenderer()

    with pytest.raises(
        ValueError,
        match="maximum of two lines",
    ):
        renderer.render(
            plan=plan,
            source_frame_path=source,
            output_path=output,
        )