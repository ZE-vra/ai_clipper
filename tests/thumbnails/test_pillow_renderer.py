from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.thumbnails.domain.assets import (
    AssetProvenance,
    VisualAsset,
)
from src.thumbnails.domain.geometry import (
    BoundingBox,
    Point,
)
from src.thumbnails.domain.plans import (
    CompositionPlan,
    ThumbnailRenderPlan,
    TypographyBlockPlan,
    TypographyPlan,
    VisualPlacement,
    VisualTreatmentPlan,
)
from src.thumbnails.rendering.pillow_renderer import (
    InMemoryVisualAssetResolver,
    PillowThumbnailRenderer,
)


def _make_source(path: Path) -> None:
    image = Image.new("RGB", (1600, 900), "white")
    pixels = image.load()

    for y in range(900):
        for x in range(800):
            pixels[x, y] = (255, 0, 0)

    for y in range(900):
        for x in range(800, 1600):
            pixels[x, y] = (0, 0, 255)

    image.save(path)


def _make_asset(path: Path) -> VisualAsset:
    return VisualAsset(
        asset_id="source-frame",
        provenance=AssetProvenance.SOURCE_FRAME,
        path=str(path),
    )


def _make_plan(
    asset: VisualAsset,
) -> ThumbnailRenderPlan:
    placement = VisualPlacement(
        asset_id=asset.asset_id,
        focal_point=Point(0.5, 0.5),
        crop_bounds=BoundingBox(
            left=0.25,
            top=0.0,
            right=0.75,
            bottom=1.0,
        ),
    )

    composition = CompositionPlan(
        visual_placements=(placement,),
        negative_space_regions=(
            BoundingBox(
                left=0.05,
                top=0.05,
                right=0.95,
                bottom=0.35,
            ),
        ),
    )

    typography_block = TypographyBlockPlan(
        copy=object(),
        font_name="Arial",
        font_size=72,
        weight="bold",
        alignment="center",
        color="#FFFFFF",
        position=Point(0.5, 0.18),
        text_bounds=BoundingBox(
            left=0.10,
            top=0.05,
            right=0.90,
            bottom=0.30,
        ),
        max_lines=2,
        line_spacing=0.92,
        rendered_text="THIS JET COST\n$500,000",
        stroke_color="#000000",
        stroke_width=4,
    )

    return ThumbnailRenderPlan(
        canvas_width=1080,
        canvas_height=1920,
        composition=composition,
        typography=TypographyPlan(
            blocks=(typography_block,),
        ),
        visual_treatment=VisualTreatmentPlan(),
    )


def _make_renderer(
    asset: VisualAsset,
) -> PillowThumbnailRenderer:
    resolver = InMemoryVisualAssetResolver(
        {asset.asset_id: asset}
    )

    return PillowThumbnailRenderer(
        asset_resolver=resolver,
    )


def test_renderer_creates_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jpg"
    output = tmp_path / "thumbnail.jpg"

    _make_source(source)
    asset = _make_asset(source)

    renderer = _make_renderer(asset)

    result = renderer.render(
        plan=_make_plan(asset),
        output_path=output,
    )

    assert result == output
    assert output.exists()

    with Image.open(output) as image:
        assert image.size == (1080, 1920)
        assert image.mode == "RGB"


def test_renderer_honors_crop_plan(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jpg"
    output = tmp_path / "thumbnail.jpg"

    _make_source(source)
    asset = _make_asset(source)

    renderer = _make_renderer(asset)

    renderer.render(
        plan=_make_plan(asset),
        output_path=output,
    )

    with Image.open(output) as image:
        center = image.getpixel((540, 960))

        # The source is red on the left half and blue on the
        # right half. The planned crop is the middle 50%,
        # therefore the crop contains the boundary between
        # both regions.
        assert center[0] > 100
        assert center[2] > 100


def test_renderer_renders_planned_text(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jpg"
    output = tmp_path / "thumbnail.jpg"

    _make_source(source)
    asset = _make_asset(source)

    renderer = _make_renderer(asset)

    renderer.render(
        plan=_make_plan(asset),
        output_path=output,
    )

    with Image.open(output) as image:
        # Inspect the planned text area. The source has only
        # red/blue pixels, so white text with a black stroke
        # introduces pixels that could not exist in the source.
        pixels = image.load()

        found_text_pixel = False

        for y in range(100, 575, 10):
            for x in range(100, 980, 10):
                r, g, b = pixels[x, y]

                if (
                    r > 220
                    and g > 220
                    and b > 220
                ):
                    found_text_pixel = True
                    break

            if found_text_pixel:
                break

        assert found_text_pixel


def test_renderer_rejects_missing_asset(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jpg"
    output = tmp_path / "thumbnail.jpg"

    _make_source(source)
    asset = _make_asset(source)

    resolver = InMemoryVisualAssetResolver({})

    renderer = PillowThumbnailRenderer(
        asset_resolver=resolver,
    )

    try:
        renderer.render(
            plan=_make_plan(asset),
            output_path=output,
        )
    except ValueError as exc:
        assert "source-frame" in str(exc)
    else:
        raise AssertionError(
            "expected missing asset to raise ValueError"
        )


def test_renderer_rejects_missing_source_file(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.jpg"
    asset = _make_asset(missing)

    renderer = _make_renderer(asset)

    try:
        renderer.render(
            plan=_make_plan(asset),
            output_path=tmp_path / "thumbnail.jpg",
        )
    except FileNotFoundError as exc:
        assert "missing.jpg" in str(exc)
    else:
        raise AssertionError(
            "expected missing source to raise FileNotFoundError"
        )


def test_renderer_applies_visual_treatment(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jpg"
    output = tmp_path / "thumbnail.jpg"

    _make_source(source)
    asset = _make_asset(source)

    base_plan = _make_plan(asset)

    plan = ThumbnailRenderPlan(
        canvas_width=base_plan.canvas_width,
        canvas_height=base_plan.canvas_height,
        composition=base_plan.composition,
        typography=base_plan.typography,
        visual_treatment=VisualTreatmentPlan(
            contrast=1.4,
            saturation=0.5,
            sharpness=1.2,
            vignette=0.35,
            overlay_opacity=0.15,
        ),
    )

    renderer = _make_renderer(asset)

    renderer.render(
        plan=plan,
        output_path=output,
    )

    assert output.exists()

    with Image.open(output) as image:
        assert image.size == (1080, 1920)
        assert image.mode == "RGB"


def test_renderer_composites_foreground_asset_above_background(tmp_path: Path) -> None:
    background_path = tmp_path / "background.png"
    foreground_path = tmp_path / "foreground.png"
    output = tmp_path / "thumbnail.jpg"
    Image.new("RGB", (1600, 900), (220, 20, 20)).save(background_path)
    foreground_image = Image.new("RGBA", (1600, 900), (0, 0, 0, 0))
    foreground_pixels = foreground_image.load()
    for y in range(100, 800):
        for x in range(600, 1000):
            foreground_pixels[x, y] = (20, 40, 230, 255)
    foreground_image.save(foreground_path)

    background = _make_asset(background_path)
    foreground = VisualAsset(
        asset_id="subject-cutout",
        provenance=AssetProvenance.SOURCE_FRAME,
        path=str(foreground_path),
    )
    base_plan = _make_plan(background)
    plan = ThumbnailRenderPlan(
        canvas_width=base_plan.canvas_width,
        canvas_height=base_plan.canvas_height,
        composition=base_plan.composition,
        typography=TypographyPlan(blocks=()),
        visual_treatment=VisualTreatmentPlan(),
        foreground_asset_id=foreground.asset_id,
    )
    renderer = PillowThumbnailRenderer(
        asset_resolver=InMemoryVisualAssetResolver(
            {background.asset_id: background, foreground.asset_id: foreground}
        )
    )

    renderer.render(plan=plan, output_path=output)

    with Image.open(output) as image:
        center = image.getpixel((540, 960))
        corner = image.getpixel((50, 50))
        assert center[2] > center[0]  # foreground subject is visible
        assert corner[0] > corner[2]  # transparent pixels preserve background
