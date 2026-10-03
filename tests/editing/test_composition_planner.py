from src.editing.planning.composition_planner import CompositionPlanner


def test_composition_planner_creates_vertical_blurred_background_plan():
    planner = CompositionPlanner()

    plan = planner.create_plan("clip_01")

    assert plan.canvas.width == 1080
    assert plan.canvas.height == 1920

    assert plan.background.source == "same_video"
    assert plan.background.blur_radius == 60.0
    assert plan.background.brightness == 0.58

    assert plan.foreground.preserve_aspect_ratio is True
    assert plan.foreground.scale == 1.12


def test_composition_planner_accepts_custom_configuration():
    planner = CompositionPlanner(
        canvas_width=720,
        canvas_height=1280,
        blur_radius=15.0,
        background_brightness=0.5,
        foreground_scale=0.9,
    )

    plan = planner.create_plan("clip_02")

    assert plan.canvas.width == 720
    assert plan.canvas.height == 1280

    assert plan.background.blur_radius == 15.0
    assert plan.background.brightness == 0.5

    assert plan.foreground.scale == 0.9


def test_composition_planner_rejects_invalid_clip_id():
    planner = CompositionPlanner()

    try:
        planner.create_plan("")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for empty clip_id")


def test_composition_planner_rejects_invalid_background_brightness():
    try:
        CompositionPlanner(
            background_brightness=0.0
        ).create_plan("clip_01")
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for invalid background brightness"
        )


def test_composition_planner_rejects_invalid_foreground_scale():
    planner = CompositionPlanner(foreground_scale=0.0)

    try:
        planner.create_plan("clip_01")
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for invalid foreground scale"
        )