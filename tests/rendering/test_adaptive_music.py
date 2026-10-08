from pathlib import Path

from src.editing.models import (
    BackgroundPlan,
    CanvasPlan,
    CaptionPlan,
    CompositionPlan,
    EditingPlan,
    ForegroundPlan,
    RenderConfig,
)
from src.editing.rendering.editing_renderer import EditingRenderer
from src.rendering.music.models import MusicPlan, MusicTrack
from src.rendering.music.selector import MusicCatalog, MusicSelector


def make_plan(music: MusicPlan, duration: float = 30.0) -> EditingPlan:
    return EditingPlan(
        clip_id="clip_01",
        composition=CompositionPlan(
            canvas=CanvasPlan(width=1080, height=1920),
            background=BackgroundPlan(source="same_video"),
            foreground=ForegroundPlan(),
        ),
        captions=CaptionPlan(enabled=False),
        render_config=RenderConfig(),
        audio=music,
        clip_duration=duration,
    )


def test_music_catalog_discovers_tracks_by_mood_folder(tmp_path):
    tension = tmp_path / "tension"
    tension.mkdir()
    track = tension / "founder_crisis.mp3"
    track.write_bytes(b"fake")

    tracks = MusicCatalog(tmp_path).tracks()

    assert len(tracks) == 1
    assert tracks[0].track_id == "founder_crisis"
    assert tracks[0].moods == ("tension",)


def test_music_selector_prefers_matching_mood(tmp_path):
    for mood in ("tension", "motivational"):
        directory = tmp_path / mood
        directory.mkdir()
        (directory / f"{mood}_01.mp3").write_bytes(b"fake")

    selector = MusicSelector(MusicCatalog(tmp_path))
    selected = selector.select(
        title="We Nearly Lost Everything",
        reason="The founder explains how the company nearly collapsed.",
    )

    assert selected is not None
    assert selected.moods == ("tension",)


def test_audio_filter_without_music_preserves_voice_mastering():
    plan = make_plan(MusicPlan(enabled=False))

    audio_filter = EditingRenderer._build_audio_filter(plan)

    assert "[0:a]" in audio_filter
    assert "acompressor=" in audio_filter
    assert "loudnorm=" in audio_filter
    assert "sidechaincompress" not in audio_filter


def test_audio_filter_with_music_adds_ducking_and_mix(tmp_path):
    track_path = tmp_path / "motivational.mp3"
    track_path.write_bytes(b"fake")
    track = MusicTrack(
        track_id="motivational",
        path=track_path,
        moods=("motivational",),
    )
    plan = make_plan(MusicPlan(enabled=True, track=track))

    audio_filter = EditingRenderer._build_audio_filter(plan)

    assert "sidechaincompress=" in audio_filter
    assert "threshold=0.06" in audio_filter
    assert "ratio=8.0" in audio_filter
    assert "attack=20.0" in audio_filter
    assert "release=300.0" in audio_filter
    assert "amix=inputs=2:duration=first" in audio_filter
    assert "afade=t=in" in audio_filter
    assert "afade=t=out" in audio_filter
    assert "loudnorm=I=-14:TP=-1.5:LRA=11" in audio_filter


def test_renderer_adds_music_input_and_maps_final_audio(tmp_path):
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    music_path = tmp_path / "music.mp3"
    source.write_bytes(b"fake source")
    music_path.write_bytes(b"fake music")

    plan = make_plan(
        MusicPlan(
            enabled=True,
            track=MusicTrack("music", music_path, ("neutral",)),
        )
    )
    command, caption_file = EditingRenderer()._build_command(
        source_path=source,
        output_path=output,
        plan=plan,
    )

    assert caption_file is None
    assert "-stream_loop" in command
    assert "-1" in command
    assert str(music_path) in command
    assert "[final_audio]" in command[command.index("-filter_complex") + 1]
    assert command[command.index("-map", command.index("[final]")) + 1] == "[final_audio]"
