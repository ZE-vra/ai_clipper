from pathlib import Path

import pytest
import yt_dlp

from src.exceptions import NetworkError
from src.ingestion import downloader


def test_youtube_download_uses_long_timeout_and_retries(monkeypatch, tmp_path):
    captured = {}

    class FakeYoutubeDL:
        def __init__(self, options):
            captured.update(options)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download):
            (tmp_path / "audio.mp3").write_bytes(b"fake audio")
            return {"title": "Example"}

    monkeypatch.setattr(downloader.yt_dlp, "YoutubeDL", FakeYoutubeDL)

    path, title = downloader._download_youtube_audio("https://youtu.be/example", tmp_path)

    assert path == tmp_path / "audio.mp3"
    assert title == "Example"
    assert captured["socket_timeout"] == 60
    assert captured["retries"] >= 5
    assert captured["extractor_retries"] >= 3
    assert captured["fragment_retries"] >= 5
    assert callable(captured["retry_sleep_functions"]["http"])


@pytest.mark.parametrize("message", [
    "Connection timed out",
    "Unable to download webpage: HTTP Error 503",
    "Temporary failure in name resolution",
])
def test_transient_download_failures_are_reported_as_network_errors(monkeypatch, tmp_path, message):
    class FakeYoutubeDL:
        def __init__(self, options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download):
            raise yt_dlp.utils.DownloadError(message)

    monkeypatch.setattr(downloader.yt_dlp, "YoutubeDL", FakeYoutubeDL)

    with pytest.raises(NetworkError, match="automatic retries"):
        downloader._download_youtube_audio("https://youtu.be/example", tmp_path)
