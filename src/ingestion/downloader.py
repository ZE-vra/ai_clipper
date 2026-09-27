"""Module 1: Ingestion

Handles acquiring audio from both YouTube URLs and local video files,
saving extracted audio to the project's workspace audio directory.
"""

from pathlib import Path
import subprocess
from typing import Tuple
import yt_dlp

from src.config import ProjectWorkspace
from src.exceptions import IngestionError, NetworkError, VideoUnavailableError
from src.schemas import VideoSource


def _is_youtube_url(location: str) -> bool:
    """Checks if input string is a YouTube URL."""
    return "youtube.com" in location.lower() or "youtu.be" in location.lower()


def _download_youtube_audio(url: str, output_dir: Path) -> Tuple[Path, str]:
    """Downloads audio stream from YouTube using yt-dlp Python API and converts to MP3."""
    out_file = output_dir / "audio.mp3"
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': str(output_dir / 'audio.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_title = info.get('title', 'YouTube Video') if info else 'YouTube Video'
    except yt_dlp.utils.DownloadError as e:
        err_msg = str(e).lower()
        if "private" in err_msg or "unavailable" in err_msg or "removed" in err_msg:
            raise VideoUnavailableError(f"Video is unavailable or private: {url}") from e
        elif "network" in err_msg or "unable to download" in err_msg:
            raise NetworkError(f"Network error downloading video: {url}") from e
        else:
            raise IngestionError(f"yt-dlp failed to download audio: {e}") from e
    except Exception as e:
        raise IngestionError(f"Unexpected error during audio download: {e}") from e

    if not out_file.exists():
        mp3_files = list(output_dir.glob("*.mp3"))
        if mp3_files:
            return mp3_files[0], video_title
        raise IngestionError(f"Expected audio file missing at {out_file}")

    return out_file, video_title


def _extract_local_audio(file_path: Path, output_dir: Path) -> Tuple[Path, str]:
    """Extracts MP3 audio stream from a local video/audio file via FFmpeg."""
    if not file_path.exists():
        raise IngestionError(f"Local file does not exist: {file_path}")

    audio_file = output_dir / "audio.mp3"
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(file_path),
        "-vn",
        "-acodec", "libmp3lame",
        "-q:a", "2",
        str(audio_file),
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise IngestionError(f"FFmpeg failed to extract audio from {file_path}: {e.stderr}") from e

    return audio_file, file_path.stem


def ingest_audio(source_location: str, workspace: ProjectWorkspace) -> Tuple[VideoSource, Path]:
    """Ingests source media (YouTube URL or local file path) and extracts audio."""
    if _is_youtube_url(source_location):
        audio_path, title = _download_youtube_audio(source_location, workspace.audio_dir)
        source = VideoSource(source_type="youtube", location=source_location, title=title)
    else:
        local_path = Path(source_location).resolve()
        audio_path, title = _extract_local_audio(local_path, workspace.audio_dir)
        source = VideoSource(source_type="local", location=str(local_path), title=title)

    return source, audio_path