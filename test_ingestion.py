from src.config import Config
from src.ingestion.downloader import ingest_audio

# Create workspace definition and initialize filesystem
workspace = Config.create_workspace("yt_test")
workspace.initialize()

print(f"Created workspace at: {workspace.root_dir}")

# Test downloading a video audio stream
test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
source, audio_path = ingest_audio(test_url, workspace)

print("\n✅ Ingestion Successful!")
print(f"Source Type:  {source.source_type}")
print(f"Title:        {source.title}")
print(f"Audio Path:   {audio_path}")
print(f"Audio Exists: {audio_path.exists()}")