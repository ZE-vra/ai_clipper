# AI Clipper

**Turn a long-form video into ready-to-publish short-form clips—with captions, titles, metadata, and custom thumbnails.**

AI Clipper is a checkpointed Python pipeline for extracting engaging moments from YouTube videos or local video files. It combines AI-assisted editorial decisions with deterministic media processing, validation, and resumable execution.

## Highlights

- **End-to-end CLI:** provide a YouTube URL or local video and run the pipeline.
- **Transcript-driven discovery:** Whisper transcription feeds candidate generation and clip evaluation.
- **Editorial selection:** Gemini evaluates candidate moments; a deterministic planner selects the final clips.
- **Vertical video rendering:** FFmpeg-based rendering and caption burning for short-form platforms.
- **Publishing package:** generates per-clip titles, hooks, thumbnail copy, and content angles.
- **Thumbnail V2:** composes a thumbnail for each clip using its saved package and local frame analysis.
- **Resumable by design:** persists stage artifacts and reuses valid checkpoints to avoid repeating expensive work.
- **Graceful partial failure:** a thumbnail failure does not discard an otherwise successful rendered clip.

## Pipeline

```text
YouTube URL or local video
          |
          v
     Ingestion ──> Audio extraction
          |                |
          |             Whisper
          |                |
          v                v
    Workspace         Transcript
                           |
                           v
                Candidate generation
                           |
                           v
                 Gemini evaluation
                           |
                           v
                 Deterministic planner
                           |
                           v
                 Clip acquisition
                           |
                           v
                  FFmpeg rendering
                           |
                           v
              Vertical edit + captions
                           |
                           v
                 Gemini packaging
                           |
                           v
                 Thumbnail V2 (local)
                           |
                           v
          Clips + metadata + thumbnails
```

The thumbnail stage reuses each clip's saved packaging and makes **no additional Gemini request**. Thumbnail work is checkpointed and reused when valid.

## Quick start

### Requirements

- Python 3.10+ (use the Python version supported by your installed dependencies)
- FFmpeg and FFprobe available on `PATH`
- A Gemini API key
- Internet access for YouTube sources
- Whisper model weights on first transcription (downloaded by Whisper as needed)

### Install

Windows PowerShell:

```powershell
git clone https://github.com/ZE-vra/ai_clipper.git
cd ai_clipper
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file in the repository root:

```dotenv
GEMINI_API_KEY=your_gemini_api_key
```

### Run the full pipeline

```powershell
python cli.py "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
```

You can also pass a local video path:

```powershell
python cli.py "C:\\path\\to\\video.mp4"
```

The CLI prints the selected clips and their output paths. Project artifacts are stored under `projects/`; this generated workspace is intentionally excluded from version control.

## Regenerate one thumbnail

Once the pipeline has created the project's source-section and packaging checkpoints, regenerate a thumbnail without rerunning transcription, clip selection, packaging, or video rendering:

```powershell
python -m src.thumbnails.project_cli "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --clip-id 1
```

Useful options:

- `--clip-id N`: choose the clip.
- `--samples 15`: sample more frames for discovery.
- `--output path/to/preview.jpg`: save a separate preview rather than replacing the checkpoint.

The command requires the saved artifacts from the original run. It does not silently start the full pipeline if those artifacts are missing.

## Thumbnail V2 standalone tools

For experiments independent of the main clipper, the V2 CLI accepts a structured request:

```powershell
python -m src.thumbnails.cli path/to/input.mp4 --input examples/thumbnail_v2_request.json --output output/thumbnail.jpg
```

It also supports transcript input or automatic transcription:

```powershell
python -m src.thumbnails.cli path/to/input.mp4 --transcript path/to/transcript.txt --output output/thumbnail.jpg
python -m src.thumbnails.cli path/to/input.mp4 --auto-transcribe --output output/thumbnail.jpg
```

Transcript-driven modes require `GEMINI_API_KEY` and the Google GenAI SDK. Automatic transcription requires Whisper and FFmpeg. The optional subject-segmentation checkpoint must be supplied locally with `--subject-model`; the CLI does not silently download YOLO weights.

## Tests

Run the thumbnail test suite:

```powershell
python -m pytest tests/thumbnails -v
```

Run the broader test suite:

```powershell
python -m pytest -v
```

GitHub Actions runs focused regression tests for the thumbnail subsystem and related pipeline integration.

## Repository layout

- `src/ingestion/` — source ingestion and audio extraction
- `src/perception/` — transcription
- `src/candidates/` — candidate clip-window generation
- `src/intelligence/` — AI-assisted candidate evaluation
- `src/planning/` — deterministic clip selection
- `src/editing/` — edit planning and rendering
- `src/packaging/` — publishing metadata and persistence
- `src/thumbnails/` — Thumbnail V2 domain, intelligence, layout, rendering, evaluation, and pipeline integration
- `tests/` — unit and integration tests
- `examples/` — example inputs for the standalone thumbnail workflow

## Design principles

1. Use AI for editorial judgment, not routine media execution.
2. Keep rendering, geometry, validation, and persistence deterministic.
3. Persist checkpoints so interrupted runs can resume safely.
4. Keep providers replaceable and responsibilities modular.
5. Prefer explicit failure and recovery behavior over silent data loss.

## Current limitations

- AI-generated editorial decisions and publishing copy still require human review before publishing.
- Output quality depends on source audio/video quality, transcript accuracy, and the chosen moments.
- Thumbnail subject segmentation is optional and requires a compatible local model checkpoint.
- Platform performance cannot be guaranteed by the pipeline; validate results with real audience data.
