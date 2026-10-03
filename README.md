# AI Clipper

An automated, checkpointed video-clipping pipeline that transforms long-form video into publishable short-form clips.

The system combines deterministic media processing with AI-assisted content analysis while maintaining clear stage boundaries, persisted checkpoints, validation, retry handling, and resumable execution.

## What It Does

Given a video source, the pipeline can:

1. Ingest the source and extract audio
2. Transcribe the audio using Whisper
3. Discover candidate clip windows from the transcript
4. Evaluate candidates using Gemini
5. Select clips through a deterministic planning stage
6. Acquire the required video sections
7. Render base clips with FFmpeg
8. Convert clips into vertical social-video format
9. Generate and burn captions into the video
10. Generate publishing metadata using Gemini
11. Persist the results for resumable execution

The goal is not simply to generate clips, but to build a reliable pipeline that can recover from failures without unnecessarily repeating expensive work.

## Architecture

```text
CLI
 ↓
PipelineOrchestrator
 ↓
Source / Workspace
 ↓
Audio Ingestion
 ↓
Whisper
 ↓
transcript.json
 ↓
Candidate Generation
 ↓
candidates.json
 ↓
GeminiDirector
 ↓
evaluations.json
 ↓
ClipPlanner
 ↓
clip_plan.json
 ↓
Source Acquisition
 ↓
Local Section
 ↓
FFmpegRenderer
 ↓
Base Render
 ↓
EditingPlanner
 ↓
EditingRenderer
 ↓
Final Vertical Clip
 ↓
GeminiPackager
 ↓
Packaging JSON

## Thumbnail V2 (experimental)

Thumbnail V2 is available through a separate CLI entry point so it does not change the existing clip-generation workflow. It currently expects a **local video** and an explicit JSON request containing a creative `brief` and structured `understanding`; it does not automatically generate those inputs.

### Run

```powershell
python -m src.thumbnails.cli path/to/input.mp4 --input examples/thumbnail_v2_request.json --output output/thumbnail.jpg
```

FFmpeg and FFprobe must be installed and available on `PATH`. Use `--ffmpeg` and `--ffprobe` to specify alternate executable paths. The output and working directories are created automatically when needed.

To enable person segmentation and subject-cutout composition, pass a **local** Ultralytics segmentation checkpoint:

```powershell
python -m src.thumbnails.cli path/to/input.mp4 --input examples/thumbnail_v2_request.json --output output/thumbnail.jpg --subject-model path/to/local-segmentation-checkpoint.pt
```

The CLI does not download model weights implicitly. Without `--subject-model`, it can use source-frame and enhanced-frame strategies but makes no semantic subject-detection claims.

The request schema is illustrated in [examples/thumbnail_v2_request.json](examples/thumbnail_v2_request.json). The `brief` and `understanding` sections are required; `target` is optional and defaults to a 1280×720 YouTube target. Times in events are seconds from the start of the source video.

This entry point is experimental and does not replace the existing clipper CLI.
