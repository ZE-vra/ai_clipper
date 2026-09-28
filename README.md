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