# Skill-MoE

Mixture-of-Experts framework for video understanding with iterative reasoning. A ReAct-style LLM agent (gpt-4o-mini) decomposes video questions into a multi-step chain of specialized skill calls (ASR, OCR, object detection, tracking, etc.), and a local video LLM (Qwen2.5-Omni-7B) generates the final answer grounded in the actual video plus accumulated skill evidence.

## Architecture

The pipeline uses a **ReAct (Reasoning + Acting) loop** rather than a single-pass router. The LLM router reasons step-by-step — producing a Thought, selecting an Action, and observing the Result — before deciding the next move.

```
Question + Video
      |
      v
+---> LLM Router (ReAct Agent) <----+
|         |                          |
|     Thought: reason about          |
|       what info is missing         |
|         |                          |
|     Action: CALL_SKILL(name)       |  History
|         |                          |  (concise
|         v                          |   summaries)
|     Skill Runner executes          |
|     (OCR / Motion / ASR / ...)     |
|         |                          |
|     Observation: skill result      |
|         |                          |
+---------+--------------------------+
          |
          | Action: FINISH
          v
    Video LLM (Qwen2.5-Omni-7B)
      sees: video + question + all skill outputs
          |
          v
      Final answer
```

**Example trace** — "What is the license plate number?"

```
Turn 1  Thought: I need to find where the car is first.
        Action:  CALL_SKILL(object_detect)
        Result:  Detected car at 2.0-8.0s

Turn 2  Thought: Car found. Now I need to read the plate text.
        Action:  CALL_SKILL(ocr)
        Result:  Extracted "京A12345" at 3.0-6.0s

Turn 3  Thought: I have the plate number from OCR. Done.
        Action:  FINISH
```

## Layout

```
skill_moe/
  base.py          # Data classes: SkillRequest, SkillResponse, SkillMetadata,
                   #   RouterDecision, ReasoningStep, ReasoningTrace
  config.py        # Pydantic config models, YAML loader
  llm_clients.py   # LiteLLM-based unified LLM client
  registry.py      # Filesystem skill registry (scans SKILL.md files)
  router.py        # ReAct iterative router (Thought → Action → Observation)
  pipeline.py      # Iterative execution loop with max_turns safety cap
  answerer.py      # Final answer synthesis (video LLM or text LLM fallback)
  video_llm.py     # Qwen2.5-Omni wrapper (load once, reuse)
  env.py           # Minimal .env loader
  skill_fs.py      # SKILL.md YAML front-matter parser

skills/
  asr/             # Speech recognition (local faster-whisper by default, optional cloud fallback)
  ocr/             # Text extraction (PaddleOCR, temporal deduplication)
  motion/          # Object tracking & counting (YOLOv8 + ByteTrack)
  object_detect/   # Open-vocabulary object detection (YOLO-World v2)
  action/          # Action/activity recognition (placeholder)
  face/            # Face & emotion analysis (DeepFace)
  scene/           # Zero-shot scene classification (OpenCLIP ViT-B-32)
  spatial/         # Spatial relationship reasoning (placeholder)

config.yaml        # Pipeline configuration
demo.py            # CLI entry point
```

## Skills

| Skill | Status | Backend | Description |
|-------|--------|---------|-------------|
| **asr** | Implemented | faster-whisper / OpenAI Whisper API | Transcribes speech; local GPU-first by default, optional cloud fallback |
| **ocr** | Implemented | PaddleOCR (ch+en) | Extracts on-screen text (subtitles, signage, plates) with temporal deduplication via SequenceMatcher |
| **motion** | Implemented | YOLOv8s + ByteTrack | Tracks objects across frames; reports **exact counts** of unique instances and per-track visibility spans |
| **object_detect** | Implemented | YOLO-World v2 (yolov8s-worldv2) | Open-vocabulary detection — accepts arbitrary target classes via router arguments; aggregates detections into temporal spans |
| action | Placeholder | — | Action/activity recognition (VideoMAE, TimeSformer) |
| **face** | Implemented | DeepFace (opencv backend) | Detects faces and analyses emotion, age, and gender; aggregates dominant emotion and average age across frames |
| **scene** | Implemented | OpenCLIP (ViT-B-32, laion2b) | Zero-shot scene classification against 45+ environment labels; supports dynamic candidate labels via router arguments |
| spatial | Placeholder | — | Spatial relationship reasoning from bounding boxes |

## Quick start

```bash
# Install
uv sync

# Run with video LLM (loads Qwen2.5-Omni-7B, needs GPU)
uv run python demo.py --question "What is happening in this video?" --video demo.mp4

# Run without video LLM (text-only answerer via LiteLLM proxy)
uv run python demo.py --question "车牌号是多少？" --video demo.mp4 --no-video-llm

# Override max reasoning turns
uv run python demo.py --question "How many people are there?" --video demo.mp4 --max-turns 5
```

## Configuration

### config.yaml

```yaml
router:
  strategy: auto          # "llm", "rules", or "auto" (try llm, fall back to rules)
  model: gpt-4o-mini
  max_tokens: 200         # increased for ReAct Thought+Action output

answerer:
  model: gpt-4o-mini
  max_tokens: 256

video_llm:
  enabled: true
  model_name: Qwen/Qwen2.5-Omni-7B
  torch_dtype: auto
  device_map: auto

skills_root: skills
max_turns: 3              # max ReAct reasoning turns before forced FINISH
max_skill_calls: null     # optional hard cap on number of executed skills
```

**`max_turns`** controls how many Thought→Action→Observation cycles the router may execute before forcing a FINISH. Higher values allow deeper multi-step reasoning at the cost of more LLM calls.

**`object_detect` dynamic classes**: The router can pass `target_classes` via its `arguments` dict (e.g., `CALL_SKILL(object_detect)` with `{"target_classes": ["fire extinguisher", "red cup"]}`). If none are provided, a broad 40-class default vocabulary is used.

### .env

```
OPENAI_API_KEY=<your-key>
OPENAI_BASE_URL=<litellm-proxy-url>
OPENAI_MODEL=gpt-4o-mini
OPENAI_ASR_MODEL=whisper-1

# ASR cost-control defaults (no ASR API spend)
ASR_BACKEND=local
ASR_ALLOW_REMOTE_FALLBACK=0
FAST_WHISPER_MODEL=large-v3
FAST_WHISPER_DEVICE=cuda
FAST_WHISPER_COMPUTE=float16
ASR_CACHE_DIR=.cache/asr
```

### Skill-specific environment variables

| Variable | Default | Skill | Description |
|----------|---------|-------|-------------|
| `YOLOWORLD_MODEL` | `yolov8s-worldv2.pt` | object_detect | YOLO-World model weight file |
| `YOLOWORLD_CONF` | `0.3` | object_detect | Confidence threshold |
| `YOLOWORLD_SAMPLE_FPS` | `1.0` | object_detect | Frame sampling rate |
| `MOTION_YOLO_MODEL` | `yolov8s.pt` | motion | YOLO model for tracking |
| `MOTION_CONF` | `0.35` | motion | Confidence threshold |
| `MOTION_SAMPLE_FPS` | `2.0` | motion | Frame sampling rate |
| `OCR_SAMPLE_FPS` | `1.0` | ocr | Frame sampling rate |
| `OCR_DEDUP_THRESHOLD` | `0.8` | ocr | SequenceMatcher similarity threshold for deduplication |
| `OCR_MIN_TEXT_LEN` | `2` | ocr | Minimum text length to keep |
| `SCENE_CLIP_MODEL` | `ViT-B-32` | scene | OpenCLIP model architecture |
| `SCENE_CLIP_PRETRAINED` | `laion2b_s34b_b79k` | scene | OpenCLIP pretrained weights |
| `SCENE_NUM_KEYFRAMES` | `5` | scene | Number of keyframes to sample |
| `FACE_SAMPLE_FPS` | `0.5` | face | Frame sampling rate (1 frame per 2s) |
| `FACE_DETECTOR` | `opencv` | face | DeepFace detector backend |
| `ASR_BACKEND` | `local` | asr | Backend policy: `local`, `local_first`, `auto`, `cloud_first`, `cloud` |
| `ASR_ALLOW_REMOTE_FALLBACK` | `0` | asr | If `1`, allows cloud ASR fallback when local path fails |
| `FAST_WHISPER_MODEL` | `large-v3` | asr | faster-whisper model size |
| `FAST_WHISPER_DEVICE` | `cuda` if available | asr | Device for faster-whisper (`cuda` or `cpu`) |
| `FAST_WHISPER_COMPUTE` | `float16` on CUDA, else `int8` | asr | faster-whisper compute type |
| `FAST_WHISPER_BEAM_SIZE` | `5` | asr | Beam size for local ASR decoding |
| `FAST_WHISPER_VAD_FILTER` | `true` | asr | Enable VAD filtering for local ASR |
| `ASR_CACHE_DIR` | `.cache/asr` | asr | Transcript cache directory (persisted to disk) |

## Adding a skill

1. Create `skills/<name>/SKILL.md` with YAML front matter:
   ```
   ---
   name: <name>
   description: What this skill does.
   tags: tag1, tag2
   when_to_use: When to activate this skill.
   ---
   ```

2. Optionally add `skills/<name>/runner.py`:
   ```python
   from skill_moe.base import SkillRequest, SkillResponse, SkillMetadata

   def run(request: SkillRequest, metadata: SkillMetadata) -> SkillResponse:
       # your implementation
       return SkillResponse(skill_name=metadata.name, summary="...", artifacts={})
   ```

3. The ReAct router will automatically discover and route to the new skill.

## TODO

- [x] Implement real OCR runner (`skills/ocr/runner.py`) — PaddleOCR with temporal deduplication
- [x] Implement real motion tracking runner (`skills/motion/runner.py`) — YOLOv8 + ByteTrack
- [x] Implement open-vocabulary object detection (`skills/object_detect/runner.py`) — YOLO-World v2
- [x] Upgrade router from single-pass to iterative ReAct reasoning loop
- [x] Implement scene classification runner (`skills/scene/runner.py`) — OpenCLIP ViT-B-32 zero-shot
- [ ] Implement action recognition runner (`skills/action/runner.py`)
- [x] Implement face/emotion detection runner (`skills/face/runner.py`) — DeepFace emotion, age, gender
- [ ] Implement spatial reasoning runner (`skills/spatial/runner.py`)
- [ ] Populate `SkillResponse.cost_estimate` for efficiency benchmarking
- [ ] Add `clip_range` support in ASR runner to trim inference cost
- [ ] Router argument passing for `target_classes` in structured output mode
