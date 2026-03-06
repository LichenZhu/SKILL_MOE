# Visual Skill-MoE

Mixture-of-Experts framework for video multiple-choice QA. A local video LLM (Qwen2.5-Omni-7B) provides the baseline answer; a parallel fleet of specialized skills provides structured evidence; a CONFIRM/CONTRADICT gate decides whether skill evidence is strong enough to override the baseline.

## Architecture

```
Question + Video
      |
      v
 Video LLM (Qwen2.5-Omni-7B)
 → baseline_answer (A/B/C/D)
      |
      +-----> MetaRouter (rule-based + LLM gate)
      |          selects skills for this question
      |
      +-----> Parallel Skill Execution
      |          asr / ocr / focus_vqa / tracking / ...
      |          each returns evidence or option_scores
      |
      v
 CONFIRM / CONTRADICT check (lightweight LLM call)
      |
  CONFIRM ──→ return baseline_answer  (fast path, no extra VLM call)
      |
  CONTRADICT
      |
      v
 Video LLM re-answers with skill evidence injected
      |
      v
 Final answer
```

**Skill types**

| Type | Behavior |
|------|----------|
| `support` | Provides text or visual evidence; cannot directly override baseline |
| `override` | Returns structured `option_scores` / deterministic count; can directly inform answer |

## Skills

| Skill | Type | Backend | When to use |
|-------|------|---------|-------------|
| **asr** | support | faster-whisper (local GPU) | Questions about spoken words or dialogue |
| **rag_asr** | support | asr + LLM filter | "What was mentioned/discussed" — filters transcript to relevant sentences |
| **ocr** | support | EasyOCR (CPU, en+zh) | On-screen text, scores, timers, signs, license plates |
| **focus_vqa** | support | GroundingDINO-tiny + Spotlight | Fine-grained object detail — text on labels, logos, what someone is holding |
| **grounding** | support | GroundingDINO-tiny | Localise a named object; returns bounding boxes |
| **temporal_segment** | support | Vision LLM on 24-frame grid | Questions targeting a specific temporal window ("second half", "last scene") |
| **action_storyboard** | support | Frame-diff + Spotlight 2×2 grid | Motion direction, trajectory, action sequence |
| **event_graph_rag** | support | CLIP ViT-H-14 scene retrieval | Long videos (>3 min): cross-scene retrieval, "what is the video about" |
| **visual_option_match** | override | CLIP ViT-H-14 @ 1 FPS | Visual attribute MCQ (clothing color/style, object appearance) |
| **zero_shot_identity** | override | YOLOv8n + CLIP ViT-H-14 | Person identity — "who is wearing X", "what role does Y play" |
| **tracking** | override | YOLOv8n + ByteTrack | Unique entity count — "how many challengers", "how many people appear" |
| **temporal_action_counter** | override | CLIP ViT-H-14 NO→YES transitions | Event frequency — "how many times does X happen" |
| **temporal_ordering** | support | CLIP ViT-H-14 @ 2 FPS | Chronological order of MCQ options (MetaRouter-gated, risky) |

## Layout

```
skill_moe/
  base.py           # SkillRequest, SkillResponse, SkillMetadata, ReasoningTrace
  config.py         # Pydantic config, YAML loader
  llm_clients.py    # LiteLLM-based unified LLM client (OPENAI_BASE_URL proxy)
  registry.py       # Filesystem skill registry (scans skills/*/SKILL.md)
  router.py         # Sequential LLM router (legacy path, still used as fallback)
  pipeline.py       # Main pipeline: VLM baseline → skills → CONFIRM/CONTRADICT → re-answer
  answerer.py       # Evidence-first answer synthesis (counting, ASR/OCR lexical match)
  video_llm.py      # Qwen2.5-Omni-7B wrapper
  llm_clients.py    # LiteLLM client (supports local proxy via OPENAI_BASE_URL)
  verifier.py       # Optional evidence verifier (disabled by default)
  visual_answerer.py# VLM re-answer with visual crop injection

skills/
  asr/              # faster-whisper speech transcription
  rag_asr/          # ASR + LLM relevance filter
  ocr/              # EasyOCR text extraction
  focus_vqa/        # GroundingDINO crop + Spotlight injection
  grounding/        # GroundingDINO object localisation
  temporal_segment/ # Dense frame sampling + vision LLM description
  action_storyboard/# Frame-diff motion storyboard
  event_graph_rag/  # CLIP scene retrieval (long videos)
  visual_option_match/ # CLIP ViT-H-14 MCQ scoring
  zero_shot_identity/  # YOLOv8 + CLIP person identity
  tracking/         # YOLOv8 + ByteTrack entity counting
  temporal_action_counter/ # CLIP event frequency counting
  temporal_ordering/   # CLIP chronological ordering (risky, gated)

config.yaml         # Pipeline configuration
benchmark.py        # Evaluation on VideoMME-style JSON datasets
demo.py             # CLI entry point
run_benchmark.sh    # Convenience wrapper (auto-selects free GPU)
```

## Quick start

```bash
# Install
uv sync

# Run benchmark (auto-selects GPU with most free VRAM)
./run_benchmark.sh benchmarks/analysis/random500_seed20260304.json

# Or manually on a specific GPU
CUDA_VISIBLE_DEVICES=6 python benchmark.py \
    --dataset benchmarks/analysis/random500_seed20260304.json \
    --limit 500

# Baseline only (no skills)
CUDA_VISIBLE_DEVICES=6 python benchmark.py \
    --dataset benchmarks/analysis/random500_seed20260304.json \
    --no-skills

# Demo on a single video
python demo.py --question "What sport is being played?" --video demo.mp4
```

## Configuration

### config.yaml

```yaml
router:
  strategy: auto          # "llm", "rules", or "auto"
  model: gpt-5.2-codex    # LLM for MetaRouter semantic gate
  max_tokens: 220

answerer:
  model: gpt-5.2-codex
  max_tokens: 256

video_llm:
  enabled: true
  model_name: Qwen/Qwen2.5-Omni-7B
  torch_dtype: auto
  device_map: auto
  max_frames: 64
  total_pixels: 20971520  # ~26.8K visual tokens
  use_audio: false

skills_root: skills
max_turns: 5

verifier:
  enabled: false          # optional evidence verifier (not used by default)
```

### Environment variables

```bash
# LLM proxy (required for MetaRouter, CONFIRM/CONTRADICT, rag_asr, temporal_segment)
OPENAI_API_KEY=<key>
OPENAI_BASE_URL=<litellm-proxy-url>
OPENAI_MODEL=gpt-5.2-codex

# ASR
ASR_BACKEND=local                  # local | cloud
ASR_ALLOW_REMOTE_FALLBACK=0
FAST_WHISPER_MODEL=large-v3
FAST_WHISPER_DEVICE=cuda
FAST_WHISPER_COMPUTE=float16
ASR_CACHE_DIR=.cache/asr

# OCR
OCR_SAMPLE_FPS=1.0

# focus_vqa
FOCUS_VQA_NUM_FRAMES=8
FOCUS_VQA_MAX_CROPS=3
FOCUS_VQA_BOX_THRESHOLD=0.22
FOCUS_VQA_CROP_MAX_SIDE=1024

# grounding
GROUNDING_MODEL=IDEA-Research/grounding-dino-tiny

# tracking
TRACKING_MODEL=yolov8n.pt
TRACKING_SAMPLE_FPS=2.0
TRACKING_MAX_FRAMES=600

# zero_shot_identity
ZSI_NUM_FRAMES=8
ZSI_MAX_CROPS_PER_FRAME=4
ZSI_YOLO_CONF=0.20
ZSI_YOLO_MODEL=yolov8n.pt

# visual_option_match
VISOPT_CLIP_MODEL=hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K
VISOPT_SAMPLE_FPS=1.0
VISOPT_MAX_FRAMES=120
```

## Adding a skill

1. Create `skills/<name>/SKILL.md`:
   ```yaml
   ---
   name: <name>
   description: What this skill does.
   tags: tag1, tag2
   when_to_use: When to activate this skill.
   skill_type: support   # or: override
   ---
   ```

2. Add `skills/<name>/runner.py`:
   ```python
   from skill_moe.base import SkillRequest, SkillResponse, SkillMetadata

   def run(request: SkillRequest, metadata: SkillMetadata) -> SkillResponse:
       return SkillResponse(skill_name=metadata.name, summary="...", artifacts={})
   ```

3. Add routing triggers in `pipeline.py` → `_route_skills()`.

4. If the skill is unreliable or expensive, add it to `_RISKY_SKILLS` so the MetaRouter LLM gate approves it per-question.
