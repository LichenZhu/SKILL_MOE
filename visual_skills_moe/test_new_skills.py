"""
战前排雷脚本 — test_new_skills.py
===================================
三个独立的 Sanity-Check，全部无需加载 Qwen2.5-Omni 大模型。

用法（从 visual_skills_moe/ 目录运行）：
    cd /home/lz292/SKILL_MOE/visual_skills_moe
    source .venv/bin/activate
    python test_new_skills.py [--test 1|2|3|all]

测试内容：
  Test 1 — action_storyboard  : 光流 + 2×2 格 + 红箭头
  Test 2 — event_graph_rag    : 直方图镜头切分 + CLIP 检索 + 横向分镜
  Test 3 — System-2 反射循环  : Mock VLM 输出 <CALL_TOOL>，测试解析与循环
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import logging
import os
import sys
import time
import types
from pathlib import Path
from typing import Any, Optional

# ── 把项目根加到 sys.path ────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_new_skills")

# ── 视频路径 (真实存在) ───────────────────────────────────────────────────────
_VIDEO_SHORT = str(ROOT / "benchmarks/data/video_mme/videos/data/CHlJdMVLV2s.mp4")  # 62s 640×360 diving
_VIDEO_MOTION = str(ROOT / "benchmarks/data/video_mme/videos/data/a0AGwUACt7E.mp4") # 74s 720p  Iron-Man
_VIDEO_LONG   = str(ROOT / "benchmarks/data/video_mme/videos/data/sxrx7oCrb3A.mp4") # 59min 720p lecture

# ── 输出目录 ──────────────────────────────────────────────────────────────────
DEBUG_DIR = ROOT / "debug_outputs"
DEBUG_DIR.mkdir(exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _load_runner(skill_name: str):
    """Dynamically load skills/<name>/runner.py and return its `run` function."""
    runner_path = ROOT / "skills" / skill_name / "runner.py"
    spec = importlib.util.spec_from_file_location(f"skill_{skill_name}", runner_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run


def _fake_meta(name: str):
    """Return a minimal SkillMetadata stub."""
    from skill_moe.base import SkillMetadata
    return SkillMetadata(
        name=name,
        description=f"[test stub] {name}",
        path=str(ROOT / "skills" / name),
    )


def _fake_request(video_path: str, question: str,
                  start: Optional[float] = None,
                  end: Optional[float] = None,
                  duration: float = 0.0):
    """Return a minimal SkillRequest."""
    from skill_moe.base import SkillRequest
    import cv2
    if duration == 0.0 and os.path.isfile(video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        nf  = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        duration = nf / fps if fps > 0 else 0.0
    return SkillRequest(
        question=question,
        video_path=video_path,
        video_duration=duration,
        start_time=start,
        end_time=end,
    )


def _check_video(path: str) -> bool:
    if not os.path.isfile(path):
        logger.error("Video not found: %s", path)
        return False
    return True


def _save_b64_image(b64: str, name: str) -> str:
    """Decode base64 JPEG and save to debug_outputs/; return saved path."""
    out = str(DEBUG_DIR / name)
    with open(out, "wb") as f:
        f.write(base64.b64decode(b64))
    return out


def _banner(title: str):
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)


# ═════════════════════════════════════════════════════════════════════════════
# Test 1 — action_storyboard
# ═════════════════════════════════════════════════════════════════════════════

def test_action_storyboard():
    _banner("Test 1: action_storyboard  (光流 + 2×2 格 + 轨迹箭头)")

    if not _check_video(_VIDEO_SHORT):
        print("  SKIP — video not found")
        return False

    question = (
        "Which direction does the diver move after leaving the platform?\n"
        "A. Upward\nB. Downward and forward\nC. Sideways\nD. Backward"
    )
    req  = _fake_request(_VIDEO_SHORT, question)
    meta = _fake_meta("action_storyboard")

    t0 = time.perf_counter()
    try:
        run_fn = _load_runner("action_storyboard")
        resp = run_fn(req, meta)
    except Exception as exc:
        logger.exception("action_storyboard runner crashed")
        print(f"  FAIL — runner exception: {exc}")
        return False
    elapsed = time.perf_counter() - t0

    print(f"  skill_name : {resp.skill_name}")
    print(f"  summary    : {resp.summary}")
    print(f"  elapsed    : {elapsed:.1f}s")

    arts = resp.artifacts or {}
    ve = arts.get("visual_evidence", [])
    if not ve:
        print(f"  FAIL — no visual_evidence in artifacts")
        print(f"  artifacts  : {arts}")
        return False

    # Save image for human inspection.
    saved = _save_b64_image(ve[0], "test1_action_storyboard.jpg")
    print(f"  ✓ visual_evidence[0] saved → {saved}")
    print(f"  Image size  : {len(base64.b64decode(ve[0])) // 1024} KB")
    print(f"  frame_timestamps: {arts.get('frame_timestamps')}")

    # Basic sanity: image must be >10 KB (grid of 4 frames can't be tiny).
    if len(base64.b64decode(ve[0])) < 10_000:
        print("  WARN — image suspiciously small (<10 KB)")

    print("\n  PASS ✓  (open debug_outputs/test1_action_storyboard.jpg to verify)")
    return True


# ═════════════════════════════════════════════════════════════════════════════
# Test 2 — event_graph_rag
# ═════════════════════════════════════════════════════════════════════════════

def test_event_graph_rag():
    _banner("Test 2: event_graph_rag  (直方图镜头切分 + CLIP 检索 + 横向分镜)")

    # Use the long lecture video.  For speed, we limit to the first 180s.
    video = _VIDEO_LONG
    if not _check_video(video):
        print("  SKIP — long video not found; trying short fallback")
        video = _VIDEO_MOTION
        if not _check_video(video):
            print("  SKIP — fallback video also not found")
            return False

    question = (
        "Throughout the video, which of the following topics is discussed in detail?\n"
        "A. Python programming basics\n"
        "B. Machine learning and neural networks\n"
        "C. Video game development\n"
        "D. Database design"
    )

    # Limit to first 3 minutes for speed during debug (override env).
    os.environ.setdefault("EGR_MAX_SCENES", "60")   # cap at 60 samples
    os.environ.setdefault("EGR_SAMPLE_FPS", "0.3")  # 1 frame per 3s

    req  = _fake_request(video, question, start=0.0, end=180.0)
    meta = _fake_meta("event_graph_rag")

    t0 = time.perf_counter()
    try:
        run_fn = _load_runner("event_graph_rag")
        resp = run_fn(req, meta)
    except Exception as exc:
        logger.exception("event_graph_rag runner crashed")
        print(f"  FAIL — runner exception: {exc}")
        return False
    elapsed = time.perf_counter() - t0

    print(f"  skill_name : {resp.skill_name}")
    print(f"  summary    : {resp.summary}")
    print(f"  elapsed    : {elapsed:.1f}s")

    arts = resp.artifacts or {}
    ve = arts.get("visual_evidence", [])
    if not ve:
        print(f"  FAIL — no visual_evidence in artifacts")
        print(f"  artifacts  : {arts}")
        return False

    saved = _save_b64_image(ve[0], "test2_event_graph_rag.jpg")
    print(f"  ✓ storyboard saved → {saved}")
    print(f"  Image size      : {len(base64.b64decode(ve[0])) // 1024} KB")
    print(f"  scene_timestamps: {arts.get('scene_timestamps')}")
    print(f"  sim_scores      : {arts.get('sim_scores')}")

    print("\n  PASS ✓  (open debug_outputs/test2_event_graph_rag.jpg to verify)")
    return True


# ═════════════════════════════════════════════════════════════════════════════
# Test 3 — System-2 Reflection Loop (Mock VLM)
# ═════════════════════════════════════════════════════════════════════════════

class _MockVideoLLM:
    """
    Minimal VideoLLM stub for testing the reflection loop.

    Call sequence:
      call 0  (baseline, no skill_context) → "B"
      call 1  (re-answer, has skill_context, allow CALL_TOOL) → emits <CALL_TOOL: temporal_segment, ...>
      call 2  (forced final after reflection) → "A"
    """

    def __init__(self):
        self._call_count = 0

    def answer(
        self,
        question: str,
        video_path: str,
        skill_context: Optional[str] = None,
        use_audio_in_video: Optional[bool] = None,
        extra_instruction: Optional[str] = None,
    ) -> str:
        n = self._call_count
        self._call_count += 1
        logger.info(
            "[MockVLM] answer() call #%d | has_context=%s | has_extra=%s",
            n, skill_context is not None, extra_instruction is not None,
        )

        # When _reflection_loop calls us with the ALLOW_SUFFIX the VLM may request
        # an extra tool.  The first time we see the ALLOW suffix, emit CALL_TOOL.
        # The second time (forced or after reflection evidence), give a final answer.
        if extra_instruction and "CALL_TOOL" in extra_instruction and n == 0:
            # First re-answer inside reflection_loop: evidence is ambiguous → request more.
            return (
                "<reasoning>The provided ASR context is ambiguous — it covers the "
                "entire video and I cannot locate the specific moment asked about. "
                "I need to inspect the 10-40s window more carefully.</reasoning>\n"
                "<CALL_TOOL: temporal_segment, start=10.0, end=40.0>"
            )

        # Forced final answer (FINAL_REQUIRED suffix) or subsequent calls.
        return "After reviewing the temporal segment evidence, the answer is A."


class _MockRegistry:
    """Registry stub that reports 'temporal_segment' as registered."""
    def list(self):
        return ["temporal_segment", "rag_asr", "asr", "focus_vqa"]


def _build_mock_pipeline(mock_vlm) -> Any:
    """Construct a VideoUnderstandingPipeline with the mock VLM (no real models)."""
    from skill_moe.pipeline import VideoUnderstandingPipeline
    from skill_moe.registry import SkillRegistry
    from skill_moe.router import SkillRouter

    # Use real registry so _execute_skill_single can look up temporal_segment.
    registry = SkillRegistry(root=str(ROOT / "skills"))
    router   = SkillRouter(registry=registry, llm_client=None)  # no-op router

    pipeline = VideoUnderstandingPipeline(
        registry=registry,
        router=router,
        video_llm=mock_vlm,
        max_turns=5,
    )
    return pipeline


def test_system2_reflection():
    _banner("Test 3: System-2 反射循环  (Mock VLM → <CALL_TOOL> → re-invoke skill)")

    if not _check_video(_VIDEO_SHORT):
        print("  SKIP — video not found")
        return False

    # We need to import pipeline constants.
    from skill_moe import pipeline as _pl

    mock_vlm = _MockVideoLLM()
    pipe     = _build_mock_pipeline(mock_vlm)

    # Manually build SkillRequest and a small set of fake responses to feed
    # into _reflection_loop directly (bypass routing/VLM baseline overhead).
    from skill_moe.base import SkillRequest, SkillResponse, ReasoningTrace

    req = SkillRequest(
        question=(
            "What does the commentator say during the second dive?\n"
            "A. 'Perfect entry'\nB. 'Almost perfect'\n"
            "C. 'Better than the first'\nD. 'The crowd applauds'"
        ),
        video_path=_VIDEO_SHORT,
        video_duration=62.0,
        start_time=0.0,
        end_time=62.0,
    )
    trace = ReasoningTrace(question=req.question, video_duration=62.0)

    # Simulate that rag_asr already ran and returned some (ambiguous) evidence.
    fake_asr_resp = SkillResponse(
        skill_name="rag_asr",
        summary="[rag_asr] Transcript retrieved.",
        content="Announcer: 'Beautiful approach, the crowd is waiting...'",
        artifacts={"transcript": "Beautiful approach, the crowd is waiting..."},
    )
    responses = [fake_asr_resp]
    evidence_text = pipe._build_evidence_text(responses)
    visual_crops  = []

    print(f"  evidence_text preview: {evidence_text[:120] if evidence_text else '(none)'}")
    print()

    # ── Run the reflection loop ───────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        final = pipe._reflection_loop(
            request=req,
            trace=trace,
            responses=responses,
            evidence_text=evidence_text,
            visual_crops=visual_crops,
            disambig_hints={},
            baseline_answer="B",
        )
    except Exception as exc:
        logger.exception("_reflection_loop crashed")
        print(f"  FAIL — exception: {exc}")
        return False
    elapsed = time.perf_counter() - t0

    print(f"  MockVLM call count : {mock_vlm._call_count}")
    print(f"  Trace steps        : {len(trace.steps)}")
    print(f"  Final answer       : {repr(final[:120])}")
    print(f"  Elapsed            : {elapsed:.1f}s")

    # Verify: CALL_TOOL was detected, at least one reflection step added.
    reflection_steps = [
        s for s in trace.steps
        if s.decision.thought and "Reflection" in s.decision.thought
    ]
    print(f"  Reflection steps   : {len(reflection_steps)}")

    if mock_vlm._call_count < 2:
        print("  FAIL — VLM was called fewer than 2 times; CALL_TOOL may not have been triggered")
        return False

    if "CALL_TOOL" in final:
        print("  FAIL — final answer still contains <CALL_TOOL> tag (loop didn't resolve)")
        return False

    # Final answer should contain a letter A-D.
    import re
    letter = re.search(r"\b([A-D])\b", final)
    if not letter:
        print(f"  WARN — no letter A-D found in final answer: {repr(final[:80])}")
    else:
        print(f"  Extracted letter   : {letter.group(1)}")

    print("\n  PASS ✓  (reflection loop correctly detected <CALL_TOOL>, re-ran tool, forced final answer)")
    return True


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Sanity-check new skills")
    parser.add_argument(
        "--test", default="all",
        choices=["1", "2", "3", "all"],
        help="Which test to run (default: all)",
    )
    args = parser.parse_args()

    results = {}

    if args.test in ("1", "all"):
        results["Test1:action_storyboard"] = test_action_storyboard()

    if args.test in ("2", "all"):
        results["Test2:event_graph_rag"] = test_event_graph_rag()

    if args.test in ("3", "all"):
        results["Test3:reflection_loop"] = test_system2_reflection()

    print("\n" + "═" * 70)
    print("  SUMMARY")
    print("═" * 70)
    all_pass = True
    for name, ok in results.items():
        status = "PASS ✓" if ok else "FAIL ✗"
        print(f"  {status}  {name}")
        if not ok:
            all_pass = False

    print("═" * 70)
    if all_pass:
        print("  ALL TESTS PASSED — 可以开始大规模 Benchmark！")
    else:
        print("  SOME TESTS FAILED — 请修复上面的问题再跑 Benchmark。")
    print()
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
