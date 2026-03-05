"""
Automated benchmarking suite for Skill-MoE on VideoMME.

Supports two dataset formats:
  1. VideoMME parquet (multiple-choice, accuracy-based evaluation)
  2. Custom JSON (open-ended, LLM-as-Judge scoring 0-5)

Usage:
    # Run on mini_test.json (20 items sampled from VideoMME)
    uv run python benchmark.py --dataset benchmarks/mini_test.json --no-video-llm

    # Run on full VideoMME parquet
    uv run python benchmark.py --dataset benchmarks/data/video_mme/videomme/test-00000-of-00001.parquet --no-video-llm

    # Limit number of samples for quick testing
    uv run python benchmark.py --dataset benchmarks/mini_test.json --no-video-llm --limit 5

    # Override max turns and judge model
    uv run python benchmark.py --dataset benchmarks/mini_test.json --max-turns 5 --judge-model gpt-4o
"""

from __future__ import annotations

import argparse
import faulthandler
import gc
import json
import os
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

# Print C-level traceback on segfault / fatal signal
faulthandler.enable()

import pandas as pd
from tqdm import tqdm

from skill_moe.env import load_env

load_env()

from skill_moe.answerer import answer
from skill_moe.base import SkillRequest
from skill_moe.config import load_config
from skill_moe.llm_clients import LiteLLMClient
from skill_moe.pipeline import VideoUnderstandingPipeline
from skill_moe.registry import SkillRegistry
from skill_moe.router import SkillRouter

# ---------------------------------------------------------------------------
# LLM-as-a-Judge (for open-ended questions only)
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are an impartial judge evaluating a video understanding system.

Compare the system's prediction against the ground truth answer.

## Scoring rubric (0-5)
- 5: Perfect — prediction matches ground truth exactly or with trivial paraphrasing.
- 4: Mostly correct — captures the key information with minor omissions or additions.
- 3: Partially correct — contains the right idea but misses important details.
- 2: Weak — only tangentially related to the ground truth.
- 1: Mostly wrong — contains significant errors or hallucinations.
- 0: Completely wrong or irrelevant.

## Input
Question: {question}
Ground Truth: {ground_truth}
Prediction: {prediction}

## Output format
Respond with EXACTLY this format:
Score: <integer 0-5>
Reason: <one sentence explanation>
"""


def judge_prediction(
    question: str,
    ground_truth: str,
    prediction: str,
    client: LiteLLMClient,
) -> tuple[int, str]:
    """Ask the LLM judge to score a prediction. Returns (score, reason)."""
    prompt = _JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        prediction=prediction,
    )
    try:
        raw = client.complete(prompt, max_tokens=100)
    except Exception as exc:
        return -1, f"Judge LLM error: {exc}"

    score_match = re.search(r"Score:\s*(\d)", raw)
    score = int(score_match.group(1)) if score_match else -1
    score = min(max(score, 0), 5) if score >= 0 else -1

    reason_match = re.search(r"Reason:\s*(.+)", raw, re.DOTALL)
    reason = reason_match.group(1).strip() if reason_match else raw.strip()

    return score, reason


# ---------------------------------------------------------------------------
# Multiple-choice answer extraction
# ---------------------------------------------------------------------------

def extract_mc_answer(prediction: str) -> str:
    """Extract a single letter (A/B/C/D) from a model prediction.

    When CoT is active the model wraps reasoning in <reasoning>...</reasoning>
    and outputs the final letter immediately after.  We search that suffix
    first so the reasoning content (which may mention wrong options by letter)
    cannot pollute the result.
    """
    if not prediction:
        return ""
    # CoT format: prefer the letter that appears right after </reasoning>
    tag_end = prediction.find("</reasoning>")
    if tag_end != -1:
        after = prediction[tag_end + len("</reasoning>"):]
        m = re.search(r"\b([A-D])\b", after)
        if m:
            return m.group(1)
        m = re.search(r"(?:answer|option|choice)\s*(?:is\s*)?([A-D])", after, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    # Non-CoT / fallback: first letter in the full response
    m = re.search(r"\b([A-D])\b", prediction)
    if m:
        return m.group(1)
    m = re.search(r"(?:answer|option|choice)\s*(?:is\s*)?([A-D])", prediction, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return prediction.strip()[:1].upper()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> tuple[list[dict[str, Any]], bool]:
    """
    Load a dataset. Returns (items, is_multiple_choice).

    Supports:
      - .json: list of dicts with {question, ground_truth, ...}
      - .parquet: VideoMME format with {question, options, answer, videoID, ...}
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
        items = []
        for _, row in df.iterrows():
            opts = row["options"]
            if hasattr(opts, "tolist"):
                opts = opts.tolist()
            items.append({
                "id": str(row.get("question_id", "")),
                "video_id": str(row.get("video_id", "")),
                "video_path": f"benchmarks/data/video_mme/videos/data/{row['videoID']}.mp4",
                "question": str(row["question"]),
                "options": opts,
                "ground_truth": str(row["answer"]),
                "category": str(row.get("task_type", "")),
                "domain": str(row.get("domain", "")),
                "duration": str(row.get("duration", "")),
            })
        return items, True

    # JSON
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON array of objects.")

    is_mc = any("options" in item for item in data)

    for i, item in enumerate(data):
        if "question" not in item or "ground_truth" not in item:
            raise ValueError(
                f"Item {i} missing required fields 'question' and/or 'ground_truth'."
            )
    return data, is_mc


# ---------------------------------------------------------------------------
# Single test-case runner
# ---------------------------------------------------------------------------

def run_single(
    item: dict[str, Any],
    pipeline: VideoUnderstandingPipeline,
    video_llm: Any,
    judge_client: LiteLLMClient | None,
    answerer_max_tokens: int,
    is_mc: bool,
) -> dict[str, Any]:
    """Run one test case and return a result dict with full trace log."""
    qid = item.get("id", "?")
    question = item["question"]
    ground_truth = item["ground_truth"]
    video_path = item.get("video_path", "demo.mp4")
    category = item.get("category", "")
    options = item.get("options", [])

    # For MC questions, append options to the question
    if options:
        options_text = "\n".join(options)
        full_question = (
            f"{question}\n\nChoose the correct answer from:\n{options_text}\n\n"
            f"Answer with ONLY the letter (A, B, C, or D)."
        )
    else:
        full_question = question

    request = SkillRequest(question=full_question, video_path=video_path)

    # --- run pipeline ---
    t0 = time.perf_counter()
    trace_log: list[dict[str, Any]] = []
    try:
        trace = pipeline.run_trace(request)
        responses = trace.responses

        # Capture detailed reasoning trace
        for step in trace.steps:
            d = step.decision
            step_info: dict[str, Any] = {
                "turn": step.step,
                "thought": d.thought,
                "action": d.action.value,
                "skill_name": d.skill_name or "",
                "arguments": d.parameters,
            }
            if step.response:
                step_info["skill_output"] = step.response.summary[:500]
                step_info["skill_artifacts"] = (
                    str(step.response.artifacts)[:300]
                    if step.response.artifacts else ""
                )
            trace_log.append(step_info)

        # If the pipeline already produced a final answer (triage mode),
        # use it directly.  Otherwise fall back to the old answerer.
        if trace.final_answer:
            prediction = trace.final_answer
        else:
            prediction = answer(
                full_question,
                responses,
                trace=trace,
                video_path=video_path,
                video_llm=video_llm,
                max_tokens=answerer_max_tokens,
            )
        error = ""
    except Exception:
        exc_text = traceback.format_exc()
        _is_oom = (
            "out of memory" in exc_text.lower()
            or "outofmemory" in exc_text.lower()
            or "cudaoutofmemory" in exc_text.lower()
        )
        if _is_oom and video_llm is not None:
            # ── Graceful OOM Fallback ─────────────────────────────────────
            # A skill model (Whisper/CLIP) loaded alongside the VideoLLM and
            # pushed combined GPU memory over the limit.  Clear all cached
            # skill models, then re-run the VideoLLM with NO skill evidence
            # so the case is scored rather than marked as an error.
            tqdm.write(
                f"  [{qid}] ⚠ OOM — clearing GPU caches & falling back to VLM-only baseline"
            )
            try:
                import torch
                pipeline.clear_caches()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                prediction = video_llm.answer(
                    question=full_question,
                    video_path=video_path,
                )
                trace = None
                responses = []
                trace_log = [{"turn": 1, "thought": "OOM fallback: VLM-only (no skills)",
                               "action": "ANSWER", "skill_name": ""}]
                error = ""
            except Exception:
                trace = None
                responses = []
                prediction = ""
                error = traceback.format_exc()
        else:
            trace = None
            responses = []
            prediction = ""
            error = exc_text
    latency = time.perf_counter() - t0

    # --- trace stats ---
    total_turns = len(trace.steps) if trace else 0
    skills_used = ", ".join(trace.executed_skills) if trace else ""
    skill_count = len(trace.executed_skills) if trace else 0

    # --- evaluate ---
    if is_mc:
        predicted_letter = extract_mc_answer(prediction)
        correct = predicted_letter == ground_truth.strip().upper()
        score = 1 if correct else 0
        reason = (
            f"Predicted '{predicted_letter}', "
            f"correct is '{ground_truth.strip()}'"
        )
    else:
        if prediction and not error and judge_client:
            score, reason = judge_prediction(
                question, ground_truth, prediction, judge_client
            )
        else:
            score = -1
            reason = f"Pipeline error: {error[:200]}" if error else "No prediction"

    # --- with/without skills comparison ---
    initial_answer = trace.initial_answer if trace else ""
    initial_confidence = trace.initial_confidence if trace else ""
    final_answer = trace.final_answer if trace else ""
    # Score the initial answer (before skills) for comparison.
    if is_mc and initial_answer:
        initial_letter = extract_mc_answer(initial_answer)
        initial_score = 1 if initial_letter == ground_truth.strip().upper() else 0
    else:
        initial_letter = ""
        initial_score = -1

    return {
        "id": qid,
        "category": category,
        "question": question,
        "ground_truth": ground_truth,
        "options": options,
        "prediction": prediction,
        "score": score,
        "reason": reason,
        "latency_sec": round(latency, 2),
        "total_turns": total_turns,
        "skills_used": skills_used,
        "skill_count": skill_count,
        "trace": trace_log,
        "video_path": video_path,
        "error": error if error else "",
        # Comparison: before-skills vs after-skills.
        "initial_answer": initial_answer,
        "initial_letter": initial_letter if is_mc else "",
        "initial_score": initial_score,
        "initial_confidence": initial_confidence,
        "final_answer": final_answer,
    }


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, is_mc: bool) -> None:
    if is_mc:
        _print_mc_summary(df)
    else:
        _print_openended_summary(df)


def _print_mc_summary(df: pd.DataFrame) -> None:
    """Print accuracy-based summary for multiple-choice evaluation."""
    valid = df[df["error"] == ""]

    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS — Multiple Choice Accuracy")
    print("=" * 70)

    if valid.empty:
        print("  No successfully evaluated items.")
        print("=" * 70)
        return

    accuracy = valid["score"].mean() * 100
    avg_latency = valid["latency_sec"].mean()
    avg_turns = valid["total_turns"].mean()
    avg_skills = valid["skill_count"].mean()

    print(f"  Total cases:       {len(df)}")
    print(f"  Evaluated:         {len(valid)}")
    print(f"  Errors:            {len(df) - len(valid)}")
    print(f"  Accuracy:          {accuracy:.1f}%  ({valid['score'].sum()}/{len(valid)})")
    print(f"  Average latency:   {avg_latency:.2f}s")
    print(f"  Average turns:     {avg_turns:.1f}")
    print(f"  Average skills:    {avg_skills:.1f}")

    # Per-category
    if "category" in valid.columns and valid["category"].any():
        print(f"\n  {'Category':<25} {'Accuracy':>10} {'Count':>6}")
        print(f"  {'-'*25} {'-'*10} {'-'*6}")
        for cat, group in sorted(valid.groupby("category")):
            if not cat:
                continue
            acc = group["score"].mean() * 100
            print(f"  {cat:<25} {acc:>9.1f}% {len(group):>6}")

    # --- With/without skills comparison (triage mode) ---
    if "initial_score" in valid.columns:
        has_initial = valid[valid["initial_score"] >= 0]
        if not has_initial.empty:
            init_acc = has_initial["initial_score"].mean() * 100
            final_acc = has_initial["score"].mean() * 100
            helped = ((has_initial["score"] == 1) & (has_initial["initial_score"] == 0)).sum()
            hurt = ((has_initial["score"] == 0) & (has_initial["initial_score"] == 1)).sum()
            both_right = ((has_initial["score"] == 1) & (has_initial["initial_score"] == 1)).sum()
            both_wrong = ((has_initial["score"] == 0) & (has_initial["initial_score"] == 0)).sum()
            skipped = (has_initial["skill_count"] == 0).sum()

            print("\n  ── With/Without Skills Comparison ──")
            print(f"  Before skills (initial): {init_acc:.1f}%  ({has_initial['initial_score'].sum()}/{len(has_initial)})")
            print(f"  After skills (final):    {final_acc:.1f}%  ({has_initial['score'].sum()}/{len(has_initial)})")
            print(f"  Skills helped:  {helped}  |  Skills hurt: {hurt}")
            print(f"  Both right: {both_right}  |  Both wrong: {both_wrong}")
            print(f"  Skipped (high confidence): {skipped}")
            if hurt > 0:
                hurt_cases = has_initial[(has_initial["score"] == 0) & (has_initial["initial_score"] == 1)]
                print(f"\n  Cases where skills HURT:")
                for _, row in hurt_cases.iterrows():
                    print(f"    [{row['id']}] {row.get('category', '')} "
                          f"GT={row['ground_truth']} "
                          f"initial={row.get('initial_letter', '?')} "
                          f"final={extract_mc_answer(str(row['prediction']))} "
                          f"skills=[{row['skills_used']}]")

    print("=" * 70)


def _print_openended_summary(df: pd.DataFrame) -> None:
    """Print LLM-judge score summary for open-ended evaluation."""
    scored = df[df["score"] >= 0]

    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS — Open-Ended (LLM Judge)")
    print("=" * 70)

    if scored.empty:
        print("  No successfully scored items.")
        print("=" * 70)
        return

    avg_score = scored["score"].mean()
    avg_latency = scored["latency_sec"].mean()
    avg_turns = scored["total_turns"].mean()
    avg_skills = scored["skill_count"].mean()

    print(f"  Total cases:       {len(df)}")
    print(f"  Scored cases:      {len(scored)}")
    print(f"  Errors:            {len(df) - len(scored)}")
    print(f"  Average score:     {avg_score:.2f} / 5.00")
    print(f"  Average latency:   {avg_latency:.2f}s")
    print(f"  Average turns:     {avg_turns:.1f}")
    print(f"  Average skills:    {avg_skills:.1f}")

    if "category" in scored.columns and scored["category"].any():
        print(f"\n  {'Category':<25} {'Avg Score':>10} {'Count':>6}")
        print(f"  {'-'*25} {'-'*10} {'-'*6}")
        for cat, group in sorted(scored.groupby("category")):
            if not cat:
                continue
            print(f"  {cat:<25} {group['score'].mean():>10.2f} {len(group):>6}")

    # Score distribution
    print("\n  Score distribution:")
    for s in range(6):
        count = (scored["score"] == s).sum()
        bar = "#" * count
        print(f"    {s}: {count:>3}  {bar}")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Skill-MoE Benchmark Suite")
    parser.add_argument(
        "--dataset", required=True,
        help="Path to dataset (JSON or parquet)"
    )
    parser.add_argument("--config", default="config.yaml", help="Config YAML")
    parser.add_argument(
        "--max-turns", type=int, default=None,
        help="Override max reasoning turns"
    )
    parser.add_argument(
        "--judge-model", default=None,
        help="LLM model for judging open-ended questions (default: config answerer model)"
    )
    parser.add_argument(
        "--no-video-llm", action="store_true", help="Skip video LLM"
    )
    parser.add_argument(
        "--output-dir", default="benchmarks",
        help="Directory to save results CSV"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of test cases to run"
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Start from this index (0-based), useful for resuming"
    )
    parser.add_argument(
        "--duration", default=None, choices=["short", "medium", "long"],
        help="Filter VideoMME by video duration (parquet only)"
    )
    parser.add_argument(
        "--no-skills", action="store_true",
        help="Disable all skills (baseline: Video LLM only, no routing)"
    )
    parser.add_argument(
        "--clear-caches-each-case", action="store_true",
        help="Clear skill runner/model caches after each case (lower VRAM, slower).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable INFO-level logging (shows MetaRouter, visual-evidence, routing decisions)",
    )
    args = parser.parse_args()

    import logging
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
    else:
        logging.basicConfig(level=logging.WARNING)

    # --- config ---
    cfg = load_config(args.config)
    if args.no_skills:
        max_turns = 0
    elif args.max_turns is not None:
        max_turns = args.max_turns
    else:
        max_turns = cfg.max_turns
    # --- video LLM (load first so pipeline can use it for triage) ---
    video_llm = None
    if cfg.video_llm.enabled and not args.no_video_llm:
        from skill_moe.video_llm import VideoLLM

        print(f"Loading video LLM: {cfg.video_llm.model_name} ...")
        video_llm = VideoLLM(
            model_name=cfg.video_llm.model_name,
            torch_dtype=cfg.video_llm.torch_dtype,
            device_map=cfg.video_llm.device_map,
            max_frames=cfg.video_llm.max_frames,
            total_pixels=cfg.video_llm.total_pixels,
            use_audio=cfg.video_llm.use_audio,
        )

    # --- pipeline ---
    print("Initialising pipeline ...")
    registry = SkillRegistry(root=cfg.skills_root)
    router_client = None
    has_llm_endpoint = (
        bool(os.getenv("OPENAI_API_KEY"))
        or bool(os.getenv("ANTHROPIC_API_KEY"))
        or bool(os.getenv("OPENAI_BASE_URL"))
    )
    if has_llm_endpoint and cfg.router.strategy.strip().lower() in {"llm", "auto"}:
        router_client = LiteLLMClient(model=cfg.router.model)

    router = SkillRouter(
        registry,
        llm_client=router_client,
        strategy=cfg.router.strategy,
        llm_max_tokens=cfg.router.max_tokens,
    )
    # Pass video_llm to enable 2-phase triage pipeline.
    # When video_llm is set and max_turns > 0, the pipeline uses
    # triage → parallel skills → answer (one model, context preserved).
    pipeline_video_llm = video_llm if not args.no_skills else None

    # Build evidence verifier when enabled in config and an LLM endpoint exists.
    verifier = None
    if cfg.verifier.enabled and has_llm_endpoint and not args.no_skills:
        from skill_moe.verifier import EvidenceVerifier
        verifier_llm = LiteLLMClient(
            model=cfg.verifier.model,
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_BASE_URL"),
        )
        verifier = EvidenceVerifier(verifier_llm, max_tokens=cfg.verifier.max_tokens)
        print(f"Evidence Verifier enabled (model={cfg.verifier.model})")

    pipeline = VideoUnderstandingPipeline(
        registry,
        router,
        max_turns=max_turns,
        video_llm=pipeline_video_llm,
        verifier=verifier,
        llm_client=router_client if has_llm_endpoint else None,
    )

    # --- dataset ---
    dataset, is_mc = load_dataset(args.dataset)

    # Filter by duration if requested
    if args.duration:
        dataset = [d for d in dataset if d.get("duration") == args.duration]

    # Apply start/limit slicing
    if args.start > 0:
        dataset = dataset[args.start:]
    if args.limit:
        dataset = dataset[: args.limit]

    # --- judge client (only needed for open-ended) ---
    judge_client = None
    if not is_mc:
        judge_model = args.judge_model or cfg.answerer.model
        judge_client = LiteLLMClient(model=judge_model)
        print(f"Judge model: {judge_model}")

    mode = "Multiple Choice (accuracy)" if is_mc else "Open-Ended (LLM judge)"
    print(f"Evaluation mode: {mode}")
    print(f"Loaded {len(dataset)} test case(s) from {args.dataset}")
    print(f"Max turns: {max_turns}\n")

    # --- run metadata ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_meta = {
        "timestamp": timestamp,
        "dataset": args.dataset,
        "num_cases": len(dataset),
        "max_turns": max_turns,
        "video_llm_enabled": video_llm is not None,
        "video_llm_model": cfg.video_llm.model_name if video_llm else None,
        "router_model": cfg.router.model,
        "router_strategy": cfg.router.strategy,
        "router_max_tokens": cfg.router.max_tokens,
        "answerer_model": cfg.answerer.model,
        "duration_filter": args.duration,
        "no_skills": args.no_skills,
        "clear_caches_each_case": args.clear_caches_each_case,
    }

    # --- output dir ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- run ---
    results: list[dict[str, Any]] = []
    partial_path = output_dir / f"partial_{timestamp}.json"
    for item in tqdm(dataset, desc="Benchmarking", unit="case"):
        result = run_single(
            item, pipeline, video_llm, judge_client,
            cfg.answerer.max_tokens, is_mc
        )
        results.append(result)
        # Live feedback
        if is_mc:
            status = "✓" if result["score"] == 1 else "✗"
        else:
            status = f"score={result['score']}" if result["score"] >= 0 else "ERROR"
        # Show comparison when triage data is available.
        cmp = ""
        if result.get("initial_score") is not None and result["initial_score"] >= 0:
            i_s = "✓" if result["initial_score"] == 1 else "✗"
            f_s = "✓" if result["score"] == 1 else "✗"
            cmp = f"  init={i_s}→final={f_s}"
        tqdm.write(
            f"  [{result['id']}] {status}  "
            f"turns={result['total_turns']}  "
            f"latency={result['latency_sec']:.1f}s  "
            f"skills=[{result['skills_used']}]{cmp}"
        )
        # Save partial results after each case (crash protection)
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump({"run": run_meta, "results": results}, f, ensure_ascii=False)
        if args.clear_caches_each_case:
            # Optional memory-saving mode for constrained GPUs.
            pipeline.clear_caches()
        # Always free GPU memory between cases to prevent OOM accumulation.
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # --- save CSV (without trace column for readability) ---
    csv_results = [{k: v for k, v in r.items() if k != "trace"} for r in results]
    df = pd.DataFrame(csv_results)
    csv_path = output_dir / f"results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults CSV saved to {csv_path}")

    # --- save detailed JSON log (with full trace) ---
    scored = df[df["error"] == ""] if is_mc else df[df["score"] >= 0]
    if is_mc:
        accuracy = scored["score"].mean() * 100 if len(scored) > 0 else 0
        summary_stats = {"accuracy_pct": round(accuracy, 2), "correct": int(scored["score"].sum()), "total": len(scored)}
    else:
        summary_stats = {"avg_score": round(scored["score"].mean(), 2) if len(scored) > 0 else 0, "total": len(scored)}

    # Comparison stats for triage mode.
    comparison_stats = {}
    if is_mc and "initial_score" in df.columns:
        has_init = scored[scored["initial_score"] >= 0]
        if not has_init.empty:
            comparison_stats = {
                "initial_accuracy_pct": round(has_init["initial_score"].mean() * 100, 2),
                "final_accuracy_pct": round(has_init["score"].mean() * 100, 2),
                "skills_helped": int(((has_init["score"] == 1) & (has_init["initial_score"] == 0)).sum()),
                "skills_hurt": int(((has_init["score"] == 0) & (has_init["initial_score"] == 1)).sum()),
            }

    log_data = {
        "run": run_meta,
        "summary": {
            **summary_stats,
            "errors": len(df) - len(scored),
            "avg_latency_sec": round(df["latency_sec"].mean(), 2),
            "avg_turns": round(df["total_turns"].mean(), 1),
            "avg_skills": round(df["skill_count"].mean(), 1),
            **comparison_stats,
        },
        "results": results,
    }
    log_path = output_dir / f"log_{timestamp}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    print(f"Detailed log saved to {log_path}")

    # --- cleanup partial ---
    if partial_path.exists():
        partial_path.unlink()

    # --- summary ---
    print_summary(df, is_mc)


if __name__ == "__main__":
    main()
