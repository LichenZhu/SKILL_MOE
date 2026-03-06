"""
Demo for Skill-MoE aligned with Claude/SkillsBench skill format.

Run (with Qwen2.5-Omni video LLM answerer):
  python demo.py --question "What is happening?" --video demo.mp4

Run (text-only LLM answerer, skip loading video LLM):
  python demo.py --question "车牌号是多少？" --video demo.mp4 --no-video-llm
"""

import argparse
import os

from skill_moe.env import load_env

# Load .env once at the application entry point.
load_env()

from skill_moe.base import SkillRequest
from skill_moe.answerer import answer
from skill_moe.config import load_config
from skill_moe.llm_clients import LiteLLMClient
from skill_moe.pipeline import VideoUnderstandingPipeline
from skill_moe.registry import SkillRegistry
from skill_moe.router import SkillRouter


def build_pipeline(
    skills_root: str = "skills",
    max_turns: int = 3,
    router_strategy: str = "auto",
    router_model: str = "gpt-5.2-codex",
    router_max_tokens: int = 200,
    video_llm: object = None,
) -> VideoUnderstandingPipeline:
    registry = SkillRegistry(root=skills_root)
    router_client = None
    has_llm_endpoint = (
        bool(os.getenv("OPENAI_API_KEY"))
        or bool(os.getenv("ANTHROPIC_API_KEY"))
        or bool(os.getenv("OPENAI_BASE_URL"))
    )
    if has_llm_endpoint and router_strategy.strip().lower() in {"llm", "auto"}:
        router_client = LiteLLMClient(model=router_model)
    router = SkillRouter(
        registry,
        llm_client=router_client,
        strategy=router_strategy,
        llm_max_tokens=router_max_tokens,
    )
    return VideoUnderstandingPipeline(
        registry,
        router,
        max_turns=max_turns,
        video_llm=video_llm,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True, help="User question")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--skills-root", default=None, help="Directory containing skill folders (overrides config)")
    parser.add_argument("--max-turns", type=int, default=None, help="Max skill turns (overrides config)")
    parser.add_argument("--no-video-llm", action="store_true", help="Skip video LLM; use text-only answerer")
    args = parser.parse_args()

    cfg = load_config(args.config)
    skills_root = args.skills_root or cfg.skills_root
    max_turns = args.max_turns if args.max_turns is not None else cfg.max_turns

    # --- optionally load video LLM at startup ---
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

    pipeline = build_pipeline(
        skills_root,
        max_turns=max_turns,
        router_strategy=cfg.router.strategy,
        router_model=cfg.router.model,
        router_max_tokens=cfg.router.max_tokens,
        video_llm=video_llm,
    )
    request = SkillRequest(question=args.question, video_path=args.video)

    # Run the pipeline and get the full trace.
    trace = pipeline.run_trace(request)

    # Display the reasoning trace.
    print(f"\n{'='*60}")
    print(f"Question: {args.question}")
    print(f"{'='*60}")
    for step in trace.steps:
        d = step.decision
        print(f"\n--- Turn {step.step} ---")
        print(f"Thought: {d.thought}")
        print(f"Action:  {d.action.value}", end="")
        if d.skill_name:
            print(f"({d.skill_name})", end="")
        print()
        if step.response:
            print(f"Result:  {step.response.summary[:300]}")

    responses = trace.responses
    if responses:
        print(f"\n{'='*60}")
        print(f"Skills executed: {', '.join(r.skill_name for r in responses)}")
    else:
        print("\nNo skills were executed.")

    if trace.final_answer:
        final = trace.final_answer
    else:
        final = answer(
            args.question,
            responses,
            trace=trace,
            video_path=args.video,
            video_llm=video_llm,
            max_tokens=cfg.answerer.max_tokens,
        )
    print(f"\n{'='*60}")
    print("Final answer:\n", final)


if __name__ == "__main__":
    main()
