"""
Gradio Web UI for Skill-MoE.

Visualizes the ReAct chain-of-thought reasoning process for ECCV
qualitative analysis.  The reasoning trace is rendered with colour-coded
Thought / Action / Observation blocks.

Usage:
    uv run python web_demo.py
    uv run python web_demo.py --no-video-llm          # skip GPU model
    uv run python web_demo.py --config config.yaml     # custom config
"""

from __future__ import annotations

import argparse
import html
import os
import traceback
from typing import Optional

import gradio as gr

from skill_moe.env import load_env

load_env()

from skill_moe.answerer import answer
from skill_moe.base import ActionType, ReasoningTrace, SkillRequest
from skill_moe.config import PipelineConfig, load_config
from skill_moe.llm_clients import LiteLLMClient
from skill_moe.pipeline import VideoUnderstandingPipeline
from skill_moe.registry import SkillRegistry
from skill_moe.router import SkillRouter

# ---------------------------------------------------------------------------
# Global state (initialised once, reused across requests)
# ---------------------------------------------------------------------------
_pipeline: Optional[VideoUnderstandingPipeline] = None
_video_llm: object = None  # Optional[VideoLLM]
_cfg: Optional[PipelineConfig] = None


def _init_pipeline(cfg: PipelineConfig, skip_video_llm: bool = False) -> None:
    global _pipeline, _video_llm, _cfg
    _cfg = cfg
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
    if cfg.video_llm.enabled and not skip_video_llm:
        from skill_moe.video_llm import VideoLLM

        _video_llm = VideoLLM(
            model_name=cfg.video_llm.model_name,
            torch_dtype=cfg.video_llm.torch_dtype,
            device_map=cfg.video_llm.device_map,
            max_frames=cfg.video_llm.max_frames,
            total_pixels=cfg.video_llm.total_pixels,
            use_audio=cfg.video_llm.use_audio,
        )

    _pipeline = VideoUnderstandingPipeline(
        registry,
        router,
        max_turns=cfg.max_turns,
        video_llm=_video_llm,
    )


# ---------------------------------------------------------------------------
# Trace → HTML formatting
# ---------------------------------------------------------------------------

_CSS_STYLES = """
<style>
.react-trace {
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 14px;
    line-height: 1.6;
}
.react-turn {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    margin-bottom: 12px;
    overflow: hidden;
}
.react-turn-header {
    background: #f5f5f5;
    padding: 8px 14px;
    font-weight: 700;
    font-size: 14px;
    color: #333;
    border-bottom: 1px solid #e0e0e0;
}
.react-thought {
    background: #eef4ff;
    border-left: 4px solid #4a90d9;
    padding: 10px 14px;
    margin: 0;
    color: #1a3a5c;
}
.react-action {
    background: #fff0f0;
    border-left: 4px solid #d94a4a;
    padding: 10px 14px;
    margin: 0;
    color: #5c1a1a;
}
.react-action code {
    background: #f8d7da;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 600;
}
.react-observation {
    background: #f0fff4;
    border-left: 4px solid #4ad94a;
    padding: 10px 14px;
    margin: 0;
    color: #1a5c1a;
    white-space: pre-wrap;
    word-break: break-word;
}
.react-finish {
    background: #f5f0ff;
    border-left: 4px solid #7c4ad9;
    padding: 10px 14px;
    margin: 0;
    color: #3a1a5c;
    font-weight: 600;
}
.react-label {
    font-weight: 700;
    margin-right: 6px;
}
</style>
"""


def _format_trace_html(trace: ReasoningTrace) -> str:
    """Render the full ReasoningTrace as styled HTML."""
    if not trace.steps:
        return "<p>No reasoning steps were executed.</p>"

    parts = [_CSS_STYLES, '<div class="react-trace">']

    for step in trace.steps:
        d = step.decision
        parts.append('<div class="react-turn">')
        parts.append(
            f'<div class="react-turn-header">Turn {step.step}</div>'
        )

        # Thought
        thought_text = html.escape(d.thought) if d.thought else "<em>No explicit thought</em>"
        parts.append(
            f'<div class="react-thought">'
            f'<span class="react-label">Thought:</span> {thought_text}'
            f"</div>"
        )

        # Action
        if d.action == ActionType.CALL_SKILL:
            skill = html.escape(d.skill_name or "?")
            args_str = ""
            if d.parameters:
                args_str = ", ".join(
                    f'{k}="{html.escape(str(v))}"' for k, v in d.parameters.items()
                )
                args_str = f" ({args_str})"
            parts.append(
                f'<div class="react-action">'
                f'<span class="react-label">Action:</span> '
                f"<code>CALL_SKILL({skill})</code>{args_str}"
                f"</div>"
            )

            # Observation
            if step.response:
                obs = html.escape(step.response.summary)
                parts.append(
                    f'<div class="react-observation">'
                    f'<span class="react-label">Observation:</span> {obs}'
                    f"</div>"
                )
        elif d.action == ActionType.FINISH:
            parts.append(
                '<div class="react-finish">'
                '<span class="react-label">Action:</span> FINISH'
                "</div>"
            )

        parts.append("</div>")  # close react-turn

    parts.append("</div>")  # close react-trace
    return "\n".join(parts)


def _format_summary_md(trace: ReasoningTrace) -> str:
    """One-line Markdown summary of the executed skills."""
    executed = trace.executed_skills
    n_turns = len(trace.steps)
    if not executed:
        return f"Completed in {n_turns} turn(s) — no skills executed."
    return (
        f"Completed in {n_turns} turn(s) — "
        f"executed: **{', '.join(executed)}**"
    )


# ---------------------------------------------------------------------------
# Prediction function
# ---------------------------------------------------------------------------

def predict(
    video_path: str | None,
    question: str,
    max_turns: int,
) -> tuple[list[dict], str, str]:
    """
    Run the Skill-MoE pipeline and return:
        (chatbot_messages, trace_html, summary_md)
    """
    if not video_path:
        return (
            [{"role": "assistant", "content": "Please upload a video first."}],
            "",
            "",
        )
    if not question.strip():
        return (
            [{"role": "assistant", "content": "Please enter a question."}],
            "",
            "",
        )

    assert _pipeline is not None, "Pipeline not initialised"
    assert _cfg is not None, "Config not loaded"

    # Allow per-request max_turns override.
    _pipeline.max_turns = int(max_turns)

    request = SkillRequest(question=question.strip(), video_path=video_path)

    try:
        trace = _pipeline.run_trace(request)
    except Exception:
        tb = traceback.format_exc()
        return (
            [{"role": "assistant", "content": f"Pipeline error:\n```\n{tb}\n```"}],
            "",
            "",
        )

    # Generate final answer.
    if trace.final_answer:
        final_answer = trace.final_answer
    else:
        try:
            final_answer = answer(
                question,
                trace.responses,
                trace=trace,
                video_path=video_path,
                video_llm=_video_llm,
                max_tokens=_cfg.answerer.max_tokens,
            )
        except Exception:
            final_answer = (
                "Error generating final answer. Skill outputs:\n"
                + "\n".join(r.summary for r in trace.responses)
            )

    # Build chatbot messages.
    messages: list[dict] = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": final_answer},
    ]

    trace_html = _format_trace_html(trace)
    summary_md = _format_summary_md(trace)

    return messages, trace_html, summary_md


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Skill-MoE: Video Understanding",
        theme=gr.themes.Soft(),
        css="""
        .header-text { text-align: center; margin-bottom: 4px; }
        .subheader  { text-align: center; color: #666; margin-top: 0; font-size: 15px; }
        """,
    ) as app:

        gr.Markdown(
            "<h1 class='header-text'>Skill-MoE</h1>"
            "<p class='subheader'>"
            "Mixture-of-Experts Video Understanding"
            "</p>"
        )

        with gr.Row():
            # ---- Left column: inputs ----
            with gr.Column(scale=1):
                video_input = gr.Video(label="Upload Video")
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="e.g. What is the license plate number?",
                    lines=2,
                )
                max_turns_slider = gr.Slider(
                    minimum=1,
                    maximum=8,
                    value=3,
                    step=1,
                    label="Max Reasoning Turns",
                )
                submit_btn = gr.Button("Ask", variant="primary", size="lg")

                gr.Markdown("### Pipeline Summary")
                summary_output = gr.Markdown(elem_id="summary-output")

            # ---- Right column: outputs ----
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    type="messages",
                    height=280,
                )

                with gr.Accordion(
                    "Reasoning Trace (Chain-of-Thought)", open=True
                ):
                    trace_output = gr.HTML()

        # ---- Examples ----
        gr.Examples(
            examples=[
                ["What is happening in this video?", 3],
                ["How many people are in the video?", 3],
                ["What did the person say?", 2],
                ["What is the license plate number?", 3],
                ["What text appears on screen?", 2],
            ],
            inputs=[question_input, max_turns_slider],
            label="Example Questions",
        )

        # ---- Event binding ----
        submit_btn.click(
            fn=predict,
            inputs=[video_input, question_input, max_turns_slider],
            outputs=[chatbot, trace_output, summary_output],
        )
        question_input.submit(
            fn=predict,
            inputs=[video_input, question_input, max_turns_slider],
            outputs=[chatbot, trace_output, summary_output],
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Skill-MoE Web Demo")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    parser.add_argument("--no-video-llm", action="store_true", help="Skip loading video LLM")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print("Initialising Skill-MoE pipeline ...")
    _init_pipeline(cfg, skip_video_llm=args.no_video_llm)
    print("Pipeline ready.")

    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
