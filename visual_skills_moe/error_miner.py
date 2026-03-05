#!/usr/bin/env python3
"""
error_miner.py — Automatic failure taxonomy for VideoLLM benchmark logs.

Usage
-----
    python error_miner.py benchmarks/log_*.json
    python error_miner.py benchmarks/log_20260304_001559.json --max-errors 300
    python error_miner.py benchmarks/log_*.json --output report.md --batch-size 15

Taxonomy categories
-------------------
    A  Temporal Reasoning Loss       — time order / sequence / causality failure
    B  Spatial/Relationship Halluc.  — object positions / counts / relationships
    C  Fine-grained Detail Miss      — tiny text, brief frames, small objects
    D  Skill Misdirection            — tool evidence misled the VLM
    E  VideoLLM Knowledge Deficit    — world-knowledge outside the video frame
    U  Unclassified                  — LLM could not determine a clear category
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import textwrap
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Taxonomy definition
# ---------------------------------------------------------------------------

CATEGORIES: dict[str, str] = {
    "A": "Temporal Reasoning Loss",
    "B": "Spatial / Relationship Hallucination",
    "C": "Fine-grained Detail Miss",
    "D": "Skill Misdirection",
    "E": "VideoLLM Knowledge Deficit",
    "U": "Unclassified",
}

CAT_DESCRIPTIONS: dict[str, str] = {
    "A": (
        "The question requires understanding time order, event sequence, causal chain, "
        "or 'before/after/first/last/at the end' relationships that the model got wrong."
    ),
    "B": (
        "The model confused object positions (left/right/behind), misidentified visual "
        "relationships between multiple objects/people, or hallucinated spatial context."
    ),
    "C": (
        "The correct answer requires reading tiny on-screen text, noticing a brief "
        "transient frame, identifying a small or occluded object, or fine-grained "
        "color/shape discrimination that the globally-downsampled video missed."
    ),
    "D": (
        "A skill tool was invoked and its output (even if plausible-sounding) actively "
        "changed the model from a correct initial answer to a wrong final answer, or "
        "injected misleading evidence into an otherwise solvable question."
    ),
    "E": (
        "The question tests factual/encyclopedic world knowledge, domain expertise, or "
        "common-sense facts that are not directly observable from the video frames — "
        "the model answered incorrectly due to knowledge gaps."
    ),
    "U": "Could not be confidently assigned to a single category by the classifier.",
}

# ---------------------------------------------------------------------------
# LLM client (mirrors skill_moe/llm_clients.py; no import to stay standalone)
# ---------------------------------------------------------------------------

def _llm_complete(prompt: str, max_tokens: int = 200) -> str:
    """Call gpt-4o-mini via litellm. Returns raw text."""
    import litellm  # type: ignore

    api_base = os.getenv("OPENAI_BASE_URL")
    api_key  = os.getenv("OPENAI_API_KEY", "sk-placeholder")
    model    = os.getenv("ERROR_MINER_MODEL", "gpt-4o-mini")
    litellm_model = f"openai/{model}" if api_base else model

    resp = litellm.completion(
        model=litellm_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        api_base=api_base,
        api_key=api_key,
        timeout=30,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_results(log_paths: list[str]) -> tuple[list[dict], dict]:
    """Load and merge results from one or more log JSON files."""
    all_results: list[dict] = []
    meta: dict[str, Any] = {}

    for path in log_paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if not meta:
            meta = {
                "log_path":  path,
                "run":       data.get("run", {}),
                "summary":   data.get("summary", {}),
            }
        # Support both {"results": [...]} and bare [...]
        results = data.get("results", data) if isinstance(data, dict) else data
        if isinstance(results, list):
            all_results.extend(results)
        elif isinstance(results, dict):
            # keyed by id
            all_results.extend(results.values())

    return all_results, meta


def _filter_wrong(results: list[dict]) -> list[dict]:
    return [r for r in results if int(r.get("score", 1)) == 0 and not r.get("error")]


def _format_options(options: Any) -> str:
    if isinstance(options, list):
        return "  ".join(str(o).strip() for o in options)
    return str(options or "").strip()


def _question_stem(question: str) -> str:
    """Strip trailing MCQ options appended after the question stem."""
    return re.split(r"\n\s*[A-D]\.", question or "")[0].strip()


# ---------------------------------------------------------------------------
# Batch LLM classification
# ---------------------------------------------------------------------------

_CLASSIFY_PROMPT = """\
You are an expert in video-language model failure analysis.

Classify each VideoLLM error into EXACTLY ONE category:

  A  Temporal Reasoning Loss       — time order / sequence / causality (first/last/before/after/when)
  B  Spatial/Relationship Halluc.  — confused object positions, counts, or visual relationships
  C  Fine-grained Detail Miss      — needs tiny text, brief frame, small object, color nuance
  D  Skill Misdirection            — tool/skill evidence is available but misled the model
  E  VideoLLM Knowledge Deficit    — factual / world-knowledge question, not solvable from video alone

Rules:
- If skills were used AND the initial answer was correct but the final answer is wrong, ALWAYS choose D.
- For each case output ONLY the index and letter, one per line, e.g.:  1: A
- Do not explain. Do not output anything else.

Cases:
{cases}"""

_CASE_TEMPLATE = """\
[{idx}] Cat: {category}  Skills: {skills}  Init-correct: {init_ok}
  Q: {stem}
  Opts: {opts}
  Correct={truth}  Predicted={pred}"""


def _build_classify_prompt(batch: list[dict]) -> str:
    case_blocks = []
    for i, r in enumerate(batch, 1):
        init_ok  = "YES" if int(r.get("initial_score", 0)) == 1 else "NO"
        skills   = r.get("skills_used", "") or "none"
        case_blocks.append(_CASE_TEMPLATE.format(
            idx      = i,
            category = r.get("category", "?"),
            skills   = skills,
            init_ok  = init_ok,
            stem     = _question_stem(r.get("question", ""))[:200],
            opts     = _format_options(r.get("options", ""))[:180],
            truth    = r.get("ground_truth", "?"),
            pred     = r.get("prediction", "?")[:30],
        ))
    return _CLASSIFY_PROMPT.format(cases="\n\n".join(case_blocks))


def _parse_labels(raw: str, batch_size: int) -> list[str]:
    """Extract category letters from LLM response.

    Handles formats like:  "1: A"  "Case 2: B"  "3. E"  "4 - D"  "[5] C"
    """
    labels: dict[int, str] = {}
    for line in raw.splitlines():
        # Match a digit (possibly preceded by non-digit word like "Case") then separator then letter
        m = re.search(r"\b(\d+)\b[^\w]*([A-Ea-eu])\b", line)
        if m:
            idx    = int(m.group(1))
            letter = m.group(2).upper()
            if letter not in CATEGORIES:
                letter = "U"
            if 1 <= idx <= batch_size:
                labels[idx] = letter
    return [labels.get(i, "U") for i in range(1, batch_size + 1)]


def classify_batch(batch: list[dict]) -> list[str]:
    """Classify a batch of wrong cases; returns list of category letters."""
    prompt = _build_classify_prompt(batch)
    try:
        raw = _llm_complete(prompt, max_tokens=len(batch) * 8 + 20)
        labels = _parse_labels(raw, len(batch))
        return labels
    except Exception as exc:
        print(f"  [WARN] Batch classification failed: {exc!r}; marking as U", file=sys.stderr)
        return ["U"] * len(batch)


def classify_all(
    wrong: list[dict],
    batch_size: int = 20,
    verbose: bool = True,
) -> list[tuple[dict, str]]:
    """Classify all wrong cases in batches. Returns [(case, label), ...]."""
    # Pre-assign D for clear Skill Misdirection cases (saves LLM calls).
    def _is_clear_skill_misdirection(r: dict) -> bool:
        return (
            int(r.get("initial_score", 0)) == 1
            and int(r.get("score", 1)) == 0
            and bool(r.get("skills_used", "").strip())
        )

    results: list[tuple[dict, str]] = []
    pre_assigned: list[tuple[dict, str]] = []
    needs_llm: list[dict] = []

    for r in wrong:
        if _is_clear_skill_misdirection(r):
            pre_assigned.append((r, "D"))
        else:
            needs_llm.append(r)

    if verbose and pre_assigned:
        print(f"  Pre-assigned D (Skill Misdirection): {len(pre_assigned)} cases")

    # Batch LLM classification for the rest.
    n_batches = (len(needs_llm) + batch_size - 1) // batch_size
    for b_idx in range(n_batches):
        batch = needs_llm[b_idx * batch_size : (b_idx + 1) * batch_size]
        if verbose:
            done  = b_idx * batch_size
            total = len(needs_llm)
            print(f"  Classifying batch {b_idx + 1}/{n_batches} ({done}–{done + len(batch)}/{total})", end="\r")
        labels = classify_batch(batch)
        for case, label in zip(batch, labels):
            results.append((case, label))

    if verbose:
        print()  # newline after \r progress

    results.extend(pre_assigned)
    # Sort back to original order by id
    id_order = {r["id"]: i for i, r in enumerate(wrong)}
    results.sort(key=lambda x: id_order.get(x[0].get("id", ""), 999999))
    return results


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------

_WRAP = 90


def _md_wrap(text: str, prefix: str = "> ") -> str:
    lines = []
    for para in text.splitlines():
        wrapped = textwrap.fill(para, width=_WRAP - len(prefix), break_long_words=False)
        for line in wrapped.splitlines():
            lines.append(prefix + line)
    return "\n".join(lines)


def _case_block(r: dict, label: str) -> str:
    """Format one wrong case for the Markdown report."""
    qid      = r.get("id", "?")
    cat      = r.get("category", "?")
    q        = _question_stem(r.get("question", ""))
    opts     = r.get("options", [])
    truth    = r.get("ground_truth", "?")
    pred     = r.get("prediction", "?")
    skills   = r.get("skills_used", "") or "_none_"
    init_ok  = "✓" if int(r.get("initial_score", 0)) == 1 else "✗"
    final_ok = "✓" if int(r.get("score", 0)) == 1 else "✗"

    opts_str = ""
    if isinstance(opts, list):
        opts_str = "  ".join(str(o).strip() for o in opts)
    else:
        opts_str = str(opts)

    lines = [
        f"**`{qid}`** — _{cat}_",
        f"> **Q:** {q}",
        f"> **Opts:** {opts_str[:200]}",
        f"> **Correct:** `{truth}` | **Predicted:** `{pred[:40]}`",
        f"> **Skills used:** `{skills}` | Init {init_ok} → Final {final_ok}",
    ]
    return "\n".join(lines)


def build_report(
    classified: list[tuple[dict, str]],
    meta: dict,
    n_examples: int = 2,
    seed: int = 42,
) -> str:
    rng = random.Random(seed)

    # Group by category
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for case, label in classified:
        by_cat[label].append(case)

    total_errors   = len(classified)
    total_cases    = meta.get("summary", {}).get("total", "?")
    accuracy       = meta.get("summary", {}).get("accuracy_pct", "?")
    log_path       = meta.get("log_path", "?")
    run_ts         = meta.get("run", {}).get("timestamp", "?")
    dataset        = meta.get("run", {}).get("dataset", "?")

    lines: list[str] = []

    # ── Header ───────────────────────────────────────────────────────────────
    lines += [
        "# VideoLLM Error Taxonomy Report",
        "",
        f"- **Log:** `{log_path}`",
        f"- **Run timestamp:** `{run_ts}`",
        f"- **Dataset:** `{dataset}`",
        f"- **Total cases:** {total_cases}  |  "
        f"**Wrong cases analyzed:** {total_errors}  |  "
        f"**Accuracy:** {accuracy}%",
        f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    # ── Summary table ─────────────────────────────────────────────────────────
    lines += [
        "## Summary Table",
        "",
        "| # | Code | Category | Count | % of Errors |",
        "|---|------|----------|------:|------------:|",
    ]
    sorted_cats = sorted(CATEGORIES.keys(), key=lambda c: -len(by_cat.get(c, [])))
    for rank, code in enumerate(sorted_cats, 1):
        name  = CATEGORIES[code]
        count = len(by_cat.get(code, []))
        pct   = 100 * count / max(1, total_errors)
        lines.append(f"| {rank} | **{code}** | {name} | {count} | {pct:.1f}% |")
    lines.append("")

    # ── Skill-specific breakdown for category D ───────────────────────────────
    d_cases = by_cat.get("D", [])
    if d_cases:
        skill_counts: Counter = Counter()
        for r in d_cases:
            for s in (r.get("skills_used") or "").split(","):
                s = s.strip()
                if s:
                    skill_counts[s] += 1
        lines += [
            "### D — Skills involved in misdirection",
            "",
            "| Skill | Hurt Cases |",
            "|-------|----------:|",
        ]
        for skill, cnt in skill_counts.most_common():
            lines.append(f"| `{skill}` | {cnt} |")
        lines.append("")

    # ── VLM category breakdown ────────────────────────────────────────────────
    lines += ["## Video-MME Category Breakdown", ""]
    all_cats = sorted({r.get("category", "?") for r, _ in classified})
    cat_data: dict[str, Counter] = defaultdict(Counter)
    for r, label in classified:
        cat_data[r.get("category", "?")][label] += 1

    header = "| Video-MME Category | Errors | " + " | ".join(CATEGORIES.keys()) + " |"
    sep    = "|---|---:|" + "|---:" * len(CATEGORIES) + "|"
    lines += [header, sep]
    for vc in sorted(all_cats, key=lambda c: -sum(cat_data[c].values())):
        total_vc = sum(cat_data[vc].values())
        cats_str = " | ".join(str(cat_data[vc].get(c, 0)) for c in CATEGORIES.keys())
        lines.append(f"| {vc} | {total_vc} | {cats_str} |")
    lines.append("")

    # ── Per-category examples ─────────────────────────────────────────────────
    lines += ["## Error Cases by Category", ""]

    for code in CATEGORIES.keys():
        name  = CATEGORIES[code]
        cases = by_cat.get(code, [])
        count = len(cases)
        pct   = 100 * count / max(1, total_errors)

        lines += [
            f"---",
            f"### {code}: {name} &nbsp; `{count} cases ({pct:.1f}%)`",
            "",
            f"> _{CAT_DESCRIPTIONS[code]}_",
            "",
        ]

        if not cases:
            lines += ["_No cases in this category._", ""]
            continue

        # Random sample of n_examples
        sample = rng.sample(cases, min(n_examples, count))
        for i, r in enumerate(sample, 1):
            lines += [
                f"**Example {i}**",
                "",
                _case_block(r, code),
                "",
            ]

    # ── Skill hit / hurt overview ─────────────────────────────────────────────
    lines += [
        "---",
        "## Skill Performance Summary",
        "",
        "Skill cases drawn from the analyzed error set.",
        "",
        "| Skill | Total errors with skill | D (misdirected) | Other failure |",
        "|-------|------------------------:|----------------:|---------------:|",
    ]
    skill_errors:   Counter = Counter()
    skill_d_errors: Counter = Counter()
    for r, label in classified:
        for s in (r.get("skills_used") or "").split(","):
            s = s.strip()
            if not s:
                continue
            skill_errors[s] += 1
            if label == "D":
                skill_d_errors[s] += 1
    for skill, total in skill_errors.most_common():
        d_cnt   = skill_d_errors.get(skill, 0)
        other   = total - d_cnt
        lines.append(f"| `{skill}` | {total} | {d_cnt} | {other} |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automatic failure taxonomy for VideoLLM benchmark logs."
    )
    parser.add_argument(
        "logs",
        nargs="+",
        help="Path(s) to benchmark log JSON file(s).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Write Markdown report to this file (default: print to stdout).",
    )
    parser.add_argument(
        "--max-errors", "-n",
        type=int,
        default=None,
        help="Randomly sample at most N wrong cases (useful for quick analysis).",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=20,
        help="Number of cases per LLM classification call (default: 20).",
    )
    parser.add_argument(
        "--examples", "-e",
        type=int,
        default=2,
        help="Number of example cases to show per category (default: 2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42).",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM classification; use rule-based pre-labelling only.",
    )
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"Loading {len(args.logs)} log file(s)...", file=sys.stderr)
    results, meta = _load_results(args.logs)
    wrong = _filter_wrong(results)
    total = len(results)

    print(
        f"  Total cases: {total}  |  Wrong: {len(wrong)}  |  "
        f"Accuracy: {100 * (total - len(wrong)) / max(1, total):.1f}%",
        file=sys.stderr,
    )

    # ── Optional sampling ─────────────────────────────────────────────────────
    if args.max_errors and len(wrong) > args.max_errors:
        rng = random.Random(args.seed)
        wrong = rng.sample(wrong, args.max_errors)
        print(f"  Sampled {args.max_errors} wrong cases for classification.", file=sys.stderr)

    if not wrong:
        print("No wrong cases found. Exiting.", file=sys.stderr)
        sys.exit(0)

    # ── Classification ────────────────────────────────────────────────────────
    if args.no_llm:
        print("Skipping LLM classification (--no-llm). Pre-assigning D where applicable.", file=sys.stderr)
        classified: list[tuple[dict, str]] = []
        for r in wrong:
            if int(r.get("initial_score", 0)) == 1 and bool(r.get("skills_used", "").strip()):
                classified.append((r, "D"))
            else:
                classified.append((r, "U"))
    else:
        print(f"Classifying {len(wrong)} cases (batch_size={args.batch_size})...", file=sys.stderr)
        classified = classify_all(wrong, batch_size=args.batch_size)

    # ── Build report ─────────────────────────────────────────────────────────
    print("Building report...", file=sys.stderr)
    report = build_report(classified, meta, n_examples=args.examples, seed=args.seed)

    # ── Output ───────────────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(report, encoding="utf-8")
        print(f"Report written to: {out_path}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
