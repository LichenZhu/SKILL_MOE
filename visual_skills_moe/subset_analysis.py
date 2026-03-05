#!/usr/bin/env python3
"""Subset analysis for Skill-MoE benchmark logs.

Filters benchmark results to cases where the router deployed at least one skill,
then reports baseline vs. Skill-MoE accuracy on that "Tool-Dependent" subset.

Usage:
    python subset_analysis.py benchmarks/log_YYYYMMDD_HHMMSS.json
    python subset_analysis.py benchmarks/log_*.json   # compare multiple runs
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def load_results(log_path: str) -> List[Dict]:
    with open(log_path) as f:
        data = json.load(f)
    return data["results"] if isinstance(data, dict) else data


def analyse(results: List[Dict], label: str = "") -> None:
    # ── Filter to skill-deployed cases ──────────────────────────────────────
    skill_cases = [
        r for r in results
        if r.get("skills_used", "") and str(r.get("skills_used", "")).strip()
    ]
    no_skill_cases = [r for r in results if r not in skill_cases]

    total = len(results)
    n_skill = len(skill_cases)
    n_no_skill = len(no_skill_cases)

    if n_skill == 0:
        print(f"[{label}] No skill-deployed cases found in {total} results.")
        return

    # ── Overall accuracy ────────────────────────────────────────────────────
    overall_init = sum(r["initial_score"] for r in results) / total
    overall_final = sum(r["score"] for r in results) / total

    # ── Skill-deployed subset ────────────────────────────────────────────────
    subset_init = sum(r["initial_score"] for r in skill_cases) / n_skill
    subset_final = sum(r["score"] for r in skill_cases) / n_skill
    subset_delta = subset_final - subset_init

    # Helped / hurt
    helped = [r for r in skill_cases if r["score"] > r["initial_score"]]
    hurt = [r for r in skill_cases if r["score"] < r["initial_score"]]
    neutral = [r for r in skill_cases if r["score"] == r["initial_score"]]

    # ── Per-skill breakdown ─────────────────────────────────────────────────
    skill_stats: Dict[str, Dict] = defaultdict(lambda: {
        "n": 0, "init_correct": 0, "final_correct": 0, "helped": 0, "hurt": 0
    })
    for r in skill_cases:
        for sk in str(r["skills_used"]).split(","):
            sk = sk.strip()
            if not sk:
                continue
            d = skill_stats[sk]
            d["n"] += 1
            d["init_correct"] += r["initial_score"]
            d["final_correct"] += r["score"]
            if r["score"] > r["initial_score"]:
                d["helped"] += 1
            elif r["score"] < r["initial_score"]:
                d["hurt"] += 1

    # ── Category breakdown for skill cases ──────────────────────────────────
    cat_stats: Dict[str, Dict] = defaultdict(lambda: {
        "n": 0, "init_correct": 0, "final_correct": 0
    })
    for r in skill_cases:
        cat = r.get("category", "Unknown")
        cat_stats[cat]["n"] += 1
        cat_stats[cat]["init_correct"] += r["initial_score"]
        cat_stats[cat]["final_correct"] += r["score"]

    # ── Print ────────────────────────────────────────────────────────────────
    title = f"  SUBSET ANALYSIS — {label}" if label else "  SUBSET ANALYSIS"
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)

    print(f"\n  Total cases in log:          {total}")
    print(f"  Cases WITH skills deployed:  {n_skill}  ({n_skill/total*100:.1f}%)")
    print(f"  Cases WITHOUT skills:        {n_no_skill}")

    print(f"\n  ── Global accuracy ──")
    print(f"  Baseline (no skills):   {overall_init*100:.1f}%  ({sum(r['initial_score'] for r in results)}/{total})")
    print(f"  Skill-MoE (final):      {overall_final*100:.1f}%  ({sum(r['score'] for r in results)}/{total})")
    print(f"  Net global delta:       {(overall_final - overall_init)*100:+.1f}%")

    print(f"\n  ── Tool-Dependent Subset ({n_skill} cases) ──")
    print(f"  Baseline VLM accuracy:  {subset_init*100:.1f}%  ({sum(r['initial_score'] for r in skill_cases)}/{n_skill})")
    print(f"  Skill-MoE accuracy:     {subset_final*100:.1f}%  ({sum(r['score'] for r in skill_cases)}/{n_skill})")
    print(f"  Net subset delta:       {subset_delta*100:+.1f}%  ← paper number")
    print(f"  Helped: {len(helped)}   Hurt: {len(hurt)}   Neutral: {len(neutral)}")

    print(f"\n  ── Per-skill accuracy (subset cases that used each skill) ──")
    print(f"  {'Skill':<28} {'N':>4}  {'Baseline':>9}  {'Skill-MoE':>9}  {'Delta':>7}  {'Help/Hurt'}")
    print(f"  {'-'*28} {'-'*4}  {'-'*9}  {'-'*9}  {'-'*7}  {'-'*9}")
    for sk, d in sorted(skill_stats.items(), key=lambda x: -x[1]["n"]):
        n = d["n"]
        base = d["init_correct"] / n * 100
        final = d["final_correct"] / n * 100
        delta = final - base
        print(f"  {sk:<28} {n:>4}  {base:>8.1f}%  {final:>8.1f}%  {delta:>+6.1f}%  +{d['helped']}/-{d['hurt']}")

    print(f"\n  ── Per-category accuracy (skill-deployed subset) ──")
    print(f"  {'Category':<26} {'N':>4}  {'Baseline':>9}  {'Skill-MoE':>9}  {'Delta':>7}")
    print(f"  {'-'*26} {'-'*4}  {'-'*9}  {'-'*9}  {'-'*7}")
    for cat, d in sorted(cat_stats.items(), key=lambda x: -x[1]["n"]):
        n = d["n"]
        base = d["init_correct"] / n * 100
        final = d["final_correct"] / n * 100
        delta = final - base
        print(f"  {cat:<26} {n:>4}  {base:>8.1f}%  {final:>8.1f}%  {delta:>+6.1f}%")

    print()
    print(f"  ★ Headline for paper: On the {n_skill}-case Tool-Dependent subset,")
    print(f"    Skill-MoE improves accuracy from {subset_init*100:.1f}% → {subset_final*100:.1f}%  ({subset_delta*100:+.1f}%)")
    print("=" * 72)
    print()

    # ── Helped / hurt case listing ───────────────────────────────────────────
    if helped:
        print(f"  Cases where skills HELPED ({len(helped)}):")
        for r in helped:
            print(f"    [{r['id']}] {r.get('category','')} GT={r['ground_truth']} "
                  f"init={r.get('initial_letter','?')} → {r['prediction'][:30]}  "
                  f"skills=[{r['skills_used']}]")
    if hurt:
        print(f"\n  Cases where skills HURT ({len(hurt)}):")
        for r in hurt:
            print(f"    [{r['id']}] {r.get('category','')} GT={r['ground_truth']} "
                  f"init={r.get('initial_letter','?')} → {r['prediction'][:30]}  "
                  f"skills=[{r['skills_used']}]")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Subset analysis for Skill-MoE logs")
    parser.add_argument("logs", nargs="+", help="Path(s) to benchmark log JSON file(s)")
    parser.add_argument("--skill", default=None,
                        help="Filter to cases that used this specific skill only")
    args = parser.parse_args()

    for log_path in args.logs:
        results = load_results(log_path)
        label = Path(log_path).stem

        if args.skill:
            results = [r for r in results if args.skill in str(r.get("skills_used", ""))]
            label += f" [skill={args.skill}]"

        analyse(results, label=label)


if __name__ == "__main__":
    main()
