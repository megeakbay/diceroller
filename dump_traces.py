"""
Write each response's reasoning out as readable files, one per puzzle.

The results JSON holds everything, but it is awkward to read: the reasoning and
the answer are long strings embedded in one line. This unpacks a run into a
directory tree so the working can be read directly, and diffed between runs.

Layout:
    traces/<run-name>/
        summary.txt                 per-puzzle verdicts, all in one place
        puzzle_0001/
            prompt.txt              exactly what the model was sent
            reasoning.txt           API-returned reasoning, if any
            answer.txt              the visible response
            verdict.txt             ground truth vs stated states, step by step

Usage:
    python dump_traces.py --results text_only_2p5flash_think.json
    python dump_traces.py --results claude_opus5.json --out traces/claude
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from verify_steps import FACE_CODES, parse_states, verify_row

# Render states in the order a reader expects, not dict order.
DISPLAY_ORDER = [
    ("top", "T"),
    ("bottom", "B"),
    ("north", "UL"),
    ("east", "UR"),
    ("west", "DL"),
    ("south", "DR"),
]


def format_state(state: Dict[str, int]) -> str:
    return "  ".join(f"{label}={state[key]}" for key, label in DISPLAY_ORDER)


def write_verdict(path: Path, row: Dict[str, Any], check: Dict[str, Any]) -> None:
    """Lay the model's stated states next to ground truth, step by step."""
    meta = json.loads((Path(row["puzzle_dir"]) / "metadata.json").read_text())
    truths = [t["die"] for t in meta["trace"]]
    stated = parse_states(
        "\n".join(filter(None, [row.get("reasoning"), row.get("response")]))
    )

    lines: List[str] = []
    lines.append(f"puzzle {row['puzzle_id']}  (level {row['level']}, variant {row['variant']})")
    lines.append(f"model: {row.get('model')}")
    lines.append("")
    lines.append(f"path:     {' -> '.join(meta['path'])}")
    lines.append(f"expected: {row['expected']}")
    lines.append(f"answered: {row['predicted']}"
                 f"   [{'correct' if check['answer_correct'] else 'WRONG'}]")
    lines.append("")

    lines.append("GROUND TRUTH")
    for i, t in enumerate(truths):
        lines.append(f"  step {i}: {format_state(t)}")
    lines.append("")

    if stated:
        lines.append(f"STATES THE MODEL STATED ({len(stated)} found)")
        for i, s in enumerate(stated):
            where = truths.index(s) if s in truths else None
            tag = f"matches step {where}" if where is not None else "NOT ON TRAJECTORY"
            lines.append(f"  #{i}: {format_state(s)}   <- {tag}")
    else:
        lines.append("STATES THE MODEL STATED: none")
        if check["narrated_without_values"]:
            lines.append(
                "  The reasoning summary describes the process but never gives\n"
                "  face values, so the derivation cannot be checked."
            )
        else:
            lines.append("  The response showed no working.")
    lines.append("")

    if check["reasoning_sound"]:
        verdict = "SOUND - every step stated, all on the true trajectory"
    elif check["off_trajectory"]:
        verdict = (
            f"UNSOUND - {len(check['off_trajectory'])} stated state(s) are not on "
            f"the true trajectory"
        )
    elif check["narrated_without_values"]:
        verdict = "UNVERIFIABLE - summary narrates the process but states no values"
    elif not check["has_trace"]:
        verdict = "UNVERIFIABLE - no working shown"
    else:
        verdict = (
            f"PARTIAL - {len(check['steps_hit_in_order'])}/{check['steps_expected']} "
            f"steps shown, all correct, derivation incomplete"
        )
    lines.append(f"VERDICT: {verdict}")

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Write reasoning traces out as files")
    ap.add_argument("--results", required=True, help="Results JSON from a rollout")
    ap.add_argument(
        "--out",
        default=None,
        help="Output directory (default: traces/<results filename stem>)",
    )
    args = ap.parse_args()

    results_path = Path(args.results)
    rows = json.loads(results_path.read_text())
    if isinstance(rows, dict):
        rows = rows.get("responses", rows.get("results", []))
    if not rows:
        raise SystemExit(f"No rows in {args.results}")

    out_dir = Path(args.out) if args.out else Path("traces") / results_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: List[str] = []
    summary.append(f"run: {results_path.name}")
    summary.append(f"model: {rows[0].get('model')}")
    if "thinking" in rows[0]:
        summary.append(f"thinking: {rows[0]['thinking']}")
    summary.append("")

    for row in rows:
        check = verify_row(row)
        pdir = out_dir / f"puzzle_{row['puzzle_id']:04d}"
        pdir.mkdir(exist_ok=True)

        (pdir / "prompt.txt").write_text(row.get("prompt", ""))
        (pdir / "answer.txt").write_text(row.get("response", ""))

        reasoning = row.get("reasoning") or ""
        if reasoning:
            (pdir / "reasoning.txt").write_text(reasoning)
        else:
            # Be explicit about why the file is absent rather than leaving a gap.
            (pdir / "reasoning.txt").write_text(
                "(the API returned no reasoning for this response)\n"
            )

        write_verdict(pdir / "verdict.txt", row, check)

        if check["reasoning_sound"]:
            tag = f"sound ({len(check['steps_hit_in_order'])}/{check['steps_expected']} steps)"
        elif check["off_trajectory"]:
            tag = f"UNSOUND ({len(check['off_trajectory'])} off-trajectory)"
        elif check["narrated_without_values"]:
            tag = "unverifiable (narrated, no values)"
        elif not check["has_trace"]:
            tag = "unverifiable (no working)"
        else:
            tag = f"partial ({len(check['steps_hit_in_order'])}/{check['steps_expected']} steps)"

        summary.append(
            f"puzzle {row['puzzle_id']}: "
            f"answer {'correct' if check['answer_correct'] else 'WRONG'} "
            f"(said {row['predicted']}, expected {row['expected']}) | {tag}"
        )
        print(f"  {pdir}")

    (out_dir / "summary.txt").write_text("\n".join(summary) + "\n")
    print(f"\nWrote {len(rows)} traces to {out_dir}/")
    print(f"Summary: {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
