"""
Evaluate model responses against ground truth.

Expects a JSON file of responses in the MentisOculi format:

    [
      {"puzzle_id": 1, "level": 3, "variant": "top", "response": "...", ...},
      ...
    ]

The answer is parsed out of the response text as JSON, matching the output
format the prompts ask for. Accuracy is reported overall and per level.

Usage:
    python evaluate_responses.py --responses responses/gpt-4o/simple/top/level_03/responses_0.json
    python evaluate_responses.py --responses results.json --dataset output
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Chance performance per variant and level, for reporting alongside accuracy.
CHANCE_PERFORMANCE = {"top": 1 / 6, "sum": 1 / 6, "two": 1 / 3}


def extract_answer(text: str) -> Optional[Any]:
    """
    Pull the answer out of a response.

    Tries, in order: a fenced JSON block, any {"answer": ...} object, an
    <answer></answer> tag (the MIRA convention), and finally a bare trailing
    integer.
    """
    if not isinstance(text, str):
        return None

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        try:
            return json.loads(fenced.group(1)).get("answer")
        except (json.JSONDecodeError, AttributeError):
            pass

    for match in re.finditer(r"\{[^{}]*\"answer\"\s*:.*?\}\s*\}?", text, re.S):
        try:
            return json.loads(match.group(0))["answer"]
        except (json.JSONDecodeError, KeyError):
            continue

    tagged = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.S)
    if tagged:
        inner = tagged.group(1).strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            if inner.isdigit():
                return int(inner)
            return inner

    trailing = re.findall(r"\b(\d+)\b", text)
    if trailing:
        return int(trailing[-1])
    return None


def is_correct(predicted: Any, expected: Any, variant: str) -> bool:
    """Compare a predicted answer to ground truth for the given variant."""
    if predicted is None:
        return False

    if variant == "two":
        if isinstance(predicted, str):
            # A bare "red"/"black" is only a partial answer; the total is required
            return False
        if not isinstance(predicted, dict):
            return False
        winner = str(predicted.get("winner", "")).strip().lower()
        try:
            total = int(predicted.get("total"))
        except (TypeError, ValueError):
            return False
        return winner == expected["winner"] and total == expected["total"]

    try:
        return int(predicted) == int(expected)
    except (TypeError, ValueError):
        return False


def load_ground_truth(dataset_dir: Path) -> Dict[tuple, Dict[str, Any]]:
    """Index every puzzle's answer by (variant, level, puzzle_id)."""
    truth: Dict[tuple, Dict[str, Any]] = {}
    for variant_dir in dataset_dir.iterdir():
        if not variant_dir.is_dir():
            continue
        for level_dir in sorted(variant_dir.glob("level_*")):
            for puzzle_dir in sorted(level_dir.glob("puzzle_*")):
                meta_path = puzzle_dir / "metadata.json"
                if not meta_path.exists():
                    continue
                meta = json.loads(meta_path.read_text())
                key = (meta["variant"], meta["level"], meta["puzzle_id"])
                truth[key] = meta
    return truth


def evaluate(
    responses: List[Dict[str, Any]], truth: Dict[tuple, Dict[str, Any]]
) -> Dict[str, Any]:
    per_level: Dict[tuple, List[bool]] = defaultdict(list)
    rows = []

    for entry in responses:
        variant = entry.get("variant")
        level = entry.get("level")
        puzzle_id = entry.get("puzzle_id")

        meta = truth.get((variant, level, puzzle_id)) if truth else None
        expected = entry.get("expected", meta["answer"] if meta else None)
        if expected is None:
            continue

        text = entry.get("response") or entry.get("final_text") or ""
        predicted = entry.get("predicted")
        if predicted is None:
            predicted = extract_answer(text)

        correct = is_correct(predicted, expected, variant)
        per_level[(variant, level)].append(correct)
        rows.append(
            {
                "variant": variant,
                "level": level,
                "puzzle_id": puzzle_id,
                "predicted": predicted,
                "expected": expected,
                "correct": correct,
            }
        )

    summary = {}
    for (variant, level), flags in sorted(per_level.items()):
        summary[f"{variant}/level_{level:02d}"] = {
            "n": len(flags),
            "correct": sum(flags),
            "accuracy": sum(flags) / len(flags) if flags else 0.0,
            "chance": CHANCE_PERFORMANCE.get(variant, 0.0),
        }

    all_flags = [r["correct"] for r in rows]
    return {
        "overall": {
            "n": len(all_flags),
            "correct": sum(all_flags),
            "accuracy": sum(all_flags) / len(all_flags) if all_flags else 0.0,
        },
        "per_level": summary,
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Rolling Dice responses")
    ap.add_argument("--responses", required=True,
                    help="JSON file of model responses")
    ap.add_argument("--dataset", default=None,
                    help="Dataset root, for looking up ground truth")
    ap.add_argument("--report", default=None,
                    help="Optional path to write the full report as JSON")
    args = ap.parse_args()

    responses = json.loads(Path(args.responses).read_text())
    if isinstance(responses, dict):
        responses = responses.get("responses", [])

    truth = load_ground_truth(Path(args.dataset)) if args.dataset else {}
    report = evaluate(responses, truth)

    overall = report["overall"]
    print("=" * 60)
    print("Rolling Dice - Evaluation")
    print("=" * 60)
    for key, stats in report["per_level"].items():
        print(
            f"  {key}: {stats['correct']}/{stats['n']} = {stats['accuracy']:.1%} "
            f"(chance {stats['chance']:.1%})"
        )
    print("-" * 60)
    print(f"  Overall: {overall['correct']}/{overall['n']} = {overall['accuracy']:.1%}")
    print("=" * 60)

    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
