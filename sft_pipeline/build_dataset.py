"""
Collect passing per-step reasoning into a single interleaved SFT dataset.

Each record is one puzzle: the question image, then one (image, reasoning) pair
per roll, then the final answer. Puzzles with any failing step are excluded by
default so the dataset carries only verified chains.

Usage:
    python sft_pipeline/build_dataset.py --output-dir output
    python sft_pipeline/build_dataset.py --output-dir output --allow-partial
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gemini_common import iter_puzzle_dirs  # noqa: E402


def step_passed(record: Dict[str, Any]) -> bool:
    verdict = record.get("verdict")
    if verdict is not None:
        return bool(verdict.get("passed"))
    # No judge verdict yet — fall back to the deterministic checks
    return bool(record.get("faces_correct")) and not record.get("hint_leak")


def build_record(
    puzzle_dir: Path, allow_partial: bool
) -> Optional[Dict[str, Any]]:
    metadata_path = puzzle_dir / "metadata.json"
    reasoning_path = puzzle_dir / "cot_reasoning.json"
    if not (metadata_path.exists() and reasoning_path.exists()):
        return None

    metadata = json.loads(metadata_path.read_text())
    reasoning = json.loads(reasoning_path.read_text())
    by_key = {(r.get("path"), r["step"]): r for r in reasoning["steps"]}

    steps: List[Dict[str, Any]] = []
    for step in metadata["steps"]:
        record = by_key.get((step.get("path"), step["step"]))
        if record is None or not step_passed(record):
            if not allow_partial:
                return None
            continue

        idx = step["step"]
        tag = step.get("path")
        image = (
            f"cot_{tag}_{idx - 1:02d}.png" if tag else f"cot_{idx - 1:02d}.png"
        )
        steps.append(
            {
                "step": idx,
                "path": tag,
                "direction": step["direction"],
                "direction_name": step["direction_name"],
                "image": image,
                "reasoning": record["reasoning"],
                "die_after": step["die_after"],
                "bottom_face": step["bottom_face"],
                "running_bottom_sum": step["running_bottom_sum"],
            }
        )

    if not steps:
        return None

    return {
        "puzzle_dir": str(puzzle_dir),
        "puzzle_id": metadata["puzzle_id"],
        "variant": metadata["variant"],
        "level": metadata["level"],
        "question": metadata["question"],
        "question_image": metadata["initial_state_image"],
        "board": metadata["board"],
        "start": metadata["start"],
        "initial_die": metadata["initial_die"],
        "steps": steps,
        "answer": metadata["answer"],
        "complete": len(steps) == len(metadata["steps"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the SFT dataset")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--variant", choices=["top", "sum", "two"], default=None)
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--dataset", default="dataset.jsonl")
    ap.add_argument("--allow-partial", action="store_true",
                    help="Include puzzles where only some steps passed")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    records = []
    skipped = 0

    for puzzle_dir in iter_puzzle_dirs(out_dir, args.variant, args.level):
        record = build_record(puzzle_dir, args.allow_partial)
        if record is None:
            skipped += 1
            continue
        records.append(record)

    dataset_path = Path(args.dataset)
    with open(dataset_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    by_variant: Dict[str, int] = {}
    by_level: Dict[int, int] = {}
    total_steps = 0
    for r in records:
        by_variant[r["variant"]] = by_variant.get(r["variant"], 0) + 1
        by_level[r["level"]] = by_level.get(r["level"], 0) + 1
        total_steps += len(r["steps"])

    print(f"Wrote {len(records)} puzzles ({total_steps} steps) to {dataset_path}")
    print(f"Skipped {skipped} puzzles with missing or failing steps")
    if by_variant:
        print("By variant: " + ", ".join(f"{k}={v}" for k, v in sorted(by_variant.items())))
        print("By level:   " + ", ".join(f"L{k}={v}" for k, v in sorted(by_level.items())))


if __name__ == "__main__":
    main()
