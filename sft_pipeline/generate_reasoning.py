"""
Generate a reasoning text for every roll of every puzzle (Gemini).

For each step the model is shown the board before the roll, the board after it,
and the ground-truth face values, and asked to explain the roll as if it had
deduced the new orientation itself. Results are written to
cot_reasoning.json inside each puzzle directory.

Usage:
    python sft_pipeline/generate_reasoning.py --output-dir output
    python sft_pipeline/generate_reasoning.py --output-dir output --variant top
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generator import FACE_NAMES  # noqa: E402
from gemini_common import (  # noqa: E402
    DEFAULT_MODEL,
    generate_json,
    iter_puzzle_dirs,
    make_client,
    png_part,
    text_part,
)

PROMPT = """
You are analyzing one step of a dice-rolling puzzle. A standard die (opposite
faces sum to 7) is being tipped across a grid board, one cell per step.

You are given:
- The board before this roll, with the die on its current cell.
- The board after this roll.
- The direction rolled and the exact face values before and after.

Your task:
Write the reasoning for THIS step: explain which faces move where when the die
tips in this direction, and why the resulting top and bottom faces follow.

CRITICAL CONSTRAINTS:
- Write as if you deduced the new orientation yourself from the rules, working
  forward from the previous state.
- Reason about the die and the board only.
- NEVER use the words "image", "images", "hint", "given", "next state",
  "after state", "ground truth", "told", or any phrase referring to how many
  views you were shown or that the answer was supplied to you.
- Refer to the faces the way they are labelled above (top, bottom, and the four
  sides named by the direction they face) and give their values.
- Two to four sentences.

Respond EXACTLY with this JSON:
{
  "reasoning": "<your reasoning for this step>",
  "top": <the top face value after the roll>,
  "bottom": <the bottom face value after the roll>
}
"""

OCTAHEDRON_PROMPT = """
You are analyzing one step of a dice-rolling puzzle. An eight-faced die (a
regular octahedron, opposite faces sum to 9) is being rolled across a
triangular grid, one cell per step. Each cell has three neighbours, and the die
tips over one edge of the triangle it is resting on.

You are given:
- The board before this roll, with the die on its current cell.
- The board after this roll.
- The direction rolled and the exact face values before and after.

Your task:
Write the reasoning for THIS step: explain which face comes to rest on the
board when the die tips in this direction, and why the resulting bottom face
follows from the previous orientation.

CRITICAL CONSTRAINTS:
- Write as if you deduced the new orientation yourself from the rules, working
  forward from the previous state.
- Reason about the die and the board only.
- NEVER use the words "image", "images", "hint", "given", "next state",
  "after state", "ground truth", "told", or any phrase referring to how many
  views you were shown or that the answer was supplied to you.
- Refer to faces as the bottom face, the face opposite the bottom, the visible
  side faces, and the hidden side faces, and give their values.
- Two to four sentences.

Respond EXACTLY with this JSON:
{
  "reasoning": "<your reasoning for this step>",
  "bottom": <the bottom face value after the roll>
}
"""

OCTAHEDRON_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "bottom": {"type": "integer"},
    },
    "required": ["reasoning", "bottom"],
}


def is_octahedron(metadata: Dict[str, Any]) -> bool:
    return metadata.get("solid") == "octahedron"


# The order face values 1..8 take, matching SYMBOL_ORDER in blender_render.py
# and the contours bake_symbols.py writes. A symbol die shows these instead of
# digits or pips, so the reasoning has to name what is actually on the face --
# without this the model is shown a heart and told to call it "1".
SYMBOL_NAMES = ["heart", "arrow", "triangle", "crescent moon", "star", "cross",
                "circle", "square"]


def uses_symbols(metadata: Dict[str, Any]) -> bool:
    """True when this puzzle's faces carry symbols rather than digits or pips."""
    return bool(metadata.get("net_symbol_image"))


def mark(value: Any, symbols: bool) -> str:
    """
    How one face reads.

    On a symbol die the value is still the thing being tracked -- opposite
    faces sum to 7 or 9 by value, not by picture -- so both are given: the
    model sees the symbol and reasons with the number behind it.
    """
    if not symbols or not isinstance(value, int) or not 1 <= value <= len(SYMBOL_NAMES):
        return str(value)
    return f"{SYMBOL_NAMES[value - 1]} ({value})"


def describe_octahedron(faces: Dict[str, Any], symbols: bool = False) -> str:
    """The eight-faced die's state, worded the way the picture shows it."""
    def seq(v):
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(mark(x, symbols) for x in v) + "]"
        return mark(v, symbols)
    return (
        f"bottom {seq(faces['bottom'])}, "
        f"opposite the bottom {seq(faces['opposite_the_bottom'])}, "
        f"visible sides {seq(faces['visible_sides'])}, "
        f"hidden sides {seq(faces['hidden_sides'])}"
    )


# Phrases that mean the text leaked how it was produced.
BAD_PHRASES = [
    "first image", "second image", "third image", "fourth image",
    "next state", "after state", "hint", "ground truth", "given state",
    "as shown", "the images", "following image", "provided image",
    "we are told", "i was told", "i am given", "you gave",
]

REASONING_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "top": {"type": "integer"},
        "bottom": {"type": "integer"},
    },
    "required": ["reasoning", "top", "bottom"],
}


def is_hint_leak(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in BAD_PHRASES)


def step_images(puzzle_dir: Path, metadata: Dict[str, Any], step: Dict[str, Any]):
    """The before/after images for one step."""
    idx = step["step"]
    tag = step.get("path")
    if tag:
        before = (
            puzzle_dir / metadata["initial_state_image"]
            if idx == 1
            else puzzle_dir / f"cot_{tag}_{idx - 2:02d}.png"
        )
        after = puzzle_dir / f"cot_{tag}_{idx - 1:02d}.png"
    else:
        before = (
            puzzle_dir / metadata["initial_state_image"]
            if idx == 1
            else puzzle_dir / f"cot_{idx - 2:02d}.png"
        )
        after = puzzle_dir / f"cot_{idx - 1:02d}.png"
    return before, after


def describe(die: Dict[str, int], symbols: bool = False) -> str:
    """Face values, labelled the way they are described to the model."""
    return ", ".join(f"{label} {mark(die[key], symbols)}"
                     for key, label in FACE_NAMES.items())


# Appended when the die carries symbols, so the model names what it can see.
SYMBOL_NOTE = """
This die is marked with symbols rather than digits or pips. Each face is given
below as its symbol with the value it stands for in brackets. Refer to the
faces by their symbols, and reason with the values behind them (opposite faces
still sum as the rules say). Do not claim a face shows a number.
"""


def build_step_prompt(step: Dict[str, Any], octahedron: bool = False,
                      symbols: bool = False) -> str:
    note = SYMBOL_NOTE if symbols else ""
    if octahedron:
        return (
            f"{OCTAHEDRON_PROMPT}{note}\n\n"
            f"Direction rolled: {step['direction_name']}\n"
            f"Faces before: {describe_octahedron(step['faces_before'], symbols)}\n"
            f"Faces after: {describe_octahedron(step['faces_after'], symbols)}\n"
        )
    return (
        f"{PROMPT}{note}\n\n"
        f"Direction rolled: {step['direction_name']}\n"
        f"Faces before: {describe(step['die_before'], symbols)}\n"
        f"Faces after: {describe(step['die_after'], symbols)}\n"
    )


def make_octahedron_record(step: Dict[str, Any],
                           parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Same shape as the cube's record, minus the top face the task ignores."""
    record = {
        "path": None,
        "step": step["step"],
        "direction": step["direction_name"],
        "reasoning": parsed["reasoning"],
        "predicted_bottom": parsed["bottom"],
        "true_bottom": step["bottom_face"],
        "hint_leak": is_hint_leak(parsed["reasoning"]),
    }
    record["faces_correct"] = record["predicted_bottom"] == record["true_bottom"]
    return record


def make_record(step: Dict[str, Any], parsed: Dict[str, Any]) -> Dict[str, Any]:
    record = {
        "path": step.get("path"),
        "step": step["step"],
        "direction": step["direction"],
        "reasoning": parsed["reasoning"],
        "predicted_top": parsed["top"],
        "predicted_bottom": parsed["bottom"],
        "true_top": step["die_after"]["top"],
        "true_bottom": step["bottom_face"],
        "hint_leak": is_hint_leak(parsed["reasoning"]),
    }
    record["faces_correct"] = (
        record["predicted_top"] == record["true_top"]
        and record["predicted_bottom"] == record["true_bottom"]
    )
    return record


def generate_for_puzzle(
    client: Any, puzzle_dir: Path, model: str, overwrite: bool = False
) -> int:
    """Fill in reasoning for each step. Returns the number of steps written."""
    metadata = json.loads((puzzle_dir / "metadata.json").read_text())
    out_path = puzzle_dir / "cot_reasoning.json"

    existing: Dict[str, Any] = {}
    if out_path.exists() and not overwrite:
        try:
            existing = json.loads(out_path.read_text())
        except json.JSONDecodeError:
            existing = {}

    by_key = {(r.get("path"), r["step"]): r for r in existing.get("steps", [])}
    octa = is_octahedron(metadata)
    syms = uses_symbols(metadata)

    written = 0
    for step in metadata["steps"]:
        key = (step.get("path"), step["step"])
        prior = by_key.get(key)
        if prior and prior.get("reasoning") and not is_hint_leak(prior["reasoning"]):
            continue

        before, after = step_images(puzzle_dir, metadata, step)
        if not before.exists() or not after.exists():
            print(f"  [skip missing image] {puzzle_dir.name} step {step['step']}",
                  file=sys.stderr)
            continue

        parts = [
            text_part("Board before this roll:"),
            png_part(before),
            text_part("Board after this roll:"),
            png_part(after),
            text_part(build_step_prompt(step, octahedron=octa, symbols=syms)),
        ]

        try:
            parsed = generate_json(
                client, model, parts,
                required_fields=["reasoning", "bottom"] if octa
                else ["reasoning", "top", "bottom"],
                schema=OCTAHEDRON_SCHEMA if octa else REASONING_SCHEMA,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [error] {puzzle_dir.name} step {step['step']}: {exc}",
                  file=sys.stderr)
            continue

        by_key[key] = (make_octahedron_record(step, parsed) if octa
                       else make_record(step, parsed))
        written += 1

    ordered = [by_key[k] for k in sorted(by_key, key=lambda k: (str(k[0]), k[1]))]
    out_path.write_text(
        json.dumps({"puzzle_id": metadata["puzzle_id"], "steps": ordered}, indent=2)
    )
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate per-step reasoning texts")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--variant", choices=["top", "sum", "two", "octahedron"],
                    default=None)
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--overwrite", action="store_true")
    # Each puzzle is independent and writes only its own cot_reasoning.json, so
    # they can run at once. Throughput keeps climbing with concurrency (measured
    # 3.5x at 8 workers, 5.4x at 16) -- the API, not the client, is the limit.
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent requests (1 = the old sequential behaviour)")
    args = ap.parse_args()

    client = make_client()

    puzzle_dirs = list(iter_puzzle_dirs(Path(args.output_dir), args.variant, args.level))
    if args.limit:
        puzzle_dirs = puzzle_dirs[: args.limit]
    if not puzzle_dirs:
        sys.exit(f"No puzzles found under {args.output_dir}")

    total = 0
    if args.workers <= 1:
        for i, puzzle_dir in enumerate(puzzle_dirs, 1):
            written = generate_for_puzzle(
                client, puzzle_dir, args.model, overwrite=args.overwrite
            )
            total += written
            print(f"[{i}/{len(puzzle_dirs)}] {puzzle_dir}: {written} steps written")
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(generate_for_puzzle, client, d, args.model,
                            overwrite=args.overwrite): d
                for d in puzzle_dirs
            }
            for fut in as_completed(futures):
                d = futures[fut]
                done += 1
                try:
                    written = fut.result()
                except Exception as exc:  # noqa: BLE001 - one puzzle must not stop the run
                    print(f"[{done}/{len(puzzle_dirs)}] {d}: FAILED {exc}",
                          file=sys.stderr)
                    continue
                total += written
                print(f"[{done}/{len(puzzle_dirs)}] {d}: {written} steps written",
                      flush=True)

    print(f"\nWrote {total} reasoning texts across {len(puzzle_dirs)} puzzles.")


if __name__ == "__main__":
    main()
