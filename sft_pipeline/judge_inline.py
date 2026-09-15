"""
Judge each step's reasoning text one-by-one, no batch job (Gemini).

A step passes only if all three hold:
  1. The stated top/bottom faces match ground truth (checked in code, not by a
     model — the kinematics are exact, so there is no reason to ask).
  2. The text contains no phrase revealing it was shown the answer.
  3. The judge model rates the explanation as logically sound and self-contained.

Verdicts are written back into cot_reasoning.json. Already-judged steps are
skipped, so re-running is idempotent.

Usage:
    python sft_pipeline/judge_inline.py --output-dir output
    python sft_pipeline/judge_inline.py --output-dir output --rejudge
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gemini_common import (  # noqa: E402
    DEFAULT_MODEL,
    generate_json,
    iter_puzzle_dirs,
    make_client,
    text_part,
)
from generate_reasoning import (  # noqa: E402
    SYMBOL_NOTE,
    describe,
    is_hint_leak,
    mark,
    uses_symbols,
)

JUDGE_PROMPT = """
You are judging one step of reasoning about a die rolled across a grid board.

The rules of the task:
- A standard die has opposite faces summing to 7.
- Rolling one cell tips the die over its leading bottom edge.
- Rolling one square tips the die over the edge it rolls toward. Four sides
  take one step around in that direction: the side facing the way it rolls goes
  to the bottom, the top becomes that side, the bottom becomes the trailing
  side, and the trailing side comes up to be the new top.
- The two sides facing sideways to the roll do not move.
- The board is drawn at an angle, so the die rolls diagonally on screen. Faces
  are labelled top, bottom, and the four sides named by the direction they
  face (up-left, up-right, down-left, down-right).

Here is the step:
Direction rolled: {direction}
Faces before: {before}
Faces after (correct): {after}

Here is the candidate reasoning:
---
{reasoning}
---

Judge it on three criteria:
1. is_correct_reasoning — Does the reasoning describe the face movement
   accurately, and are its stated results consistent with the correct faces?
2. is_correct_no_hints — Does it read as reasoning derived from the rules and
   the previous state, rather than referring to being shown or told the answer?
3. is_sound — Is it a clear, logical explanation rather than a bare restatement
   of the numbers with no reasoning?

Respond EXACTLY with this JSON:
{{
  "is_correct_reasoning": true or false,
  "is_correct_no_hints": true or false,
  "is_sound": true or false,
  "explanation": "<one sentence on the most important problem, or 'ok'>"
}}
"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "is_correct_reasoning": {"type": "boolean"},
        "is_correct_no_hints": {"type": "boolean"},
        "is_sound": {"type": "boolean"},
        "explanation": {"type": "string"},
    },
    "required": [
        "is_correct_reasoning",
        "is_correct_no_hints",
        "is_sound",
        "explanation",
    ],
}



OCTAHEDRON_JUDGE_PROMPT = """
You are judging one step of reasoning about an eight-faced die rolled across a
triangular grid.

The rules of the task:
- The die is a regular octahedron with opposite faces summing to 9.
- It rests on one triangular face. Each cell of the grid has three neighbours,
  and rolling tips the die over one edge of the triangle it is resting on.
- The triangles alternate in orientation, so the cell the die lands on points
  the opposite way to the one it left.
- Faces are described as the bottom face (on the board), the face opposite the
  bottom, the visible side faces, and the hidden side faces.

Here is the step:
Direction rolled: {direction}
Faces before: {before}
Faces after (correct): {after}

Here is the candidate reasoning:
---
{reasoning}
---

Judge it on three criteria:
1. is_correct_reasoning — Does the reasoning describe the roll accurately, and
   is its stated bottom face consistent with the correct faces?
2. is_correct_no_hints — Does it read as reasoning derived from the rules and
   the previous state, rather than referring to being shown or told the answer?
3. is_sound — Is it a clear, logical explanation rather than a bare restatement
   of the numbers with no reasoning?

Respond EXACTLY with this JSON:
{{
  "is_correct_reasoning": true or false,
  "is_correct_no_hints": true or false,
  "is_sound": true or false,
  "explanation": "<one sentence>"
}}
"""


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


def _ask_judge(client: Any, model: str, prompt: str, leaked: bool) -> Dict[str, Any]:
    """Run the judge model and fold its verdict into the standard shape."""
    try:
        verdict = generate_json(
            client, model, [text_part(prompt)],
            required_fields=["is_correct_reasoning", "is_correct_no_hints", "is_sound"],
            schema=JUDGE_SCHEMA,
        )
    except Exception as exc:  # noqa: BLE001
        return {"passed": False, "reason": f"judge failed: {exc}"}

    passed = (
        bool(verdict["is_correct_reasoning"])
        and bool(verdict["is_correct_no_hints"])
        and bool(verdict["is_sound"])
    )
    return {
        "passed": passed,
        "faces_correct": True,
        "hint_leak": leaked,
        "sound": bool(verdict["is_sound"]),
        "reason": verdict.get("explanation", ""),
    }


def judge_step(
    client: Any,
    step_meta: Dict[str, Any],
    record: Dict[str, Any],
    model: str,
    symbols: bool = False,
) -> Dict[str, Any]:
    """Return a verdict dict for one step."""
    # Deterministic checks first — no API call needed if these already fail.
    # The octahedron task only asks for the bottom face, so it carries no
    # `die_after` and no top-face claim to check.
    if "die_after" in step_meta:
        faces_ok = (
            record.get("predicted_top") == step_meta["die_after"]["top"]
            and record.get("predicted_bottom") == step_meta["bottom_face"]
        )
    else:
        faces_ok = record.get("predicted_bottom") == step_meta["bottom_face"]
    leaked = is_hint_leak(record.get("reasoning", ""))

    if not faces_ok:
        return {
            "passed": False,
            "faces_correct": False,
            "hint_leak": leaked,
            "sound": None,
            "reason": "stated faces do not match ground truth",
        }
    if leaked:
        return {
            "passed": False,
            "faces_correct": True,
            "hint_leak": True,
            "sound": None,
            "reason": "reasoning leaks that the answer was supplied",
        }

    if "die_after" not in step_meta:
        # Octahedron: describe the eight-faced state instead of the cube's six.
        prompt = OCTAHEDRON_JUDGE_PROMPT.format(
            direction=step_meta["direction_name"],
            before=describe_octahedron(step_meta["faces_before"], symbols),
            after=describe_octahedron(step_meta["faces_after"], symbols),
            reasoning=record.get("reasoning", ""),
        ) + (SYMBOL_NOTE if symbols else "")
        return _ask_judge(client, model, prompt, leaked)

    prompt = JUDGE_PROMPT.format(
        direction=step_meta["direction"],
        before=describe(step_meta["die_before"], symbols),
        after=describe(step_meta["die_after"], symbols),
        reasoning=record["reasoning"],
    ) + (SYMBOL_NOTE if symbols else "")

    try:
        verdict = generate_json(
            client, model, [text_part(prompt)],
            required_fields=["is_correct_reasoning", "is_correct_no_hints", "is_sound"],
            schema=JUDGE_SCHEMA,
        )
    except Exception as exc:  # noqa: BLE001
        return {"passed": False, "reason": f"judge failed: {exc}"}

    passed = (
        bool(verdict["is_correct_reasoning"])
        and bool(verdict["is_correct_no_hints"])
        and bool(verdict["is_sound"])
    )
    return {
        "passed": passed,
        "faces_correct": True,
        "hint_leak": False,
        "sound": bool(verdict["is_sound"]),
        "reason": verdict.get("explanation", ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Judge per-step reasoning texts")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent puzzles (1 = the old sequential behaviour)")
    ap.add_argument("--variant", choices=["top", "sum", "two", "octahedron"],
                    default=None)
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--rejudge", action="store_true",
                    help="Re-judge steps that already carry a verdict")
    args = ap.parse_args()

    client = make_client()

    puzzle_dirs = list(iter_puzzle_dirs(Path(args.output_dir), args.variant, args.level))
    if args.limit:
        puzzle_dirs = puzzle_dirs[: args.limit]

    judged = passed = 0

    def judge_puzzle(puzzle_dir):
        """Judge one puzzle. Returns (judged, passed, total_steps) or None."""
        reasoning_path = puzzle_dir / "cot_reasoning.json"
        if not reasoning_path.exists():
            return None

        metadata = json.loads((puzzle_dir / "metadata.json").read_text())
        try:
            data = json.loads(reasoning_path.read_text())
        except json.JSONDecodeError:
            print(f"  [skip bad json] {reasoning_path}", file=sys.stderr)
            return None

        steps_meta = {(s.get("path"), s["step"]): s for s in metadata["steps"]}
        syms = uses_symbols(metadata)
        local_judged = local_pass = 0
        for record in data["steps"]:
            if record.get("verdict") and not args.rejudge:
                if record["verdict"].get("passed"):
                    local_pass += 1
                continue
            step_meta = steps_meta.get((record.get("path"), record["step"]))
            if step_meta is None:
                continue
            verdict = judge_step(client, step_meta, record, args.model, syms)
            record["verdict"] = verdict
            local_judged += 1
            if verdict["passed"]:
                local_pass += 1

        # Only this thread touches this file: one puzzle, one worker.
        reasoning_path.write_text(json.dumps(data, indent=2))
        return local_judged, local_pass, len(data["steps"])

    if args.workers <= 1:
        for i, puzzle_dir in enumerate(puzzle_dirs, 1):
            r = judge_puzzle(puzzle_dir)
            if r is None:
                print(f"[{i}/{len(puzzle_dirs)}] {puzzle_dir}: skipped")
                continue
            j, p_, n = r
            judged += j
            passed += p_
            print(f"[{i}/{len(puzzle_dirs)}] {puzzle_dir}: {p_}/{n} passed")
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(judge_puzzle, d): d for d in puzzle_dirs}
            for fut in as_completed(futures):
                d = futures[fut]
                done += 1
                try:
                    r = fut.result()
                except Exception as exc:  # noqa: BLE001 - keep going past one bad puzzle
                    print(f"[{done}/{len(puzzle_dirs)}] {d}: FAILED {exc}",
                          file=sys.stderr)
                    continue
                if r is None:
                    print(f"[{done}/{len(puzzle_dirs)}] {d}: skipped", flush=True)
                    continue
                j, p_, n = r
                judged += j
                passed += p_
                print(f"[{done}/{len(puzzle_dirs)}] {d}: {p_}/{n} passed", flush=True)

    print(f"\nJudged {judged} steps this run. {passed} steps currently passing.")


if __name__ == "__main__":
    main()
