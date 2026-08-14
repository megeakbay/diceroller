"""
Zero-shot text-only baseline for the Rolling Dice task.

The board, the die's visible faces, and the roll sequence are serialized to
plain text, so a model with no vision can attempt the puzzle. Nothing about the
kinematics is given away: only the three faces a viewer of the image would see
are stated, and the path is given as the directions themselves rather than as a
per-step trace.

This measures the mental-rotation half of the task in isolation. Reading the
path off an isometric render is removed, so the numbers are NOT comparable to
the image-based settings -- they are an upper bound on what a text model can do
once perception is free.

Usage:
    python text_only_rollout.py --output-dir output --variant top --level 5
    python text_only_rollout.py --puzzle output/top/level_05/puzzle_0001
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generator import DIRECTION_NAMES, FACE_NAMES  # noqa: E402
from sft_pipeline.gemini_common import (  # noqa: E402
    iter_puzzle_dirs,
    load_dotenv,
    make_client,
)

from google.genai import types  # noqa: E402

DEFAULT_MODEL = "gemini-3.1-pro-preview"
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "text_only.txt"

ANSWER_FORMATS = {
    "top": '{"answer": 3}',
    "sum": '{"answer": 14}',
    "two": '{"answer": {"winner": "red", "total": 17}}',
}

# The three faces a viewer of the render can see: the top and the two faces
# angled toward the camera. Kept identical to what the image setting exposes.
VISIBLE_FACES = ("top", "south", "east")


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def describe_board(meta: Dict[str, Any]) -> str:
    """Render a puzzle's board, die, and path as plain text."""
    board = meta["board"]
    lines: List[str] = []

    lines.append(
        f"The board is a {board['width']} by {board['height']} grid of square cells."
    )

    blocked = board.get("blocked") or []
    if blocked:
        cells = ", ".join(f"({x}, {y})" for x, y in blocked)
        lines.append(
            f"These cells are walls the die cannot enter: {cells}."
        )
    else:
        lines.append("There are no walls; every cell is open.")

    die = meta["initial_die"]
    visible = ", ".join(
        f"the {FACE_NAMES[face]} shows {die[face]}" for face in VISIBLE_FACES
    )
    lines.append("")
    lines.append(f"At the start, {visible}.")

    path = meta["path"]
    steps = ", then ".join(DIRECTION_NAMES[d] for d in path)
    lines.append("")
    lines.append(
        f"The die then makes {len(path)} rolls, in this order: {steps}."
    )

    return "\n".join(lines)


def describe_two_variant(meta: Dict[str, Any]) -> str:
    """The `two` variant carries two labelled paths rather than one."""
    board = meta["board"]
    lines: List[str] = [
        f"The board is a {board['width']} by {board['height']} grid of square cells."
    ]

    blocked = board.get("blocked") or []
    if blocked:
        cells = ", ".join(f"({x}, {y})" for x, y in blocked)
        lines.append(f"These cells are walls the die cannot enter: {cells}.")
    else:
        lines.append("There are no walls; every cell is open.")

    die = meta["initial_die"]
    visible = ", ".join(
        f"the {FACE_NAMES[face]} shows {die[face]}" for face in VISIBLE_FACES
    )
    lines.append("")
    lines.append(f"At the start, {visible}.")
    lines.append("")

    paths = meta.get("paths") or {}
    for label, path in paths.items():
        steps = ", then ".join(DIRECTION_NAMES[d] for d in path)
        lines.append(
            f"The {label} path is {len(path)} rolls: {steps}."
        )
        lines.append(
            "The die starts over from the same starting faces for this path."
        )
        lines.append("")

    return "\n".join(lines).rstrip()


def build_prompt(meta: Dict[str, Any], template: str) -> str:
    if meta["variant"] == "two" and meta.get("paths"):
        description = describe_two_variant(meta)
    else:
        description = describe_board(meta)

    return (
        template.replace("{board_description}", description)
        .replace("{question}", meta["question"])
        .replace("{answer_format}", ANSWER_FORMATS[meta["variant"]])
    )


def extract_answer(text: str) -> Optional[Any]:
    """Pull the JSON answer out of a model response."""
    if not text:
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


def call_model(client, model: str, prompt: str, thinking: bool) -> Dict[str, str]:
    """
    One zero-shot call. Retries transient failures.

    Returns the visible answer text and, separately, the model's reasoning
    summary. Requesting thought summaries does not change what the model was
    asked to do -- the prompt is untouched, so the setting stays zero-shot.
    """
    config_kwargs: Dict[str, Any] = {}
    if thinking:
        config_kwargs["thinking_config"] = types.ThinkingConfig(include_thoughts=True)
    else:
        # Zero-shot with no deliberate reasoning budget, where supported.
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

    backoff = 5.0
    last_err: Optional[Exception] = None
    for attempt in range(1, 5):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(**config_kwargs),
            )
            # Thought parts are flagged with `thought=True`; everything else is
            # the visible answer. resp.text folds both together, so split here.
            answer_parts: List[str] = []
            thought_parts: List[str] = []
            for cand in resp.candidates or []:
                for part in (cand.content.parts if cand.content else []) or []:
                    if not getattr(part, "text", None):
                        continue
                    if getattr(part, "thought", False):
                        thought_parts.append(part.text)
                    else:
                        answer_parts.append(part.text)

            text = "".join(answer_parts) or (resp.text or "")
            if not text:
                raise RuntimeError("empty response text")
            return {"text": text, "reasoning": "\n".join(thought_parts)}
        except Exception as exc:  # noqa: BLE001 - retry on anything transient
            last_err = exc
            # Thinking-only models reject a zero budget outright. Retrying will
            # not help, and silently dropping the flag would misreport the
            # setting, so fail loudly and let the caller pick another model.
            if "Budget 0 is invalid" in str(exc) or "only works in thinking" in str(exc):
                raise RuntimeError(
                    f"{model} cannot be run with thinking disabled. "
                    f"Either pass --thinking or choose a model that supports "
                    f"a zero reasoning budget."
                ) from exc
            if attempt < 4:
                time.sleep(backoff)
                backoff *= 2
    raise RuntimeError(f"generate_content failed: {last_err}")


def run_puzzle(
    client, model: str, puzzle_dir: Path, template: str, thinking: bool
) -> Dict[str, Any]:
    meta = json.loads((puzzle_dir / "metadata.json").read_text())
    prompt = build_prompt(meta, template)

    out = call_model(client, model, prompt, thinking)
    predicted = extract_answer(out["text"])

    return {
        "puzzle_id": meta["puzzle_id"],
        "level": meta["level"],
        "variant": meta["variant"],
        "puzzle_dir": str(puzzle_dir),
        "setting": "text_only_zero_shot",
        "provider": "google",
        "model": model,
        "thinking": thinking,
        "prompt": prompt,
        "response": out["text"],
        # Thought summaries, when the API returns any. Verification reads this
        # alongside `response`, so working shown only here still counts.
        "reasoning": out["reasoning"],
        "predicted": predicted,
        "expected": meta["answer"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Zero-shot text-only rollout for Rolling Dice"
    )
    ap.add_argument("--output-dir", default="output", help="Dataset root")
    ap.add_argument("--puzzle", default=None, help="Run a single puzzle directory")
    ap.add_argument("--variant", default=None, choices=["top", "sum", "two"])
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--thinking",
        action="store_true",
        help="Leave the model's reasoning budget at its default instead of zeroing it",
    )
    ap.add_argument("--results", default="text_only_results.json")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the prompts without calling the model",
    )
    args = ap.parse_args()

    template = PROMPT_PATH.read_text()

    if args.puzzle:
        puzzle_dirs = [Path(args.puzzle)]
    else:
        puzzle_dirs = list(
            iter_puzzle_dirs(Path(args.output_dir), args.variant, args.level)
        )
    if args.limit:
        puzzle_dirs = puzzle_dirs[: args.limit]

    if not puzzle_dirs:
        sys.exit("No puzzles matched.")

    if args.dry_run:
        for puzzle_dir in puzzle_dirs:
            meta = json.loads((puzzle_dir / "metadata.json").read_text())
            print("=" * 70)
            print(puzzle_dir)
            print("=" * 70)
            print(build_prompt(meta, template))
            print(f"\n[ground truth: {meta['answer']}]\n")
        return

    load_dotenv()
    client = make_client()

    results: List[Dict[str, Any]] = []
    for i, puzzle_dir in enumerate(puzzle_dirs, 1):
        print(f"[{i}/{len(puzzle_dirs)}] {puzzle_dir} ... ", end="", flush=True)
        try:
            row = run_puzzle(client, args.model, puzzle_dir, template, args.thinking)
        except Exception as exc:  # noqa: BLE001 - record and continue
            print(f"FAILED: {exc}")
            continue
        ok = row["predicted"] == row["expected"]
        print(f"predicted {row['predicted']}, expected {row['expected']} "
              f"{'OK' if ok else 'X'}")
        results.append(row)

    Path(args.results).write_text(json.dumps(results, indent=2))

    correct = sum(1 for r in results if r["predicted"] == r["expected"])
    print("-" * 70)
    print(f"  {correct}/{len(results)} = "
          f"{correct / len(results):.1%}" if results else "  no results")
    print(f"Wrote {args.results}")


if __name__ == "__main__":
    main()
