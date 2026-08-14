"""
Tool-use rollout (Gemini function calling): the model rolls the die by calling
functions and observes the result, rather than tracking the orientation in its
head.

Two functions are declared:

    roll_die(direction)  -- tip the die one cell; returns the new six-face state,
                            or an explicit rejection if the roll would leave the
                            board or hit a blocked cell
    observe()            -- re-inspect the current position and face values

Each function response is followed by a freshly rendered image of the board, so
the model sees the consequence of its own move. Illegal moves are reported
rather than silently ignored, which is what lets the model notice a collision
and re-plan.

Usage:
    python tool_use_rollout.py --puzzle output/top/level_03/puzzle_0001
    python tool_use_rollout.py --output-dir output --variant top --limit 20
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parent / "sft_pipeline"))

from gemini_common import (  # noqa: E402
    DEFAULT_MODEL,
    iter_puzzle_dirs,
    make_client,
    png_part,
    png_part_from_bytes,
    text_part,
)
from generator import FACE_NAMES, apply_action, render_state  # noqa: E402

MAX_TOOL_CALLS = 60

# The engine works in compass terms internally; the model is given plain
# on-screen directions, which is what someone looking at the picture actually
# sees. The mapping follows the projection: +y (north) recedes to the upper
# left, +x (east) recedes to the upper right.
DIRECTION_ALIASES = {
    "up-left": "N",
    "up-right": "E",
    "down-right": "S",
    "down-left": "W",
}

# Face labels shown to the model come from the generator, so the tool results
# and the rendered dataset always agree.
FACE_LABELS = FACE_NAMES

ROLL_DIE = types.FunctionDeclaration(
    name="roll_die",
    description=(
        "Roll the die one square in the direction you give, tipping it over "
        "that edge. Directions are what you see in the picture: the board is "
        "drawn at an angle, so the die moves diagonally on screen. Returns "
        "the numbers on the die after the roll. If the roll would go off the "
        "board or into a black square it is refused and the die stays put — "
        "read the reply before choosing your next move."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "direction": types.Schema(
                type=types.Type.STRING,
                enum=["up-left", "up-right", "down-left", "down-right"],
                description=(
                    "Which way the die should roll, as it looks on screen."
                ),
            )
        },
        required=["direction"],
    ),
)

OBSERVE = types.FunctionDeclaration(
    name="observe",
    description=(
        "Re-inspect the board without moving the die. Returns the current cell "
        "and the six face values. Use this to re-orient after a rejected roll."
    ),
    parameters=types.Schema(type=types.Type.OBJECT, properties={}),
)

TOOLS = [types.Tool(function_declarations=[ROLL_DIE, OBSERVE])]


# ============================================================================
# The live board the tools act on
# ============================================================================


class DiceSession:
    """Holds the true die state and answers function calls against it."""

    def __init__(self, metadata: Dict[str, Any]) -> None:
        self.board = metadata["board"]
        self.state = {
            "board": metadata["board"],
            "position": list(metadata["start"]),
            "die": dict(metadata["initial_die"]),
        }
        self.variant = metadata["variant"]
        self.history: List[Dict[str, Any]] = []
        self.bottom_sum = 0
        self.rejected = 0

    def render_png(self) -> bytes:
        """Render the live board as a PNG for the function response."""
        instance = {
            "variant": "top",
            "board": self.board,
            "start": self.state["position"],
            "initial_die": self.state["die"],
            "path": [],
            "trace": [
                {
                    "step": 0,
                    "direction": None,
                    "position": list(self.state["position"]),
                    "die": dict(self.state["die"]),
                    "bottom": self.state["die"]["bottom"],
                }
            ],
            "num_rolls": 0,
            "answer": None,
        }
        fig = render_state(instance, step=0, show_path=False)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    def _payload(self, message: str, rejected: bool) -> Dict[str, Any]:
        die = self.state["die"]
        x, y = self.state["position"]
        return {
            "message": message,
            "rejected": rejected,
            "cell": {"x": x, "y": y},
            # Reported with the same screen-relative names the model rolls with
            "faces": {
                label: die[compass] for compass, label in FACE_LABELS.items()
            },
            "bottom_faces_summed_so_far": self.bottom_sum,
        }

    def roll(self, direction: str) -> Dict[str, Any]:
        """
        Apply a roll, or report why it was rejected.

        `direction` is a screen-relative name; compass codes are also accepted
        so the session can be driven directly from a stored path.
        """
        code = DIRECTION_ALIASES.get(direction, direction)
        if code not in ("N", "S", "E", "W"):
            return self._payload(
                f"I don't know the direction {direction!r}. Roll one of: "
                f"{', '.join(sorted(DIRECTION_ALIASES))}.",
                True,
            )
        try:
            self.state = apply_action(self.state, {"direction": code})
        except ValueError as exc:
            self.rejected += 1
            # The engine words its errors in compass terms; restate them the
            # way the caller is thinking about the board.
            hit_wall = "blocked" in str(exc)
            reason = (
                "there is a black square in the way"
                if hit_wall
                else "that would go off the edge of the board"
            )
            return self._payload(
                f"The die can't roll {direction} — {reason}. It has not moved, "
                f"so it is still on the same square with the same numbers.",
                True,
            )

        bottom = self.state["die"]["bottom"]
        self.bottom_sum += bottom
        self.history.append(
            {
                "direction": direction,
                "position": list(self.state["position"]),
                "die": dict(self.state["die"]),
                "bottom": bottom,
            }
        )
        return self._payload(
            f"Rolled {direction}. The face now touching the board is {bottom}.",
            False,
        )

    def observe(self) -> Dict[str, Any]:
        return self._payload("Current state.", False)

    def dispatch(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "roll_die":
            return self.roll(args.get("direction", ""))
        if name == "observe":
            return self.observe()
        return {"message": f"Unknown function: {name}", "rejected": True}


# ============================================================================
# Rollout
# ============================================================================


def build_prompt(metadata: Dict[str, Any], prompt_template: str) -> str:
    answer_format = {
        "top": '{"answer": 3}',
        "sum": '{"answer": 14}',
        "two": '{"answer": {"winner": "red", "total": 17}}',
    }[metadata["variant"]]
    return prompt_template.replace("{question}", metadata["question"]).replace(
        "{answer_format}", answer_format
    )


def extract_answer(text: str) -> Optional[Any]:
    """Pull the JSON answer out of the model's final message."""
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
    try:
        return json.loads(text)["answer"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def run_puzzle(
    client: Any,
    puzzle_dir: Path,
    prompt_template: str,
    model: str = DEFAULT_MODEL,
    max_tool_calls: int = MAX_TOOL_CALLS,
) -> Dict[str, Any]:
    """Run one puzzle end to end and return the transcript summary."""
    metadata = json.loads((puzzle_dir / "metadata.json").read_text())
    session = DiceSession(metadata)

    contents: List[types.Content] = [
        types.Content(
            role="user",
            parts=[
                png_part(puzzle_dir / metadata["initial_state_image"]),
                text_part(build_prompt(metadata, prompt_template)),
            ],
        )
    ]

    config = types.GenerateContentConfig(tools=TOOLS)
    tool_calls = 0
    final_text = ""

    while tool_calls <= max_tool_calls:
        response = client.models.generate_content(
            model=model, contents=contents, config=config
        )

        candidate = response.candidates[0] if response.candidates else None
        if candidate is None or candidate.content is None:
            break

        contents.append(candidate.content)

        calls = [p.function_call for p in (candidate.content.parts or [])
                 if getattr(p, "function_call", None)]

        if not calls:
            final_text = "".join(
                p.text for p in (candidate.content.parts or [])
                if getattr(p, "text", None)
            )
            break

        reply_parts: List[types.Part] = []
        for call in calls:
            tool_calls += 1
            args = dict(call.args or {})
            payload = session.dispatch(call.name, args)
            reply_parts.append(
                types.Part.from_function_response(name=call.name, response=payload)
            )
            # Show the board after this call, so the model sees its own effect.
            reply_parts.append(png_part_from_bytes(session.render_png()))

        contents.append(types.Content(role="user", parts=reply_parts))

    predicted = extract_answer(final_text)
    expected = metadata["answer"]

    if metadata["variant"] == "two":
        correct = (
            isinstance(predicted, dict)
            and str(predicted.get("winner", "")).strip().lower() == expected["winner"]
            and predicted.get("total") == expected["total"]
        )
    else:
        try:
            correct = int(predicted) == int(expected)
        except (TypeError, ValueError):
            correct = False

    return {
        "puzzle": str(puzzle_dir),
        "variant": metadata["variant"],
        "level": metadata["level"],
        "predicted": predicted,
        "expected": expected,
        "correct": correct,
        "tool_calls": tool_calls,
        "rejected_moves": session.rejected,
        "rolls": [h["direction"] for h in session.history],
        "final_text": final_text,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Tool-use rollout for Rolling Dice")
    ap.add_argument("--puzzle", type=str, default=None,
                    help="Run a single puzzle directory")
    ap.add_argument("--output-dir", type=str, default="output")
    ap.add_argument("--variant", choices=["top", "sum", "two"], default=None)
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--results", type=str, default="tool_use_results.json")
    args = ap.parse_args()

    client = make_client()
    prompt_template = (Path(__file__).parent / "prompts" / "tool_use.txt").read_text()

    if args.puzzle:
        puzzle_dirs = [Path(args.puzzle)]
    else:
        puzzle_dirs = list(
            iter_puzzle_dirs(Path(args.output_dir), args.variant, args.level)
        )
        if args.limit:
            puzzle_dirs = puzzle_dirs[: args.limit]

    if not puzzle_dirs:
        sys.exit("No puzzles found.")

    results = []
    for i, puzzle_dir in enumerate(puzzle_dirs, 1):
        try:
            result = run_puzzle(client, puzzle_dir, prompt_template, model=args.model)
        except Exception as exc:  # keep going; one bad puzzle shouldn't stop the run
            result = {"puzzle": str(puzzle_dir), "error": str(exc), "correct": False}
        results.append(result)
        mark = "✓" if result.get("correct") else "✗"
        print(
            f"[{i}/{len(puzzle_dirs)}] {mark} {puzzle_dir} "
            f"pred={result.get('predicted')} exp={result.get('expected')} "
            f"calls={result.get('tool_calls')} rejected={result.get('rejected_moves')}"
        )

    correct = sum(1 for r in results if r.get("correct"))
    print(f"\nAccuracy: {correct}/{len(results)} = {correct / max(1, len(results)):.1%}")

    Path(args.results).write_text(json.dumps(results, indent=2))
    print(f"Wrote {args.results}")


if __name__ == "__main__":
    main()
