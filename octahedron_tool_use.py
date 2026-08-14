"""
Tool-use rollout for the octahedron variant (Gemini function calling).

The cube's rollout in tool_use_rollout.py is the model for this: the model
rolls by calling functions and sees a freshly rendered board after each call,
instead of tracking the solid in its head.

Two things differ from the cube and shape the design:

  Only three of the six directions are legal from any pose. Which three depends
  on the orientation the solid is resting in, because a triangle has three
  edges and the cell it lands on points the opposite way. The tool therefore
  reports the legal moves in every response rather than relying on a fixed
  enum -- an enum listing all six would invite calls that can never work.

  The question asks for the face resting on the board, not the top, so the
  payload leads with the bottom face.

Usage:
    python octahedron_tool_use.py --puzzle output/octahedron/level_05/puzzle_0001
    python octahedron_tool_use.py --output-dir output --limit 5
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
    make_client,
    png_part_from_bytes,
    text_part,
)
import octahedron as octa  # noqa: E402

MAX_TOOL_CALLS = 60

ROLL_DIE = types.FunctionDeclaration(
    name="roll_die",
    description=(
        "Roll the eight-sided die one triangle in the direction you give. "
        "Only three directions are possible at any moment, because the die is "
        "resting on a triangle and can only tip over one of its three edges. "
        "Every reply lists which directions are available next. Returns the "
        "number now facing down. If the roll is not possible it is refused and "
        "the die stays put -- read the reply before choosing your next move."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "direction": types.Schema(
                type=types.Type.STRING,
                enum=sorted(set(octa.STEP_NAMES.values())),
                description="Which way the die should roll, as it looks on screen.",
            )
        },
        required=["direction"],
    ),
)

OBSERVE = types.FunctionDeclaration(
    name="observe",
    description=(
        "Re-inspect the board without moving the die. Returns the current cell, "
        "the face resting on the board, and which directions are available."
    ),
    parameters=types.Schema(type=types.Type.OBJECT, properties={}),
)

TOOLS = [types.Tool(function_declarations=[ROLL_DIE, OBSERVE])]


class OctahedronSession:
    """Holds the true solid state and answers function calls against it."""

    def __init__(self, metadata: Dict[str, Any]) -> None:
        self.metadata = metadata
        self.board = octa.build_board(tuple(metadata["board_radius"]))
        self.cell = (round(metadata["start"][0], 2), round(metadata["start"][1], 2))
        self.orientation = 0
        self.history: List[Dict[str, Any]] = []
        self.rejected = 0

    def legal_moves(self) -> Dict[str, Any]:
        """Direction name -> the lattice step it stands for, for legal rolls."""
        out = {}
        for step in octa.ROLL_GRAPH[self.orientation]:
            target = (round(self.cell[0] + step[0], 2), round(self.cell[1] + step[1], 2))
            if target in self.board:
                out[octa.step_name(step)] = step
        return out

    def render_png(self) -> bytes:
        """
        Render the live board for the function reply.

        The die's own position comes first, then the rest of the intended
        route, so the picture still shows where it is meant to go. Rendering
        only the current cell -- which is what the first version did -- left the
        model with no route to follow after its first move, and it wandered.
        """
        remaining = [e for e in self.metadata["trace"][len(self.history) + 1:]]
        trace = [{"cell": list(self.cell), "orientation": self.orientation,
                  "bottom": octa.bottom_value(self.orientation)}] + remaining
        instance = {
            "board_radius": self.metadata["board_radius"],
            "trace": trace,
        }
        fig = octa.render_state(instance, step=0)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    def _payload(self, message: str, rejected: bool) -> Dict[str, Any]:
        layout = octa.face_layout(self.orientation)
        return {
            "message": message,
            "rejected": rejected,
            "cell": {"x": self.cell[0], "y": self.cell[1]},
            "face_on_the_board": layout["bottom"],
            "face_opposite_the_board": layout["opposite_the_bottom"],
            "visible_side_faces": layout["visible_sides"],
            "directions_available_now": sorted(self.legal_moves()),
        }

    def roll(self, direction: str) -> Dict[str, Any]:
        moves = self.legal_moves()
        if direction not in moves:
            self.rejected += 1
            return self._payload(
                f"The die can't roll {direction} from here. It is resting on a "
                f"triangle with three edges, so only {', '.join(sorted(moves))} "
                f"are possible. It has not moved.",
                True,
            )

        step = moves[direction]
        self.orientation = octa.ROLL_GRAPH[self.orientation][step]
        self.cell = (round(self.cell[0] + step[0], 2), round(self.cell[1] + step[1], 2))
        bottom = octa.bottom_value(self.orientation)
        self.history.append({
            "direction": direction,
            "cell": list(self.cell),
            "orientation": self.orientation,
            "bottom": bottom,
        })
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


PROMPT = """
Attached is a picture of an eight-sided die (a regular octahedron) resting on a
triangular grid. The dashed orange triangle marks where the path ends. The
dashed orange line is the route the die must roll along.

The die is a fair eight-sided die: the numbers on opposite faces always sum to 9.

You can roll the die by calling roll_die. It rests on one triangular face and
can only tip over one of that triangle's three edges, so only three directions
are possible at any moment. Every reply tells you which ones. Call observe if
you need to re-check the board without moving.

Follow the drawn route to its end.

Task:
{question}

Output:
When you have finished rolling, respond in JSON:
{answer_format}
"""


def extract_answer(text: str) -> Optional[Any]:
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        try:
            return json.loads(fenced.group(1)).get("answer")
        except (json.JSONDecodeError, AttributeError):
            pass
    for m in re.finditer(r"\{[^{}]*\"answer\"\s*:.*?\}\s*\}?", text, re.S):
        try:
            return json.loads(m.group(0))["answer"]
        except (json.JSONDecodeError, KeyError):
            continue
    # A degenerate response can run to tens of thousands of digits; converting
    # that with int() raises rather than returning a wrong answer, which would
    # crash the whole rollout. The task's answers are single digits, so anything
    # longer is not an answer at all.
    trailing = re.findall(r"\b(\d{1,3})\b", text)
    return int(trailing[-1]) if trailing else None


def run_puzzle(client, puzzle_dir: Path, model: str) -> Dict[str, Any]:
    metadata = json.loads((puzzle_dir / "metadata.json").read_text())
    session = OctahedronSession(metadata)

    prompt = (PROMPT
              .replace("{question}", metadata["question"])
              .replace("{answer_format}", '{"answer": 5}'))

    contents = [types.Content(role="user", parts=[
        png_part_from_bytes((puzzle_dir / "initial.png").read_bytes()),
        text_part(prompt),
    ])]
    config = types.GenerateContentConfig(tools=TOOLS)

    calls_made = 0
    final_text = ""
    for _ in range(MAX_TOOL_CALLS):
        resp = client.models.generate_content(
            model=model, contents=contents, config=config)
        candidate = resp.candidates[0]
        contents.append(candidate.content)

        calls = [p.function_call for p in (candidate.content.parts or [])
                 if getattr(p, "function_call", None)]
        if not calls:
            final_text = "".join(
                p.text for p in (candidate.content.parts or [])
                if getattr(p, "text", None))
            break

        reply_parts = []
        for call in calls:
            calls_made += 1
            payload = session.dispatch(call.name, dict(call.args or {}))
            reply_parts.append(
                types.Part.from_function_response(name=call.name, response=payload))
            reply_parts.append(png_part_from_bytes(session.render_png()))
        contents.append(types.Content(role="user", parts=reply_parts))

    predicted = extract_answer(final_text)
    return {
        "puzzle_id": metadata["puzzle_id"],
        "level": metadata["level"],
        "variant": "octahedron",
        "puzzle_dir": str(puzzle_dir),
        "setting": "tool_use",
        "model": model,
        "tool_calls": calls_made,
        "rejected_moves": session.rejected,
        "final_text": final_text,
        "predicted": predicted,
        "expected": metadata["answer"],
        "reached_cell": list(session.cell),
        "true_final_cell": metadata["trace"][-1]["cell"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Tool-use rollout for the octahedron")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--puzzle", default=None)
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--results", default="octahedron_tool_use.json")
    args = ap.parse_args()

    if args.puzzle:
        puzzle_dirs = [Path(args.puzzle)]
    else:
        root = Path(args.output_dir) / "octahedron"
        levels = ([root / f"level_{args.level:02d}"] if args.level
                  else sorted(root.glob("level_*")))
        puzzle_dirs = [p for lv in levels for p in sorted(lv.glob("puzzle_*"))]
    if args.limit:
        puzzle_dirs = puzzle_dirs[: args.limit]
    if not puzzle_dirs:
        sys.exit("No octahedron puzzles found.")

    client = make_client()
    results = []
    for i, puzzle_dir in enumerate(puzzle_dirs, 1):
        print(f"[{i}/{len(puzzle_dirs)}] {puzzle_dir} ... ", end="", flush=True)
        try:
            row = run_puzzle(client, puzzle_dir, args.model)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED: {exc}")
            continue
        ok = row["predicted"] == row["expected"]
        print(f"predicted {row['predicted']}, expected {row['expected']} "
              f"({row['tool_calls']} calls, {row['rejected_moves']} rejected) "
              f"{'OK' if ok else 'X'}")
        results.append(row)

    Path(args.results).write_text(json.dumps(results, indent=2))
    if results:
        correct = sum(1 for r in results if r["predicted"] == r["expected"])
        print("-" * 66)
        print(f"  {correct}/{len(results)} = {correct/len(results):.1%}")
    print(f"Wrote {args.results}")


if __name__ == "__main__":
    main()
