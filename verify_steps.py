"""
Check a model's spontaneous reasoning against ground truth.

Answer-level accuracy says only that the final number matched; it does not
distinguish a model that tracked the cube correctly from one that made
compensating errors or guessed.

This reads whatever working the model volunteered under the plain zero-shot
prompt -- nothing in that prompt asks for a trace or fixes its format, so the
parser is deliberately format-agnostic: it finds every complete six-face state
anywhere in the response, in order, and aligns them against the exact
kinematics in metadata.json.

Three outcomes per puzzle:
    sound       every stated state is on the true trajectory, and the full
                path from step 0 to step N is accounted for
    unsound     at least one stated state is off the true trajectory
    no trace    the model gave an answer without showing its work, so its
                reasoning cannot be checked either way

`no trace` is not a failure of the model -- the prompt never asked for one. It
means this particular check has nothing to say about that response.

Usage:
    python verify_steps.py --results text_only_2p5flash_think.json
    python verify_steps.py --results claude_results.json --show-bad
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

# Models label the faces in their own words; map the common spellings onto the
# compass-internal names used in metadata.json.
FACE_CODES = {
    "T": "top",
    "TOP": "top",
    "B": "bottom",
    "BOTTOM": "bottom",
    "UL": "north",
    "UP-LEFT": "north",
    "DR": "south",
    "DOWN-RIGHT": "south",
    "UR": "east",
    "UP-RIGHT": "east",
    "DL": "west",
    "DOWN-LEFT": "west",
}

_CODE_ALT = "|".join(sorted(FACE_CODES, key=len, reverse=True))
FACE_PAIR = re.compile(rf"\b({_CODE_ALT})\b\s*[:=]\s*([1-6])\b", re.I)


# A model's prose restates old face values while explaining a roll ("the side
# facing the way it rolls (Up-left: 2) becomes the bottom"). Those explanatory
# mentions must not be spliced into the state being derived, so accumulation is
# reset whenever a line reads as narration rather than a state assertion.
NARRATION = re.compile(
    r"\b(becomes?|was|were|comes? up|rolled|unchanged|do(?:es)? not move|"
    r"trailing|facing|deduce|rule|sum to)\b",
    re.I,
)


# Models sometimes declare a face order once and then give states positionally,
# e.g. "Current State: (T, B, UL, DR, UR, DL)" followed by "(5, 2, 1, 6, 3, 4)".
ORDER_DECL = re.compile(
    rf"\(\s*({_CODE_ALT})\s*(?:,\s*({_CODE_ALT})\s*){{5}}\)", re.I
)
TUPLE_ROW = re.compile(r"\(\s*([1-6])\s*(?:,\s*[1-6]\s*){5}\)")


def _positional_states(text: str) -> List[Dict[str, int]]:
    """
    Read states written as bare number tuples against a declared face order.

    Returns nothing unless the response actually declares an order, so a
    coincidental six-number tuple elsewhere cannot be misread as a state.
    """
    decl = ORDER_DECL.search(text)
    if not decl:
        return []
    order = [FACE_CODES[c.upper()] for c in re.findall(rf"\b({_CODE_ALT})\b", decl.group(0), re.I)]
    if len(order) != 6 or len(set(order)) != 6:
        return []

    states: List[Dict[str, int]] = []
    for match in TUPLE_ROW.finditer(text, decl.end()):
        values = [int(v) for v in re.findall(r"[1-6]", match.group(0))]
        if len(values) == 6:
            states.append(dict(zip(order, values)))
    return states


def parse_states(text: str) -> List[Dict[str, int]]:
    """
    Find every complete six-face state in a response, in order of appearance.

    Works line by line so that a state must be asserted as a contiguous run of
    face/value pairs. Lines that read as narration break the run, which keeps
    values quoted mid-explanation from being mistaken for a new state. This
    tolerates bullet lists, inline tuples, tables, and prose, since the plain
    prompt does not constrain the format.
    """
    states: List[Dict[str, int]] = []
    current: Dict[str, int] = {}

    def flush() -> None:
        nonlocal current
        current = {}

    for line in text.splitlines():
        pairs = FACE_PAIR.findall(line)
        if not pairs:
            # A blank or purely prose line ends any run in progress.
            if line.strip():
                flush()
            continue

        # An explanatory line quotes prior values; never fold it into a state.
        if NARRATION.search(line):
            flush()
            continue

        for code, value in pairs:
            face = FACE_CODES[code.upper()]
            if face in current:
                # Same face named twice means a new state began here.
                current = {}
            current[face] = int(value)
            if len(current) == 6:
                states.append(current)
                current = {}

    # Some responses give states positionally against a declared face order
    # rather than as named pairs. Prefer whichever reading recovers more of the
    # derivation, since a response uses one style or the other throughout.
    positional = _positional_states(text)
    if len(positional) > len(states):
        states = positional

    # Collapse consecutive restatements of the same state (models often give a
    # state as a bullet list and immediately repeat it as a summary line).
    deduped: List[Dict[str, int]] = []
    for state in states:
        if not deduped or state != deduped[-1]:
            deduped.append(state)
    return deduped


def verify_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Compare one response's volunteered working against the true trace."""
    meta = json.loads((Path(row["puzzle_dir"]) / "metadata.json").read_text())
    truths = [t["die"] for t in meta["trace"]]

    # Working can appear in the visible answer, in the API-returned reasoning
    # summary, or both. Read the summary first: it is where the derivation
    # actually happens when the model keeps its answer terse.
    stated = parse_states(
        "\n".join(filter(None, [row.get("reasoning"), row.get("response")]))
    )

    # Any state the model asserted that is nowhere on the true trajectory is a
    # genuine reasoning error, regardless of ordering.
    off_trajectory = [s for s in stated if s not in truths]

    # Walk the stated states forward through the true trace; a correct
    # derivation visits the steps in order.
    hit: List[int] = []
    cursor = 0
    for state in stated:
        while cursor < len(truths) and state != truths[cursor]:
            cursor += 1
        if cursor < len(truths):
            hit.append(cursor)
            cursor += 1

    complete = hit == list(range(len(truths)))
    has_trace = bool(stated)
    sound = has_trace and complete and not off_trajectory

    answer_correct = row.get("predicted") == row.get("expected")

    # A reasoning summary that narrates the process without ever stating face
    # values ("I worked through the five rolls") is not the same as a response
    # that showed nothing at all -- worth distinguishing in the report, since
    # the fix differs: one needs a different model, the other needs nothing.
    narrated_only = not has_trace and bool((row.get("reasoning") or "").strip())

    return {
        "puzzle_id": row.get("puzzle_id"),
        "level": row.get("level"),
        "model": row.get("model"),
        "answer_correct": answer_correct,
        "has_trace": has_trace,
        "narrated_without_values": narrated_only,
        "states_found": len(stated),
        "steps_expected": len(truths),
        "steps_hit_in_order": hit,
        "off_trajectory": off_trajectory,
        "reasoning_sound": sound,
        # The case this check exists to find: right number reached through a
        # state that is provably off the true trajectory. An incomplete trace
        # is not evidence of error, so it is deliberately excluded.
        "right_answer_wrong_work": answer_correct and bool(off_trajectory),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Verify volunteered reasoning against ground truth"
    )
    ap.add_argument("--results", required=True, help="Results JSON from a rollout")
    ap.add_argument(
        "--show-bad",
        action="store_true",
        help="Print off-trajectory states in full",
    )
    ap.add_argument("--report", default=None, help="Write the full report as JSON")
    args = ap.parse_args()

    rows = json.loads(Path(args.results).read_text())
    if isinstance(rows, dict):
        rows = rows.get("responses", rows.get("results", []))
    if not rows:
        raise SystemExit(f"No rows in {args.results}")

    checks = [verify_row(r) for r in rows]

    print("=" * 70)
    print(f"Reasoning verification - {args.results}")
    print("=" * 70)
    for c in checks:
        if c["narrated_without_values"]:
            verdict = "summary narrates process but states no face values"
        elif not c["has_trace"]:
            verdict = "no trace (not asked for; unverifiable)"
        elif c["reasoning_sound"]:
            verdict = f"sound ({len(c['steps_hit_in_order'])}/{c['steps_expected']} steps)"
        elif c["off_trajectory"]:
            verdict = f"UNSOUND ({len(c['off_trajectory'])} off-trajectory states)"
        else:
            # Every state given was correct, but the derivation is not fully
            # shown. Incomplete evidence, not demonstrated error.
            verdict = (
                f"partial ({len(c['steps_hit_in_order'])}/{c['steps_expected']} "
                f"steps shown, all correct)"
            )
        ans = "correct" if c["answer_correct"] else "WRONG"
        print(f"  puzzle {c['puzzle_id']} (level {c['level']}): answer {ans}, {verdict}")
        if args.show_bad:
            for s in c["off_trajectory"]:
                print(f"      off-trajectory: {s}")

    n = len(checks)
    answers = sum(c["answer_correct"] for c in checks)
    traced = [c for c in checks if c["has_trace"]]
    sound = sum(c["reasoning_sound"] for c in checks)
    lucky = sum(c["right_answer_wrong_work"] for c in checks)

    print("-" * 70)
    print(f"  answer accuracy:  {answers}/{n} = {answers / n:.1%}")
    if traced:
        print(
            f"  reasoning sound:  {sound}/{len(traced)} of responses that showed work"
        )
    narrated = sum(c["narrated_without_values"] for c in checks)
    print(f"  responses with no checkable trace: {n - len(traced)}/{n}")
    if narrated:
        print(f"    of those, summaries that narrate but state no values: {narrated}")
    print(f"  right answer, demonstrably wrong work: {lucky}/{n}")
    print("=" * 70)

    if args.report:
        Path(args.report).write_text(json.dumps(checks, indent=2))
        print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
