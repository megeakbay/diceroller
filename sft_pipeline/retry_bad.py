"""
Retry steps whose reasoning is missing, leaks how it was produced, states the
wrong faces, or was failed by the judge (Gemini). Writes back into
cot_reasoning.json.

Each retry feeds the previous attempt's failure reason back to the model, so a
step that failed for a specific reason is not simply resampled blind.

Usage:
    python sft_pipeline/retry_bad.py --output-dir output
    python sft_pipeline/retry_bad.py --output-dir output --max-attempts 3
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

from gemini_common import (  # noqa: E402
    DEFAULT_MODEL,
    generate_json,
    iter_puzzle_dirs,
    make_client,
    png_part,
    text_part,
)
from generate_reasoning import (  # noqa: E402
    OCTAHEDRON_SCHEMA,
    REASONING_SCHEMA,
    build_step_prompt,
    is_hint_leak,
    is_octahedron,
    make_octahedron_record,
    make_record,
    step_images,
    uses_symbols,
)

RETRY_NOTE = """
A previous attempt at this step was rejected for the following reason:
    {reason}

Write a new explanation that avoids that problem. Remember: reason forward from
the previous orientation and the rolling rule, and never refer to being shown
or told the resulting faces.
"""


def needs_retry(record: Dict[str, Any]) -> Optional[str]:
    """Return the failure reason if this step should be retried, else None."""
    if not record.get("reasoning"):
        return "no reasoning was produced"
    if record.get("hint_leak") or is_hint_leak(record["reasoning"]):
        return "the text revealed that the answer had been supplied"
    if record.get("faces_correct") is False:
        return "the stated top/bottom faces did not match the true orientation"
    verdict = record.get("verdict")
    if verdict and not verdict.get("passed"):
        return verdict.get("reason") or "the judge rejected the explanation"
    return None


def retry_step(
    client: Any,
    puzzle_dir: Path,
    metadata: Dict[str, Any],
    step_meta: Dict[str, Any],
    reason: str,
    model: str,
) -> Optional[Dict[str, Any]]:
    """Regenerate one step's reasoning. Returns the new record, or None."""
    before, after = step_images(puzzle_dir, metadata, step_meta)
    if not before.exists() or not after.exists():
        return None

    octa = is_octahedron(metadata)
    syms = uses_symbols(metadata)
    parts = [
        text_part("Board before this roll:"),
        png_part(before),
        text_part("Board after this roll:"),
        png_part(after),
        text_part(build_step_prompt(step_meta, octahedron=octa, symbols=syms)
                  + RETRY_NOTE.format(reason=reason)),
    ]

    try:
        parsed = generate_json(
            client, model, parts,
            # The octahedron task asks only for the bottom face.
            required_fields=(["reasoning", "bottom"] if octa
                             else ["reasoning", "top", "bottom"]),
            schema=OCTAHEDRON_SCHEMA if octa else REASONING_SCHEMA,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [error] {puzzle_dir.name} step {step_meta['step']}: {exc}",
              file=sys.stderr)
        return None

    return (make_octahedron_record(step_meta, parsed) if octa
            else make_record(step_meta, parsed))


def main() -> None:
    ap = argparse.ArgumentParser(description="Retry failed reasoning steps")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--variant", choices=["top", "sum", "two", "octahedron"],
                    default=None)
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-attempts", type=int, default=2,
                    help="Retry passes per step (default: 2)")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent puzzles (1 = the old sequential behaviour)")
    args = ap.parse_args()

    client = make_client()

    puzzle_dirs = list(iter_puzzle_dirs(Path(args.output_dir), args.variant, args.level))
    if args.limit:
        puzzle_dirs = puzzle_dirs[: args.limit]

    retried = fixed = still_bad = 0

    def retry_puzzle(puzzle_dir):
        """Regenerate this puzzle's failed steps. Returns (retried, fixed, bad)."""
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
        local_retried = local_fixed = local_bad = 0
        for idx, record in enumerate(data["steps"]):
            reason = needs_retry(record)
            if reason is None:
                continue
            step_meta = steps_meta.get((record.get("path"), record["step"]))
            if step_meta is None:
                continue

            local_retried += 1
            print(f"  retrying {puzzle_dir.name} step {record['step']} [{reason}]",
                  flush=True)

            for _ in range(args.max_attempts):
                new_record = retry_step(
                    client, puzzle_dir, metadata, step_meta, reason, args.model
                )
                if new_record is None:
                    continue
                new_reason = needs_retry(new_record)
                if new_reason is None:
                    new_record["retried"] = True
                    data["steps"][idx] = new_record
                    local_fixed += 1
                    break
                reason = new_reason
            else:
                local_bad += 1

        if local_retried:
            # Only this thread touches this file: one puzzle, one worker.
            reasoning_path.write_text(json.dumps(data, indent=2))
        return local_retried, local_fixed, local_bad

    def tally(result, label):
        nonlocal retried, fixed, still_bad
        if result is None:
            return
        r, f, b = result
        retried += r
        fixed += f
        still_bad += b
        if r:
            print(f"{label}: retried {r}, fixed {f}", flush=True)

    if args.workers <= 1:
        for i, puzzle_dir in enumerate(puzzle_dirs, 1):
            tally(retry_puzzle(puzzle_dir), f"[{i}/{len(puzzle_dirs)}] {puzzle_dir}")
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(retry_puzzle, d): d for d in puzzle_dirs}
            for fut in as_completed(futures):
                d = futures[fut]
                done += 1
                try:
                    tally(fut.result(), f"[{done}/{len(puzzle_dirs)}] {d}")
                except Exception as exc:  # noqa: BLE001 - one puzzle must not stop the run
                    print(f"[{done}/{len(puzzle_dirs)}] {d}: FAILED {exc}",
                          file=sys.stderr)

    print(f"\nRetried {retried} steps: {fixed} fixed, {still_bad} still failing.")


if __name__ == "__main__":
    main()
