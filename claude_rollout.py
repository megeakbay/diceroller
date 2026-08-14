"""
Zero-shot text-only baseline for the Rolling Dice task, on Claude.

The Claude counterpart to text_only_rollout.py: identical puzzle serialization
and prompt, so results from the two are directly comparable across providers.
The board, the die's visible faces, and the roll sequence are given as plain
text, and nothing about the kinematics is given away.

Like the Gemini version, this measures the mental-rotation half of the task in
isolation. Reading the path off an isometric render is removed, so the numbers
are NOT comparable to the image-based settings.

Setup:
    pip install anthropic
    # then either export a key
    export ANTHROPIC_API_KEY=sk-ant-...
    # or add it to the project .env alongside GEMINI_API_KEY:
    #     ANTHROPIC_API_KEY=sk-ant-...

Usage:
    python claude_rollout.py --output-dir output --variant top --level 5
    python claude_rollout.py --puzzle output/top/level_05/puzzle_0001
    python claude_rollout.py --effort low --results claude_low.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The serialization and answer parsing are shared with the Gemini runner so the
# two settings differ only in which model is called.
from text_only_rollout import (  # noqa: E402
    PROMPT_PATH,
    build_prompt,
    extract_answer,
)
from sft_pipeline.gemini_common import iter_puzzle_dirs, load_dotenv  # noqa: E402

import anthropic  # noqa: E402

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_EFFORT = "high"
MAX_RETRIES = 4
RETRY_BACKOFF = 5.0

# Thinking is on by default on Claude Opus 5 and is capped at `high` effort when
# disabled, so the no-thinking condition is only offered below that ceiling.
NO_THINKING_MAX_EFFORT = {"low", "medium", "high"}


class MissingCredentials(RuntimeError):
    """No Anthropic credentials could be resolved."""


def make_client() -> anthropic.Anthropic:
    """
    Build a Claude client.

    Reads ANTHROPIC_API_KEY from the environment, falling back to the project
    .env that already holds GEMINI_API_KEY. A bare client also picks up an
    `ant auth login` profile, so an absent key is not necessarily fatal.
    """
    load_dotenv()
    return anthropic.Anthropic()


def call_model(
    client: anthropic.Anthropic,
    model: str,
    prompt: str,
    effort: str,
    thinking: bool,
) -> Dict[str, str]:
    """
    One zero-shot call. Retries transient failures.

    Returns the visible answer text and, separately, the model's reasoning
    summary. The raw chain of thought is never returned by the API; asking for
    `display: "summarized"` is the most that can be pulled, and it does not
    change what the model was asked to do -- the prompt is untouched, so the
    setting stays zero-shot.
    """
    kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": 16000,
        "output_config": {"effort": effort},
        "messages": [{"role": "user", "content": prompt}],
    }
    if thinking:
        kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
    else:
        kwargs["thinking"] = {"type": "disabled"}

    backoff = RETRY_BACKOFF
    last_err: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.messages.create(**kwargs)

            # Safety classifiers can decline a request; content is empty or
            # partial, so check before reading it.
            if resp.stop_reason == "refusal":
                raise RuntimeError("request was refused by safety classifiers")

            text = "".join(b.text for b in resp.content if b.type == "text")
            reasoning = "\n".join(
                b.thinking for b in resp.content
                if b.type == "thinking" and getattr(b, "thinking", "")
            )
            if not text:
                raise RuntimeError(f"empty response (stop_reason={resp.stop_reason})")
            return {"text": text, "reasoning": reasoning}
        except (anthropic.BadRequestError, anthropic.AuthenticationError):
            # Malformed request or bad credentials — retrying cannot help.
            raise
        except TypeError as exc:
            # No credentials resolved at all: the SDK raises a bare TypeError
            # before any request is made. Also terminal, so don't retry.
            if "authentication" in str(exc).lower():
                raise MissingCredentials(str(exc)) from exc
            raise
        except Exception as exc:  # noqa: BLE001 - retry on anything transient
            last_err = exc
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff *= 2

    raise RuntimeError(f"messages.create failed after {MAX_RETRIES} attempts: {last_err}")


def run_puzzle(
    client: anthropic.Anthropic,
    model: str,
    puzzle_dir: Path,
    template: str,
    effort: str,
    thinking: bool,
) -> Dict[str, Any]:
    meta = json.loads((puzzle_dir / "metadata.json").read_text())
    prompt = build_prompt(meta, template)

    out = call_model(client, model, prompt, effort, thinking)
    predicted = extract_answer(out["text"])

    return {
        "puzzle_id": meta["puzzle_id"],
        "level": meta["level"],
        "variant": meta["variant"],
        "puzzle_dir": str(puzzle_dir),
        "setting": "text_only_zero_shot",
        "provider": "anthropic",
        "model": model,
        "effort": effort,
        "thinking": thinking,
        "prompt": prompt,
        "response": out["text"],
        # Summarized reasoning, when the API returns any. Verification reads
        # this alongside `response`, so working shown only here still counts.
        "reasoning": out["reasoning"],
        "predicted": predicted,
        "expected": meta["answer"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Zero-shot text-only rollout for Rolling Dice, on Claude"
    )
    ap.add_argument("--output-dir", default="output", help="Dataset root")
    ap.add_argument("--puzzle", default=None, help="Run a single puzzle directory")
    ap.add_argument("--variant", default=None, choices=["top", "sum", "two"])
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--effort",
        default=DEFAULT_EFFORT,
        choices=["low", "medium", "high", "xhigh", "max"],
        help="Reasoning depth (default: high)",
    )
    ap.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable thinking, for the true zero-shot condition "
             "(only valid at effort high or below)",
    )
    ap.add_argument("--results", default="claude_results.json")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the prompts without calling the model",
    )
    args = ap.parse_args()

    thinking = not args.no_thinking
    if not thinking and args.effort not in NO_THINKING_MAX_EFFORT:
        sys.exit(
            f"--no-thinking is only valid at effort high or below "
            f"(got {args.effort}). Drop the flag, or lower --effort."
        )

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

    client = make_client()

    results: List[Dict[str, Any]] = []
    for i, puzzle_dir in enumerate(puzzle_dirs, 1):
        print(f"[{i}/{len(puzzle_dirs)}] {puzzle_dir} ... ", end="", flush=True)
        try:
            row = run_puzzle(
                client, args.model, puzzle_dir, template, args.effort, thinking
            )
        except (MissingCredentials, anthropic.AuthenticationError):
            sys.exit(
                "\nNo valid Anthropic credentials. Set ANTHROPIC_API_KEY in the "
                "environment, or add it to the project .env alongside "
                "GEMINI_API_KEY:\n"
                "    ANTHROPIC_API_KEY=sk-ant-..."
            )
        except Exception as exc:  # noqa: BLE001 - record and continue
            print(f"FAILED: {exc}")
            continue
        ok = row["predicted"] == row["expected"]
        print(f"predicted {row['predicted']}, expected {row['expected']} "
              f"{'OK' if ok else 'X'}")
        results.append(row)

    Path(args.results).write_text(json.dumps(results, indent=2))

    if results:
        correct = sum(1 for r in results if r["predicted"] == r["expected"])
        print("-" * 70)
        print(f"  {correct}/{len(results)} = {correct / len(results):.1%}")
    else:
        print("-" * 70)
        print("  no results")
    print(f"Wrote {args.results}")


if __name__ == "__main__":
    main()
