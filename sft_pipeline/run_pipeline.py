"""
End-to-end orchestrator for the rolling-dice SFT pipeline.

Runs sequentially:
  1. generate_reasoning.py  -- write a reasoning text for every roll
  2. judge_inline.py        -- judge each step one-by-one
  3. retry_bad.py           -- regenerate the steps that failed
  4. judge_inline.py        -- re-judge the regenerated steps
  5. build_dataset.py       -- collect the passing steps into a dataset

Each subprocess inherits stdout/stderr so progress streams live. Stops on the
first non-zero exit. Re-running is idempotent per step (generation skips filled
work, the judge skips already-judged steps, retry only touches failures).

Usage:
    python sft_pipeline/run_pipeline.py --output-dir output
    python sft_pipeline/run_pipeline.py --output-dir output --variant top
    python sft_pipeline/run_pipeline.py --output-dir output --skip reasoning
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(step: str, cmd: list) -> None:
    print(f"\n{'=' * 60}\n[{step}] {' '.join(cmd)}\n{'=' * 60}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"[{step}] failed with exit code {result.returncode}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the rolling-dice SFT pipeline")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--variant", choices=["top", "sum", "two"], default=None)
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--skip", choices=["reasoning", "judge", "retry", "dataset"],
                    action="append", default=[],
                    help="Skip a stage (repeatable)")
    args = ap.parse_args()

    python = sys.executable
    common = ["--output-dir", args.output_dir, "--model", args.model]
    if args.variant:
        common += ["--variant", args.variant]
    if args.level:
        common += ["--level", str(args.level)]
    if args.limit:
        common += ["--limit", str(args.limit)]

    if "reasoning" not in args.skip:
        run("reasoning", [python, str(HERE / "generate_reasoning.py"), *common])

    if "judge" not in args.skip:
        run("judge", [python, str(HERE / "judge_inline.py"), *common])

    if "retry" not in args.skip:
        run("retry", [python, str(HERE / "retry_bad.py"), *common])
        # Re-judge whatever the retry pass rewrote
        run("rejudge", [python, str(HERE / "judge_inline.py"), *common])

    if "dataset" not in args.skip:
        dataset_cmd = [python, str(HERE / "build_dataset.py"),
                       "--output-dir", args.output_dir]
        if args.variant:
            dataset_cmd += ["--variant", args.variant]
        run("dataset", dataset_cmd)

    print("\nAll steps completed.")


if __name__ == "__main__":
    main()
