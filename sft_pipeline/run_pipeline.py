"""
End-to-end orchestrator for the rolling-dice SFT pipeline.

Runs sequentially, per output directory:
  1. generate_reasoning.py  -- write a reasoning text for every roll
  2. judge_inline.py        -- judge each step
  3. retry_bad.py           -- regenerate the steps that failed
  4. judge_inline.py        -- re-judge the regenerated steps
  5. build_dataset.py       -- collect the passing steps into a dataset

Each subprocess inherits stdout/stderr so progress streams live. Re-running is
idempotent per stage: generation skips filled work, the judge skips already
judged steps, retry only touches failures. So an interrupted run is resumed by
repeating the same command.

Profiles (--profile) name the combinations worth having:

    full     every stage (the default)
    fast     reasoning only -- no judge, no retry, no dataset. Halves the API
             calls and leaves a complete but unvetted set of reasoning texts.
    judge    judge and retry an existing set, then build the dataset
    retry    regenerate the steps the judge rejected, and re-judge them
    dataset  build the dataset from whatever already passes

Usage:
    # one directory, every stage
    python sft_pipeline/run_pipeline.py --output-dir output

    # the eight production directories, reasoning only, 12 at a time
    python sft_pipeline/run_pipeline.py --production --profile fast --workers 12

    # judge what a --profile fast run left behind
    python sft_pipeline/run_pipeline.py --production --profile judge
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The eight directories build_dataset_full.sh produces, and the variant each
# one holds. Cube and octahedron puzzles differ in more than their markings, so
# every stage downstream needs to be told which it is looking at.
# All eight live under ds/, so a dataset is one directory to copy or ignore.
PRODUCTION_DIRS = [
    ("ds/train_cube_pin", "top"),
    ("ds/train_cube_symbols", "top"),
    ("ds/train_oct_digits", "octahedron"),
    ("ds/train_oct_symbols", "octahedron"),
    ("ds/test_cube_pin", "top"),
    ("ds/test_cube_symbols", "top"),
    ("ds/test_oct_digits", "octahedron"),
    ("ds/test_oct_symbols", "octahedron"),
]

# A profile is just the set of stages it skips.
PROFILES = {
    "full": [],
    "fast": ["judge", "retry", "dataset"],
    "judge": ["reasoning"],
    "retry": ["reasoning", "judge", "dataset"],
    "dataset": ["reasoning", "judge", "retry"],
}


def run(step: str, cmd: list, keep_going: bool = False) -> bool:
    print(f"\n{'=' * 60}\n[{step}] {' '.join(str(c) for c in cmd)}\n{'=' * 60}",
          flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        if keep_going:
            print(f"[{step}] failed with exit code {result.returncode}, continuing",
                  file=sys.stderr, flush=True)
            return False
        sys.exit(f"[{step}] failed with exit code {result.returncode}")
    return True


# The order stages run in. `retry` implies the re-judge that follows it.
STAGE_ORDER = ["reasoning", "judge", "retry", "dataset"]


def stage_command(stage: str, output_dir: str, variant: str | None, args) -> list:
    """The command line for one stage against one directory."""
    python = sys.executable
    common = ["--output-dir", output_dir, "--model", args.model]
    if variant:
        common += ["--variant", variant]
    if args.level:
        common += ["--level", str(args.level)]
    if args.limit:
        common += ["--limit", str(args.limit)]
    concurrent = ["--workers", str(args.workers)]

    if stage == "reasoning":
        return [python, str(HERE / "generate_reasoning.py"), *common, *concurrent]
    if stage == "judge":
        return [python, str(HERE / "judge_inline.py"), *common, *concurrent]
    if stage == "retry":
        return [python, str(HERE / "retry_bad.py"), *common, *concurrent]
    if stage == "dataset":
        # One file per directory. build_dataset.py defaults to dataset.jsonl,
        # so running it over eight directories in turn had each overwrite the
        # last -- the run finished with 48 records instead of 1200.
        cmd = [python, str(HERE / "build_dataset.py"), "--output-dir", output_dir,
               "--dataset", str(Path(output_dir).parent / f"{Path(output_dir).name}.jsonl")]
        if variant:
            cmd += ["--variant", variant]
        return cmd
    raise ValueError(f"unknown stage: {stage}")


def run_stage(stage: str, output_dir: str, variant: str | None, args) -> None:
    """Run one stage, re-judging afterwards when it was the retry pass."""
    run(stage, stage_command(stage, output_dir, variant, args), args.keep_going)
    if stage == "retry":
        run("rejudge", stage_command("judge", output_dir, variant, args),
            args.keep_going)


def run_one(output_dir: str, variant: str | None, args, skip: list) -> None:
    """Run the pipeline over a single output directory."""
    python = sys.executable
    common = ["--output-dir", output_dir, "--model", args.model]
    if variant:
        common += ["--variant", variant]
    if args.level:
        common += ["--level", str(args.level)]
    if args.limit:
        common += ["--limit", str(args.limit)]

    # Only the two Gemini-bound stages take --workers; retry and build do not.
    concurrent = ["--workers", str(args.workers)]

    if "reasoning" not in skip:
        run("reasoning", [python, str(HERE / "generate_reasoning.py"),
                          *common, *concurrent], args.keep_going)

    if "judge" not in skip:
        run("judge", [python, str(HERE / "judge_inline.py"),
                      *common, *concurrent], args.keep_going)

    if "retry" not in skip:
        run("retry", [python, str(HERE / "retry_bad.py"), *common, *concurrent],
            args.keep_going)
        run("rejudge", [python, str(HERE / "judge_inline.py"),
                        *common, *concurrent], args.keep_going)

    if "dataset" not in skip:
        cmd = [python, str(HERE / "build_dataset.py"), "--output-dir", output_dir]
        if variant:
            cmd += ["--variant", variant]
        run("dataset", cmd, args.keep_going)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the rolling-dice SFT pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Profiles: " + ", ".join(f"{k} (skips: {v or 'nothing'})"
                                        for k, v in PROFILES.items()))
    ap.add_argument("--output-dir",
                    help="a single directory to process")
    ap.add_argument("--production", action="store_true",
                    help="process the eight ds_train_*/ds_test_* directories")
    ap.add_argument("--variant",
                    choices=["top", "sum", "two", "octahedron"], default=None)
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent Gemini requests (1 = sequential)")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="full")
    ap.add_argument("--skip",
                    choices=["reasoning", "judge", "retry", "dataset"],
                    action="append", default=[],
                    help="skip a stage on top of the profile (repeatable)")
    ap.add_argument("--keep-going", action="store_true",
                    help="carry on after a failed stage instead of stopping")
    args = ap.parse_args()

    if bool(args.output_dir) == bool(args.production):
        ap.error("pass exactly one of --output-dir or --production")

    skip = sorted(set(PROFILES[args.profile]) | set(args.skip))
    print(f"profile={args.profile}  workers={args.workers}"
          f"  skipping={skip or 'nothing'}")

    if args.production:
        targets = [(d, v) for d, v in PRODUCTION_DIRS if Path(d).is_dir()]
        if not targets:
            sys.exit("No ds_train_*/ds_test_* directories found. "
                     "Run build_dataset_full.sh first.")
        missing = [d for d, _ in PRODUCTION_DIRS if not Path(d).is_dir()]
        if missing:
            print(f"note: skipping {len(missing)} absent directory(ies): "
                  f"{', '.join(missing)}")

        # Stage-major, not directory-major: every directory finishes reasoning
        # before any starts judging, and so on. Directory-major interleaved the
        # stages, so a progress count that only moves during its own stage kept
        # looking stalled, and a crash late in one directory left the rest with
        # nothing done at all.
        for stage in STAGE_ORDER:
            if stage in skip:
                continue
            print(f"\n{'#' * 60}\n# STAGE: {stage.upper()}  "
                  f"({len(targets)} directories)\n{'#' * 60}", flush=True)
            for i, (d, v) in enumerate(targets, 1):
                print(f"\n--- [{i}/{len(targets)}] {d} ({v}) ---", flush=True)
                run_stage(stage, d, v, args)
    else:
        run_one(args.output_dir, args.variant, args, skip)

    print("\nAll steps completed.")


if __name__ == "__main__":
    main()
