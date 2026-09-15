"""
Rolling Dice dataset generator (per-level, per-variant).

Produces the MentisOculi output layout:

    output/<variant>/level_XX/puzzle_XXXX/
        initial.png        question image (step 0)
        cot_00.png ...     one render per roll
        metadata.json      board, path, per-step trace, answer

The level number equals the number of rolls, i.e. the number of reasoning
steps required, matching the MentisOculi difficulty convention.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt

# When true, main.py writes metadata only and leaves every image to Blender
# (`render_blender.py`). The matplotlib figures below are the original 2D
# renderer; they share filenames with the Blender output, so producing both
# means whichever runs last wins. --no-images keeps that from happening.
SKIP_IMAGES = False

from generator import (
    DIRECTION_NAMES,
    bottom_sum,
    generate_instance,
    render_state,
)

VARIANTS = ("top", "sum", "two")

# Chance performance per variant, used for level_metadata.json.
# top: 1 of 6 faces. sum: guessing an integer total is effectively lower, but
# we report the coarse bound. two: 1 of 3 labels x the total, dominated by 1/3.
CHANCE_PERFORMANCE = {"top": 1 / 6, "sum": 1 / 6, "two": 1 / 3}


def build_question(instance: Dict[str, Any]) -> str:
    """The natural-language question, phrased as in the MIRA paper."""
    variant = instance["variant"]
    if variant == "top":
        return (
            "If the dice is rolled on the showed path, what will be the number "
            "on the top?"
        )
    if variant == "sum":
        return (
            "If the die is rolled along the shown path, what is the total sum "
            "of the numbers on the bottom face that touches the path at each step?"
        )
    return (
        "When the die is rolled along the black and red paths respectively, "
        'which path yields a higher total sum of the numbers on the bottom face '
        'at each step (answer with "red", "black" or "same")? What is the total '
        "sum for that path? Note: both paths have the same length."
    )


def build_step_records(instance: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    One record per roll, carrying everything needed to render and to write a
    reasoning text later in the pipeline.
    """
    records: List[Dict[str, Any]] = []

    def records_for(trace: List[Dict[str, Any]], path: List[str], tag: str = None):
        running = 0
        out = []
        for i, direction in enumerate(path):
            before = trace[i]
            after = trace[i + 1]
            running += after["bottom"]
            out.append(
                {
                    "step": i + 1,
                    "path": tag,
                    "direction": direction,
                    "direction_name": DIRECTION_NAMES[direction],
                    "from_position": before["position"],
                    "to_position": after["position"],
                    "die_before": before["die"],
                    "die_after": after["die"],
                    "bottom_face": after["bottom"],
                    "running_bottom_sum": running,
                }
            )
        return out

    if instance["variant"] == "two":
        for tag in ("black", "red"):
            records.extend(
                records_for(instance["traces"][tag], instance["paths"][tag], tag)
            )
    else:
        records.extend(records_for(instance["trace"], instance["path"]))
    return records


def puzzle_signature(instance: Dict[str, Any]) -> str:
    """
    What makes two puzzles the same puzzle.

    Two seeds can land on one puzzle -- at level 1 there are only so many
    (cell, pose, direction) triples, and 50 draws from ~2400 of them collided
    twice in practice. A dataset must not repeat itself, so the generators
    below skip a signature they have already written and move to the next
    seed rather than counting it as produced.
    """
    return json.dumps(
        {
            "start": instance.get("start"),
            "path": instance.get("path"),
            "paths": instance.get("paths"),
            "initial_die": instance.get("initial_die"),
            "board": instance.get("board"),
            # The octahedron carries none of the cube's fields: its start
            # pose lives in trace[0], and leaving it out made two puzzles that
            # begin in different orientations -- showing different faces --
            # look like the same puzzle.
            "first_state": (instance.get("trace") or [None])[0],
        },
        sort_keys=True,
    )


def save_puzzle(
    instance: Dict[str, Any],
    level: int,
    puzzle_id: int,
    seed: int,
    output_dir: Path,
) -> None:
    """Save a single puzzle: question image, per-step renders, metadata."""
    variant = instance["variant"]
    level_dir = output_dir / variant / f"level_{level:02d}"
    puzzle_dir = level_dir / f"puzzle_{puzzle_id:04d}"
    puzzle_dir.mkdir(parents=True, exist_ok=True)

    # Question image: initial state with the full path shown
    if not SKIP_IMAGES:
        fig = render_state(instance, step=0)
        fig.savefig(puzzle_dir / "initial.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # One render per roll
    cot_files: List[str] = []
    if variant == "two":
        # Both paths share a start; render each path's progression separately
        for tag in ("black", "red"):
            sub = {
                "variant": "sum",
                "board": instance["board"],
                "start": instance["start"],
                "initial_die": instance["initial_die"],
                "path": instance["paths"][tag],
                "trace": instance["traces"][tag],
                "num_rolls": instance["num_rolls"],
                "answer": instance["sums"][tag],
            }
            for step_idx in range(1, len(instance["paths"][tag]) + 1):
                name = f"cot_{tag}_{step_idx - 1:02d}.png"
                if not SKIP_IMAGES:
                    fig = render_state(sub, step=step_idx)
                    fig.savefig(puzzle_dir / name, dpi=200, bbox_inches="tight")
                    plt.close(fig)
                cot_files.append(name)
    else:
        for step_idx in range(1, len(instance["path"]) + 1):
            name = f"cot_{step_idx - 1:02d}.png"
            if not SKIP_IMAGES:
                fig = render_state(instance, step=step_idx)
                fig.savefig(puzzle_dir / name, dpi=200, bbox_inches="tight")
                plt.close(fig)
            cot_files.append(name)

    steps = build_step_records(instance)

    metadata = {
        "puzzle_id": puzzle_id,
        "level": level,
        "variant": variant,
        "seed": seed,
        "question": build_question(instance),
        "num_cot_images": len(cot_files),
        "num_rolls": instance["num_rolls"],
        "initial_state_image": "initial.png",
        "cot_images": cot_files,
        "board": instance["board"],
        "start": instance["start"],
        "initial_die": instance["initial_die"],
        "answer": instance["answer"],
        "steps": steps,
    }
    if variant == "two":
        metadata["paths"] = instance["paths"]
        metadata["traces"] = instance["traces"]
        metadata["sums"] = instance["sums"]
    else:
        metadata["path"] = instance["path"]
        metadata["trace"] = instance["trace"]

    with open(puzzle_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def generate_variant(
    variant: str,
    min_level: int,
    max_level: int,
    instances_per_level: int,
    output_dir: Path,
    seed: int,
    num_blocked: int,
    max_attempts: int = 200000,
    excluded: Optional[set] = None,
) -> Dict[int, int]:
    """Generate all levels for one variant. Level == number of rolls."""
    produced = {level: 0 for level in range(min_level, max_level + 1)}
    current_seed = seed
    attempts = 0
    seen: set = set(excluded or ())

    print(f"\n  Variant '{variant}': levels {min_level}-{max_level} "
          f"({instances_per_level} each)")

    while any(produced[l] < instances_per_level for l in produced) and attempts < max_attempts:
        level = next(l for l in sorted(produced) if produced[l] < instances_per_level)
        attempts += 1
        try:
            instance = generate_instance(
                seed=current_seed,
                variant=variant,
                num_rolls=level,
                num_blocked=num_blocked,
            )
        except RuntimeError:
            current_seed += 1
            continue

        signature = puzzle_signature(instance)
        if signature in seen:
            current_seed += 1
            continue                 # this seed repeats one already written
        seen.add(signature)

        puzzle_id = produced[level] + 1
        save_puzzle(instance, level, puzzle_id, current_seed, output_dir)
        produced[level] += 1
        current_seed += 1

        if produced[level] % 10 == 0 or produced[level] == instances_per_level:
            print(f"    Level {level}: {produced[level]}/{instances_per_level}")

    for level in sorted(produced):
        level_dir = output_dir / variant / f"level_{level:02d}"
        if not level_dir.exists():
            continue
        with open(level_dir / "level_metadata.json", "w") as f:
            json.dump(
                {
                    "level": level,
                    "variant": variant,
                    "num_rolls": level,
                    "num_cot_images": level * (2 if variant == "two" else 1),
                    "instances_generated": produced[level],
                    "instances_requested": instances_per_level,
                    "chance_performance": CHANCE_PERFORMANCE[variant],
                },
                f,
                indent=2,
            )
    return produced


def save_octahedron_puzzle(instance, level, puzzle_id, output_dir):
    """Save one octahedron puzzle: question image, per-roll renders, metadata."""
    import octahedron as oct_

    puzzle_dir = output_dir / "octahedron" / f"level_{level:02d}" / f"puzzle_{puzzle_id:04d}"
    puzzle_dir.mkdir(parents=True, exist_ok=True)

    if not SKIP_IMAGES:
        fig = oct_.render_state(instance, step=0)
        fig.savefig(puzzle_dir / "initial.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    cot_files = []
    for step_idx in range(1, len(instance["trace"])):
        name = f"cot_{step_idx - 1:02d}.png"
        if not SKIP_IMAGES:
            fig = oct_.render_state(instance, step=step_idx)
            fig.savefig(puzzle_dir / name, dpi=200, bbox_inches="tight")
            plt.close(fig)
        cot_files.append(name)

    metadata = dict(instance)
    metadata.update({
        "puzzle_id": puzzle_id,
        "level": level,
        "variant": "octahedron",
        "question": ("If the die is rolled along the shown path, what number "
                     "will be face down at the end?"),
        "num_cot_images": len(cot_files),
        "initial_state_image": "initial.png",
        "cot_images": cot_files,
    })
    with open(puzzle_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def generate_octahedron(min_level, max_level, instances_per_level, output_dir,
                        seed, excluded=None):
    """Generate all levels of the octahedron variant. Level == number of rolls."""
    import octahedron as oct_

    print(f"\n  Variant 'octahedron': levels {min_level}-{max_level} "
          f"({instances_per_level} each)")
    produced = {}
    current_seed = seed
    seen: set = set(excluded or ())
    for level in range(min_level, max_level + 1):
        made = 0
        while made < instances_per_level:
            instance = oct_.generate_instance(seed=current_seed, num_rolls=level)
            current_seed += 1
            # A path can stall early if it corners itself; skip short ones.
            if instance["num_rolls"] < level:
                continue
            signature = puzzle_signature(instance)
            if signature in seen:
                continue             # this seed repeats one already written
            seen.add(signature)
            made += 1
            save_octahedron_puzzle(instance, level, made, output_dir)
        produced[level] = made
        print(f"    Level {level}: {made}/{instances_per_level}")

        level_dir = output_dir / "octahedron" / f"level_{level:02d}"
        with open(level_dir / "level_metadata.json", "w") as f:
            json.dump({
                "level": level,
                "variant": "octahedron",
                "num_rolls": level,
                "num_cot_images": level,
                "instances_generated": made,
                "instances_requested": instances_per_level,
                "chance_performance": 1 / 8,
            }, f, indent=2)
    return produced


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling Dice dataset generator")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instances", type=int, default=50,
                        help="Instances per level")
    parser.add_argument("--min-level", type=int, default=1)
    parser.add_argument("--max-level", type=int, default=5)
    parser.add_argument("--level", type=int, default=None,
                        help="Generate only this specific level")
    parser.add_argument("--variant", choices=[*VARIANTS, "octahedron", "all"],
                        default="all",
                        help="Cube variants, or 'octahedron' for the "
                             "eight-faced die on a triangular grid")
    parser.add_argument("--num-blocked", type=int, default=0,
                        help="Impassable cells per board")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--exclude", type=str, default=None,
                        help="a pool directory whose puzzles must not be "
                             "repeated here -- pass the train pool when "
                             "generating test, so the two never overlap")
    parser.add_argument("--no-images", action="store_true",
                        help="write metadata only; render every image with "
                             "render_blender.py instead")
    args = parser.parse_args()

    global SKIP_IMAGES
    SKIP_IMAGES = args.no_images

    # Puzzles written by an earlier run that this one must not repeat.
    #
    # The signature set that keeps one run free of duplicates dies with the
    # process, so generating test after train let the two overlap -- more than
    # half of one test set also appeared in train, which makes the benchmark
    # measure memorisation. Reading the other pool back closes that.
    excluded = set()
    if args.exclude:
        for meta_path in Path(args.exclude).glob("*/level_*/puzzle_*/metadata.json"):
            try:
                excluded.add(puzzle_signature(json.loads(meta_path.read_text())))
            except (OSError, json.JSONDecodeError):
                continue
        print(f"  excluding {len(excluded)} puzzle(s) from {args.exclude}")

    if args.level is not None:
        min_level = max_level = args.level
    else:
        min_level, max_level = args.min_level, args.max_level

    # 'all' means the three cube variants; the octahedron is opt-in, since it
    # is a sibling task rather than one of MIRA's own.
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    out = Path(args.output_dir)

    print("=" * 60)
    print("Rolling Dice - Dataset Generator")
    print("=" * 60)
    print(f"Variants: {', '.join(variants)}")
    print(f"Levels: {min_level}-{max_level}   Instances per level: {args.instances}")
    print(f"Blocked cells: {args.num_blocked}   Seed: {args.seed}")
    print("=" * 60)

    total = 0
    for variant in variants:
        if variant == "octahedron":
            stats = generate_octahedron(
                min_level=min_level,
                max_level=max_level,
                instances_per_level=args.instances,
                output_dir=out,
                seed=args.seed,
                excluded=excluded,
            )
        else:
            stats = generate_variant(
                variant=variant,
                min_level=min_level,
                max_level=max_level,
                instances_per_level=args.instances,
                output_dir=out,
                seed=args.seed,
                num_blocked=args.num_blocked,
                excluded=excluded,
            )
        total += sum(stats.values())

    print(f"\nTotal: {total} instances written to {out}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
