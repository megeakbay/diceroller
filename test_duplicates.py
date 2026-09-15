"""
Checks that no two puzzles are the same. Run: python test_duplicates.py

Two things get checked, because they fail independently:

  1. The generators, by seed. Drawing N seeds should yield N distinct puzzles.
     This catches the failure directly: both generators once fixed the die's
     starting pose, so only the path varied, and a path of R rolls has at most
     a handful of forms at low R -- fifty level-1 octahedron puzzles held three
     distinct ones.

  2. A built dataset, by content and by pixels. Metadata duplicates mean the
     same puzzle was written twice; identical frames mean two puzzles render
     the same picture even if their metadata differs. The split check is the
     one that matters most: a test puzzle that also appears in train makes the
     benchmark measure memorisation.

Pass --dataset DIR to check a built dataset (default: ds, if present).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok' if ok else 'FAIL'}: {name} {detail}".rstrip())
    if not ok:
        FAILURES.append(f"{name} {detail}".strip())


def puzzle_signature(meta: dict) -> str:
    """What makes two puzzles the same puzzle."""
    return json.dumps(
        {
            "solid": meta.get("solid", "cube"),
            "start": meta.get("start"),
            "path": meta.get("path"),
            "board": meta.get("board"),
            "board_radius": meta.get("board_radius"),
            "initial_die": meta.get("initial_die"),
            "faces": meta.get("faces"),
            # The octahedron records its starting pose here and nowhere else.
            "first_state": (meta.get("trace") or [None])[0],
        },
        sort_keys=True,
    )


def _generate_pool(variant: str, levels: int, per_level: int,
                   seed: int, tmp: Path, exclude: Path | None = None) -> list[dict]:
    """Generate a pool the way production does, and read back what it wrote."""
    import subprocess

    cmd = [sys.executable, "main.py", "--variant", variant,
           "--min-level", "1", "--max-level", str(levels),
           "--instances", str(per_level), "--seed", str(seed),
           "--output-dir", str(tmp), "--no-images"]
    if exclude:
        cmd += ["--exclude", str(exclude)]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return [json.loads(f.read_text())
            for f in tmp.glob("*/level_*/puzzle_*/metadata.json")]


def test_cube_generator(per_level: int = 50) -> None:
    """
    A generated pool must hold no repeats.

    Driven through main.py rather than generator.generate_instance, because
    that is where the deduplication lives: two seeds can land on the same
    puzzle -- at level 1 there are only so many (cell, pose, direction)
    triples -- and the production loop skips a signature it has already
    written. Calling the generator directly measures the wrong layer and
    reports collisions the dataset would never contain.
    """
    print("\ncube pool")
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        pool = _generate_pool("top", 5, per_level, 1000, Path(td))
        sigs = {puzzle_signature(m) for m in pool}
        check(f"{len(pool)} puzzles, all distinct",
              len(sigs) == len(pool), f"{len(sigs)} distinct")
        for level in range(1, 6):
            at = [m for m in pool if m.get("level") == level]
            lsigs = {puzzle_signature(m) for m in at}
            check(f"level {level}: {len(at)} distinct",
                  len(lsigs) == len(at), f"{len(lsigs)} of {len(at)}")


def test_octahedron_pool(per_level: int = 20) -> None:
    """Same for the octahedron, which was the worse of the two."""
    print("\noctahedron pool")
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        pool = _generate_pool("octahedron", 5, per_level, 2000, Path(td))
        sigs = {puzzle_signature(m) for m in pool}
        check(f"{len(pool)} puzzles, all distinct",
              len(sigs) == len(pool), f"{len(sigs)} distinct")


def test_no_split_overlap(per_level: int = 10) -> None:
    """
    A test pool generated with --exclude must share nothing with train.

    Each run's signature set dies with the process, so without this the two
    pools overlapped badly -- 118 of 200 test puzzles also appeared in train,
    which turns the benchmark into a memorisation check.
    """
    print("\ntrain/test separation")
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        train = _generate_pool("top", 3, per_level * 2, 1000, root / "train")
        test = _generate_pool("top", 3, per_level, 500000, root / "test",
                              exclude=root / "train")
        a = {puzzle_signature(m) for m in train}
        b = {puzzle_signature(m) for m in test}
        check(f"train {len(a)} and test {len(b)} share nothing",
              not (a & b), f"{len(a & b)} shared")


def test_die_orientations() -> None:
    """All 24 poses, each a valid die."""
    print("\ndie orientations")
    import generator as g

    poses = g.all_orientations()
    check("24 orientations", len(poses) == 24, f"got {len(poses)}")
    keys = {(p.top, p.bottom, p.north, p.south, p.east, p.west) for p in poses}
    check("all distinct", len(keys) == len(poses))
    bad = [p for p in poses
           if p.top + p.bottom != 7 or p.north + p.south != 7 or p.east + p.west != 7]
    check("opposite faces sum to 7", not bad, f"{len(bad)} bad")
    check("every value reaches the top",
          sorted({p.top for p in poses}) == [1, 2, 3, 4, 5, 6])


def test_dataset(root: Path) -> None:
    """A built dataset: no repeats within a split, and no train/test overlap."""
    print(f"\ndataset at {root}")
    # A built dataset nests as <split_marking>/<solid>/level_*/puzzle_*, a raw
    # pool as <solid>/level_*/puzzle_*. Accept both so the check works before
    # the dataset is assembled as well as after.
    metas = sorted(root.glob("*/*/level_*/puzzle_*/metadata.json")) or \
        sorted(root.glob("*/level_*/puzzle_*/metadata.json"))
    if not metas:
        print("  skip: no puzzles found")
        return

    by_sig: dict[str, list[Path]] = collections.defaultdict(list)
    for f in metas:
        by_sig[puzzle_signature(json.loads(f.read_text()))].append(f)

    # Several markings deliberately share one puzzle pool, so the same puzzle
    # appearing in cube_pin and cube_symbols is intended. Repeats inside a
    # single directory are not.
    within = 0
    for paths in by_sig.values():
        dirs = [p.relative_to(root).parts[0] for p in paths]
        within += len(dirs) - len(set(dirs))
    # A flat pool has one directory per solid, so every repeat lands in it;
    # count outright duplicate signatures instead.
    if all(len(p.relative_to(root).parts) == 4 for p in metas):
        within = sum(len(v) - 1 for v in by_sig.values())
    check(f"{len(metas)} puzzles, no repeats inside one directory",
          within == 0, f"{within} extra copies")

    leaked = set()
    for paths in by_sig.values():
        splits = {p.relative_to(root).parts[0].split("_")[0] for p in paths}
        if "train" in splits and "test" in splits:
            leaked.update(p for p in paths
                          if p.relative_to(root).parts[0].startswith("test"))
    n_test = sum(1 for f in metas if f.relative_to(root).parts[0].startswith("test"))
    check("no test puzzle also appears in train",
          not leaked, f"{len(leaked)} of {n_test} test puzzles leak")


def test_dataset_pixels(root: Path) -> None:
    """
    No two frames show the same picture.

    Compared on decoded pixels, not on file bytes. PNG encodes the same image
    to different bytes depending on how it compressed, so hashing the files
    misses real duplicates -- in a 400-frame sample, 26 frames were pixel-for-
    pixel identical to another while every file hash was unique.

    This is the check metadata cannot make: two puzzles that begin in
    different poses can roll into the same states and render the same
    pictures, and a model would see the repeat even though the records differ.
    """
    print(f"\nframes at {root}")
    frames = [p for p in (list(root.glob("*/*/level_*/puzzle_*/*.png"))
                          or list(root.glob("*/level_*/puzzle_*/*.png")))
              if not p.name.startswith("net")]
    if not frames:
        print("  skip: no frames rendered")
        return

    try:
        from PIL import Image
    except ImportError:
        print("  skip: PIL not available (needs the conda python, not Blender's)")
        return

    by_pixels: dict[str, list[Path]] = collections.defaultdict(list)
    for p in frames:
        with Image.open(p) as im:
            digest = hashlib.md5(im.convert("RGB").tobytes()).hexdigest()
        by_pixels[digest].append(p)

    dups = {h: ps for h, ps in by_pixels.items() if len(ps) > 1}
    extra = sum(len(ps) - 1 for ps in dups.values())
    detail = ""
    if dups:
        a, b = next(iter(dups.values()))[:2]
        detail = (f"{extra} duplicates, e.g. "
                  f"{a.relative_to(root)} == {b.relative_to(root)}")
    check(f"{len(frames)} frames are pixel-distinct", not dups, detail)


def main() -> None:
    ap = argparse.ArgumentParser(description="Check that no puzzles repeat")
    ap.add_argument("--dataset", default="ds",
                    help="built dataset to check (default: ds)")
    ap.add_argument("--skip-pixels", action="store_true",
                    help="skip frame hashing, which reads every PNG")
    args = ap.parse_args()

    print("duplicate checks")
    print("=" * 60)
    test_die_orientations()
    test_cube_generator()
    test_octahedron_pool()
    test_no_split_overlap()

    root = Path(args.dataset)
    if root.is_dir():
        test_dataset(root)
        if not args.skip_pixels:
            test_dataset_pixels(root)
    else:
        print(f"\nno dataset at {root}: generator checks only")

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
