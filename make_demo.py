"""
Pick one puzzle per (variant, level) and copy it into demo/.

Twenty puzzles: four markings -- cube pips, cube symbols, octahedron digits,
octahedron symbols -- across five levels. Everything a puzzle needs travels
with it: the question frame, the per-roll frames, the reference net, the
metadata and whatever reasoning has been written so far.

    python make_demo.py                 # train split, seeded so it repeats
    python make_demo.py --seed 7        # a different draw
    python make_demo.py --split test
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

# The four markings, and the directory each split keeps them in.
VARIANTS = [
    ("cube_pin", "top", "kup + nokta"),
    ("cube_symbols", "top", "kup + sembol"),
    ("oct_digits", "octahedron", "8yuz + rakam"),
    ("oct_symbols", "octahedron", "8yuz + sembol"),
]
LEVELS = [1, 2, 3, 4, 5]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a demo folder")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="demo")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    index = []
    missing = []
    for key, solid, label in VARIANTS:
        src_root = Path(f"ds_{args.split}_{key}") / solid
        for level in LEVELS:
            level_dir = src_root / f"level_{level:02d}"
            puzzles = sorted(p for p in level_dir.glob("puzzle_*") if p.is_dir())
            if not puzzles:
                missing.append(f"{key}/level_{level:02d}")
                continue
            chosen = rng.choice(puzzles)

            dest = out / f"{key}_level_{level:02d}_{chosen.name}"
            shutil.copytree(chosen, dest)

            meta = json.loads((chosen / "metadata.json").read_text())
            reasoning = dest / "cot_reasoning.json"
            steps = 0
            if reasoning.exists():
                steps = len(json.loads(reasoning.read_text()).get("steps", []))
            index.append({
                "folder": dest.name,
                "marking": label,
                "variant": key,
                "solid": meta.get("solid", "cube"),
                "level": level,
                "source": str(chosen),
                "num_rolls": meta.get("num_rolls"),
                "answer": meta.get("answer"),
                "question": meta.get("question"),
                "net": meta.get("net_symbol_image") or meta.get("net_image"),
                "images": sorted(p.name for p in dest.glob("*.png")),
                "reasoning_steps": steps,
            })

    (out / "index.json").write_text(json.dumps(index, indent=2))

    print(f"{len(index)} puzzle -> {out}/\n")
    print(f"{'klasor':<44}{'isaret':<14}{'lvl':>4}{'kare':>6}{'cot':>5}")
    for row in index:
        print(f"{row['folder']:<44}{row['marking']:<14}{row['level']:>4}"
              f"{len(row['images']):>6}{row['reasoning_steps']:>5}")
    if missing:
        print(f"\nbulunamadi: {', '.join(missing)}")


if __name__ == "__main__":
    main()
