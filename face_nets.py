"""
Reference nets for the two dice.

Each puzzle gets a `net.png` showing its die unfolded flat: every face, its
value, and which faces touch which. This is the die's *construction* -- fixed
for the whole dataset -- not its current orientation, so it tells a model how
the solid is built without telling it which way up the solid is now.

That distinction is what makes the net safe to include. The prompts already
state the opposite-face rule (sum to 7 on the cube, 9 on the octahedron), and
`face_layout` already names the hidden values at every step; what neither gives
is the adjacency, which is exactly what a net shows. A model still has to track
the rolls to know where those faces have ended up.

Adjacency is read out of the live geometry rather than hand-drawn, so a net
cannot drift from the solid the renderer actually builds.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from generator import BOARD_EDGE, DIE_BODY, DIE_EDGE, PIP_COLOR, _PIP_LAYOUT

NET_FACE = DIE_BODY
NET_EDGE = DIE_EDGE
NET_INK = PIP_COLOR


# ============================================================================
# Octahedron
# ============================================================================

def octahedron_adjacency() -> Dict[int, List[int]]:
    """value -> the three values whose faces share an edge with it."""
    import octahedron as oct_

    out: Dict[int, List[int]] = {}
    for i in range(8):
        si = set(oct_.FACES[i])
        nb = [oct_.FACE_VALUES[j] for j in range(8)
              if j != i and len(si & set(oct_.FACES[j])) == 2]
        out[oct_.FACE_VALUES[i]] = sorted(nb)
    return out


def _octahedron_layout() -> List[Tuple[int, Tuple[float, float], bool]]:
    """
    Place the eight triangles in the standard two-strip net.

    Returns (value, (cx, cy) of the triangle's centroid, points_up).

    The arrangement is the usual one for a D8: two rows of four, each row a
    strip of alternating up- and down-pointing triangles, with the rows offset
    so that folding brings the matching edges together.
    """
    import octahedron as oct_

    adj = octahedron_adjacency()

    # Walk the adjacency graph to lay a strip out, so the drawing matches the
    # solid rather than a guess about which faces are neighbours.
    top_face = 1
    top_strip = [top_face]
    while len(top_strip) < 4:
        cur = top_strip[-1]
        nxt = [v for v in adj[cur]
               if v not in top_strip and _opposite_value(v) not in top_strip]
        if not nxt:
            nxt = [v for v in adj[cur] if v not in top_strip]
        top_strip.append(nxt[0])

    bottom_strip = [_opposite_value(v) for v in top_strip]

    # Two strips, the lower one offset by one half-triangle so the folded
    # edges line up, which is the standard D8 net.
    layout = []
    for i, v in enumerate(top_strip):
        layout.append((v, i, 0.0, i % 2 == 0))
    for i, v in enumerate(bottom_strip):
        layout.append((v, i + 1, -H, i % 2 == 1))
    return layout


def _opposite_value(value: int) -> int:
    import octahedron as oct_
    for fi, v in oct_.FACE_VALUES.items():
        if v == value:
            return oct_.FACE_VALUES[oct_.OPPOSITE[fi]]
    raise KeyError(value)


H = 0.8660254037844386          # height of a unit equilateral triangle


def _triangle(index: int, row_y: float, up: bool, size: float = 1.0):
    """
    Corners of the `index`-th triangle in a strip whose baseline is `row_y`.

    Built from the strip's own lattice rather than from a centre point. Two
    triangles in a strip share a full edge, so their corners have to be the
    same two points -- placing each one around its own centroid, as an earlier
    version did, left them overlapping and gapped because the up and down
    forms were mirrored about different baselines.
    """
    half = size / 2.0
    x = index * half
    if up:
        return [(x, row_y), (x + size, row_y), (x + half, row_y + H * size)]
    return [(x, row_y + H * size), (x + size, row_y + H * size),
            (x + half, row_y)]


def _triangle_centre(index: int, row_y: float, up: bool, size: float = 1.0):
    """Centroid of that triangle, for placing its number."""
    pts = _triangle(index, row_y, up, size)
    return (sum(p[0] for p in pts) / 3.0, sum(p[1] for p in pts) / 3.0)


def draw_octahedron_net(ax) -> None:
    """The eight-faced die, unfolded."""
    for value, index, row_y, up in _octahedron_layout():
        pts = _triangle(index, row_y, up)
        ax.add_patch(Polygon(pts, closed=True, facecolor=NET_FACE,
                             edgecolor=NET_EDGE, linewidth=1.6))
        cx, cy = _triangle_centre(index, row_y, up)
        ax.text(cx, cy, str(value), ha="center", va="center",
                fontsize=17, fontweight="bold", color=NET_INK)


# ============================================================================
# Cube
# ============================================================================

def draw_cube_net(ax) -> None:
    """
    The six-faced die, unfolded in a cross, with pips rather than numerals.

    The cube's faces carry pips in the render, so the net carries pips too --
    a reference sheet that used a different notation from the pictures would
    make the reader translate before they could use it.
    """
    from generator import canonical_die

    die = canonical_die()
    # Cross layout: the four sides in a row, top and bottom above and below the
    # second column. Positions are (col, row) in face widths.
    faces = [
        (die.north, (1, 2)),
        (die.west, (0, 1)),
        (die.top, (1, 1)),
        (die.east, (2, 1)),
        (die.bottom, (3, 1)),
        (die.south, (1, 0)),
    ]
    for value, (col, row) in faces:
        x0, y0 = col * 1.0, row * 1.0
        ax.add_patch(Polygon(
            [(x0, y0), (x0 + 1, y0), (x0 + 1, y0 + 1), (x0, y0 + 1)],
            closed=True, facecolor=NET_FACE, edgecolor=NET_EDGE, linewidth=1.6))
        for px, py in _PIP_LAYOUT[value]:
            ax.add_patch(plt.Circle((x0 + px, y0 + py), 0.075,
                                    color=NET_INK, zorder=5))


# ============================================================================
# Rendering
# ============================================================================

def render_net(variant: str, out_path: Path) -> None:
    """Write the reference net for `variant` to `out_path`."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_aspect("equal")
    ax.axis("off")

    if variant == "octahedron":
        draw_octahedron_net(ax)
        opposites = sorted({tuple(sorted((v, _opposite_value(v))))
                            for v in range(1, 9)})
        caption = ("Faces of the eight-sided die, unfolded.  "
                   "Opposite faces sum to 9:  "
                   + ",  ".join(f"{a}+{b}" for a, b in opposites))
    else:
        draw_cube_net(ax)
        caption = ("Faces of the six-sided die, unfolded.  "
                   "Opposite faces sum to 7:  1+6,  2+5,  3+4")

    ax.autoscale_view()
    ax.set_title(caption, fontsize=8.5, color=BOARD_EDGE, pad=12, wrap=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_nets(output_dir: Path, variant: str = "all",
               filename: str = "net.png") -> int:
    """
    Write the reference net into every puzzle directory under `output_dir`.

    The net depends only on which die a puzzle uses, not on its board, path or
    step, so it is rendered once per variant and copied. Regenerating it per
    puzzle would be slower and could not differ.
    """
    import shutil

    root = Path(output_dir)
    if not root.is_dir():
        raise SystemExit(f"No such output directory: {root}")

    variants = ([variant] if variant != "all"
                else [d.name for d in sorted(root.iterdir()) if d.is_dir()])

    written = 0
    for v in variants:
        vdir = root / v
        if not vdir.is_dir():
            continue
        puzzles = sorted(p for p in vdir.glob("level_*/puzzle_*") if p.is_dir())
        if not puzzles:
            continue

        master = vdir / filename
        render_net(v, master)
        for p in puzzles:
            shutil.copyfile(master, p / filename)
            written += 1

            # Record it in the puzzle's metadata, so a consumer finds the net
            # the same way it finds the frames rather than guessing a name.
            meta_path = p / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                if meta.get("net_image") != filename:
                    meta["net_image"] = filename
                    with open(meta_path, "w") as f:
                        json.dump(meta, f, indent=2)
        print(f"  {v}: {filename} in {len(puzzles)} puzzle(s)")
    return written


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Write a reference net into every puzzle directory")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--variant", default="all",
                    help="top, sum, two, octahedron, or all")
    ap.add_argument("--filename", default="net.png")
    args = ap.parse_args()

    n = write_nets(Path(args.output_dir), args.variant, args.filename)
    print(f"\nWrote {n} net image(s).")


if __name__ == "__main__":
    main()
