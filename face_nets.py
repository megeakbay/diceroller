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
# Digits
#
# The net draws its numbers with the *same* stroke paths the 3D renderer paints
# onto the faces, rather than with a system font. A reference sheet that used a
# different letterform from the pictures it describes would make the reader
# match two shapes before they could use it -- and the two are drawn by
# completely different machinery, so they diverged visibly: matplotlib's bold
# sans against the renderer's thin geometric strokes.
# ============================================================================

def draw_digit(ax, value: int, cx: float, cy: float, size: float,
               color: str = NET_INK, lw_scale: float = 0.11) -> None:
    """
    Draw `value` centred on (cx, cy), `size` tall, in the renderer's own hand.

    Uses the outlines the renderer bakes into the face texture -- the same
    `digit_outlines.json`, not a fresh read of the font. Reading the font again
    here gave a glyph that was very slightly different: baking flattens the
    curves, which shifts the bounding box, and the two ended up with aspect
    ratios of 0.768 and 0.850. Too small to see, but there is no reason for the
    reference sheet and the frames to derive their letterforms separately.
    """
    from matplotlib.patches import Polygon as MplPoly

    from blender_render import _load_outlines

    contours = _load_outlines().get(value, [])
    if not contours:
        return

    xs = [p[0] for c in contours for p in c]
    ys = [p[1] for c in contours for p in c]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    gw, gh = x1 - x0, y1 - y0
    if gw <= 0 or gh <= 0:
        return

    s = size / max(gw, gh)
    mx, my = x0 + gw / 2.0, y0 + gh / 2.0

    # Each contour drawn as its own polygon. The counters -- the hole in a 6,
    # both in an 8 -- are separate rings, so they are punched out by drawing
    # them in the face colour over the outer ring.
    rings = sorted(contours, key=lambda c: _ring_area(c), reverse=True)
    for i, c in enumerate(rings):
        pts = [((px - mx) * s + cx, (py - my) * s + cy) for px, py in c]
        ax.add_patch(MplPoly(pts, closed=True, zorder=6 + i,
                             facecolor=color if i == 0 else NET_FACE,
                             edgecolor="none"))


def _ring_area(contour) -> float:
    """Absolute area of a closed ring, used to find the outer one."""
    a = 0.0
    n = len(contour)
    for i in range(n):
        x1, y1 = contour[i]
        x2, y2 = contour[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


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


def _octahedron_layout():
    """
    Unfold the die from the pose it actually rests in.

    Rather than an arbitrary strip, this is the solid opened out from its
    resting face: that face in the middle, the three faces it touches folded
    down around it, then the three that touch the top face, and the top face
    itself on the outside. Every position is derived from the real adjacency
    graph, so the picture is the die's own construction rather than a
    convention imposed on it.

    Returns (value, [(x, y) x3]) with the corners already in place.
    """
    import octahedron as oct_

    verts = oct_.POSES[0]
    resting = oct_._bottom_face(verts)
    top = oct_.OPPOSITE[resting]

    def neighbours(i):
        si = set(oct_.FACES[i])
        return [j for j in range(8)
                if j != i and len(si & set(oct_.FACES[j])) == 2]

    ring1 = neighbours(resting)            # share an edge with the resting face
    ring2 = neighbours(top)                # share an edge with the top face

    # The central triangle, point up, with its three neighbours folded out
    # across its three edges.
    s = 1.0
    h = H * s
    centre = [(-s / 2, -h / 3), (s / 2, -h / 3), (0.0, 2 * h / 3)]

    layout = [(oct_.FACE_VALUES[resting], centre)]

    # Each edge of the centre triangle carries one ring1 face, reflected across
    # that edge so the two share it exactly.
    edges = [(centre[0], centre[1]), (centre[1], centre[2]), (centre[2], centre[0])]
    opposite_corner = [centre[2], centre[0], centre[1]]

    ring1_placed = {}
    for (a, b), apex, j in zip(edges, opposite_corner, ring1):
        outer = _reflect(apex, a, b)
        layout.append((oct_.FACE_VALUES[j], [a, b, outer]))
        ring1_placed[j] = (a, b, outer)

    # Each ring1 face has two free edges; one carries a ring2 face. Choose the
    # ring2 face that genuinely shares that edge on the solid.
    ring2_placed = {}
    for j, (a, b, outer) in ring1_placed.items():
        partners = [k for k in neighbours(j) if k in ring2 and k not in ring2_placed]
        if not partners:
            continue
        k = partners[0]
        # Fold across the edge (b, outer).
        far = _reflect(a, b, outer)
        layout.append((oct_.FACE_VALUES[k], [b, outer, far]))
        ring2_placed[k] = (b, outer, far)

    # The top face folds off whichever ring2 face still has a free edge.
    if ring2_placed:
        k, (a, b, outer) = next(iter(ring2_placed.items()))
        far = _reflect(a, b, outer)
        layout.append((oct_.FACE_VALUES[top], [b, outer, far]))

    return layout


def _reflect(p, a, b):
    """Mirror point `p` across the line through `a` and `b`."""
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    if L < 1e-12:
        return p
    t = ((px - ax) * dx + (py - ay) * dy) / L
    cx, cy = ax + dx * t, ay + dy * t
    return (2 * cx - px, 2 * cy - py)


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
    for value, pts in _octahedron_layout():
        ax.add_patch(Polygon(pts, closed=True, facecolor=NET_FACE,
                             edgecolor=NET_EDGE, linewidth=1.6))
        cx = sum(p[0] for p in pts) / 3.0
        cy = sum(p[1] for p in pts) / 3.0
        draw_digit(ax, value, cx, cy, size=0.15)


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
    # Unfolded from the pose the die rests in: the face on the board in the
    # middle, the four it touches folded out around it, and the top face
    # beyond one of them. Laid out as (col, row) in face widths.
    #
    # Reading it from the die's own state rather than hardcoding numbers keeps
    # the sheet true to the solid: change the starting orientation and the net
    # follows.
    faces = [
        (die.bottom, (1, 1)),      # resting on the board, at the centre
        (die.west, (0, 1)),
        (die.north, (1, 2)),
        (die.east, (2, 1)),
        (die.south, (1, 0)),
        (die.top, (3, 1)),         # opposite the resting face, on the outside
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
    else:
        draw_cube_net(ax)

    # No caption. The opposite-face rule is already stated in the prompts, so
    # repeating it here would only be words for a reader that is being shown a
    # picture -- the net is meant to be read as the die's shape, not annotated.
    ax.autoscale_view()
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
