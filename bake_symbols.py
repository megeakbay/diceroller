"""
Bake the symbol shapes to `symbol_outlines.json`.

An alternative face marking to the digits: a die whose faces carry a heart, an
arrow, a triangle and so on instead of 1..8. Symbols are quicker to tell apart
than numerals at a glance, and unlike digits they cannot be confused by
rotation -- a 6 and a 9 are the same shape turned round, while a heart is a
heart whichever way up it sits.

The shapes are built from geometry here rather than taken from a font. A font's
dingbats vary between machines and would reintroduce exactly the portability
problem `bake_digits.py` documents, and these forms are simple enough that
defining them directly is both shorter and reproducible.

Output matches the digit format: a list of closed contours per symbol, filled
with an even-odd rule, so `blender_render` treats the two identically.

    python bake_symbols.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent / "symbol_outlines.json"

# The seven symbols, in the order face values 1..8 are assigned. Eight faces
# need eight marks, so the set is these seven plus a square.
SYMBOL_ORDER = ["heart", "arrow", "triangle", "moon", "star", "house",
                "circle", "square"]


def _circle(cx, cy, r, steps=64, start=0.0, end=360.0):
    return [(cx + r * math.cos(math.radians(a)),
             cy + r * math.sin(math.radians(a)))
            for a in (start + (end - start) * i / steps for i in range(steps + 1))]


def heart():
    """Two lobes over a point, drawn as one closed curve."""
    pts = []
    for i in range(101):
        t = math.pi * 2 * i / 100
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        pts.append((x / 32.0 + 0.5, y / 32.0 + 0.5))
    return [pts]


def arrow():
    """A block arrow pointing up."""
    return [[
        (0.50, 0.95), (0.90, 0.55), (0.66, 0.55), (0.66, 0.08),
        (0.34, 0.08), (0.34, 0.55), (0.10, 0.55),
    ]]


def triangle():
    return [[(0.50, 0.93), (0.93, 0.16), (0.07, 0.16)]]


def moon():
    """A crescent: a disc with a second disc cut out of one side."""
    outer = _circle(0.46, 0.50, 0.42, steps=72)
    inner = _circle(0.62, 0.50, 0.36, steps=72)
    # The inner ring is a separate contour, so even-odd fill removes it.
    return [outer, inner]


def star():
    """A five-pointed star."""
    pts = []
    for i in range(10):
        a = math.radians(90 + i * 36)
        r = 0.46 if i % 2 == 0 else 0.19
        pts.append((0.5 + r * math.cos(a), 0.5 + r * math.sin(a)))
    return [pts]


def house():
    """A square with a gable on top.

    The eaves overhang the walls only slightly. Drawn wider it read as an
    arrow rather than a house, which matters here: the two are in the same
    set and have to stay distinguishable at a glance.
    """
    return [[
        (0.50, 0.95), (0.92, 0.58), (0.84, 0.58), (0.84, 0.08),
        (0.16, 0.08), (0.16, 0.58), (0.08, 0.58),
    ]]


def circle():
    return [_circle(0.5, 0.5, 0.42, steps=72)]


def square():
    return [[(0.12, 0.12), (0.88, 0.12), (0.88, 0.88), (0.12, 0.88)]]


BUILDERS = {
    "heart": heart, "arrow": arrow, "triangle": triangle, "moon": moon,
    "star": star, "house": house, "circle": circle, "square": square,
}


def bake() -> None:
    out = {}
    for name in SYMBOL_ORDER:
        contours = BUILDERS[name]()
        # Normalise into a unit box, as the digits are, so the renderer's
        # fitting maths is the same for both.
        xs = [p[0] for c in contours for p in c]
        ys = [p[1] for c in contours for p in c]
        x0, y0 = min(xs), min(ys)
        span = max(max(xs) - x0, max(ys) - y0) or 1.0
        out[name] = [[[round((x - x0) / span, 5), round((y - y0) / span, 5)]
                      for x, y in c] for c in contours]
        print(f"  {name:9} {len(contours)} contour(s), "
              f"{sum(len(c) for c in contours)} points")

    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    bake()
