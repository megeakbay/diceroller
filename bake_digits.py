"""
Bake the digit glyphs to `digit_outlines.json`.

The renderer runs inside Blender, whose bundled Python has no matplotlib, so it
cannot read a font at render time. Baking the outlines once here gives it
properly designed letterforms with no runtime dependency -- and makes the
shapes reproducible, rather than depending on whatever fonts a machine happens
to have installed.

Run after changing the font:

    python bake_digits.py
"""
from __future__ import annotations

import json
from pathlib import Path

from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MPath
from matplotlib.textpath import TextPath

FAMILY = "DejaVu Sans"     # ships with matplotlib itself
WEIGHT = "bold"
OUT = Path(__file__).resolve().parent / "digit_outlines.json"


def bake() -> None:
    prop = FontProperties(family=FAMILY, weight=WEIGHT)
    out = {}
    for d in range(10):
        # Flattened at a high resolution: `iter_segments` with curves=False
        # subdivides the beziers for us, and the outlines are only a few KB, so
        # there is no reason to be sparing. At the default flattening the
        # straight segments between control points were visible as facets on a
        # rendered face.
        tp = TextPath((0.0, 0.0), str(d), size=200.0, prop=prop)
        # Each closed contour is kept separate so the counters survive -- the
        # hole in a 6, both holes in an 8. The renderer fills them with an
        # even-odd rule, which needs them as distinct rings.
        polys, cur = [], []
        for pt, code in tp.iter_segments(curves=False):
            if code == MPath.MOVETO:
                if len(cur) > 2:
                    polys.append(cur)
                cur = [(pt[0], pt[1])]
            elif code == MPath.LINETO:
                cur.append((pt[0], pt[1]))
            elif code in (MPath.CURVE3, MPath.CURVE4):
                for i in range(0, len(pt), 2):
                    cur.append((pt[i], pt[i + 1]))
            elif code == MPath.CLOSEPOLY:
                if len(cur) > 2:
                    polys.append(cur)
                cur = []
        if len(cur) > 2:
            polys.append(cur)
        # Normalise back into a unit box so the renderer's fitting maths is
        # independent of the size used for flattening.
        xs = [x for p in polys for x, _ in p]
        ys = [y for p in polys for _, y in p]
        span = max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
        out[str(d)] = [[[round(x / span, 5), round(y / span, 5)] for x, y in p]
                       for p in polys]
        print(f"  {d}: {len(out[str(d)])} contour(s), "
              f"{sum(len(p) for p in out[str(d)])} points")

    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    bake()
