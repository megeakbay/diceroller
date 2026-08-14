"""
Rolling Dice task generator (MIRA-style, MentisOculi conventions).

A standard right-handed die is tipped over its edges across a grid board.
Three question variants are supported:

  top  -- which number is on the top face after the whole path?
  sum  -- total of the bottom-face values touching the board at each step
  two  -- two paths of equal length; which one yields the higher bottom sum?

Die state is carried as an explicit six-face dict so every intermediate step is
inspectable and renderable. Opposite faces always sum to 7.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Polygon

# Official color palette (datasets/README.md)
CERULEAN = "#0090C1"      # Primary brand
RASPBERRY = "#E85D75"     # Highlight / path A
AQUAMARINE = "#0F6B62"    # Movable secondary / start marker
# Darkened from #2EC4B6 when the board went white: the original sat at
# 2.2:1 against the surface, too faint for a dashed outline to read.
SUNSHINE = "#FFD670"      # Helper / reference
MIDNIGHT = "#1B263B"      # Outlines / text / static obstacles

# Isometric board styling: a white board ruled in black, with a heavier outer
# border, and a pale die standing on it.
#
# A white surface inverts what needs care. On the old teal board every element
# was darker than the board and read automatically; on white the danger is the
# opposite — anything pale disappears. Two colours had to move with the surface:
#
#   DIE_BODY  was #F2EFE2, which sits at 1.15:1 against white and vanished as a
#             fill. It is now a warmer grey (1.4:1) so the die separates from
#             the board by tone, not only by its outline.
#   REMAINING_PATH was #F5A742 at 2.0:1. Darkened to 4.2:1, which keeps it
#             clearly orange while staying distinct from the black path.
#
# The interior rules are mid-grey rather than black: at full black a 7x7 grid
# reads as a mesh that competes with the die. The outer border is true black and
# twice the weight, so the board's extent stays unambiguous.
BOARD_SURFACE = "#FFFFFF"
GRID_LINE = "#5A5A5A"
BOARD_EDGE = "#000000"
DIE_BODY = "#DED8C6"
DIE_EDGE = "#1A1A1A"
# Pips and face numbers are black: on a pale die body the old teal read as a
# decorative tint, and the values are the one thing the task asks the model to
# read off the image.
PIP_COLOR = "#000000"

BLOCKED = MIDNIGHT
PATH_BLACK = "#14181C"
PATH_RED = RASPBERRY
# The not-yet-travelled path, drawn over the dark board and the pale die, so it
# needs to stay legible against both.
REMAINING_PATH = "#C2620A"

# Grid directions. y grows upward, so N is +y.
DIRECTIONS: Dict[str, Tuple[int, int]] = {
    "N": (0, 1),
    "S": (0, -1),
    "E": (1, 0),
    "W": (-1, 0),
}
# Plain on-screen names for the four roll directions. Compass words stay as the
# internal codes, but anything shown to a model uses these: in the isometric
# view "north" is not up on screen, so compass wording misleads. +y recedes to
# the upper left, +x to the upper right.
DIRECTION_NAMES = {
    "N": "up-left",
    "S": "down-right",
    "E": "up-right",
    "W": "down-left",
}

# Die-face labels shown to a model. Top and bottom are what the questions ask
# about, so they keep their everyday names; each side face is named by the
# direction the die would roll to tip that face down.
FACE_NAMES = {
    "top": "top",
    "bottom": "bottom",
    "north": "up-left side",
    "south": "down-right side",
    "east": "up-right side",
    "west": "down-left side",
}


# ============================================================================
# Die state and kinematics
# ============================================================================


@dataclass
class DieState:
    """Six visible faces of a die. Opposite faces sum to 7."""

    top: int
    bottom: int
    north: int
    south: int
    east: int
    west: int

    def validate(self) -> None:
        pairs = [
            (self.top, self.bottom),
            (self.north, self.south),
            (self.east, self.west),
        ]
        for a, b in pairs:
            if a + b != 7:
                raise ValueError(f"Opposite faces must sum to 7, got {a} and {b}")
        if sorted(
            [self.top, self.bottom, self.north, self.south, self.east, self.west]
        ) != [1, 2, 3, 4, 5, 6]:
            raise ValueError("Die must show each of 1-6 exactly once")

    def as_dict(self) -> Dict[str, int]:
        return asdict(self)


def canonical_die() -> DieState:
    """The standard starting orientation used across the dataset."""
    return DieState(top=1, bottom=6, north=2, south=5, east=3, west=4)


def roll(die: DieState, direction: str) -> DieState:
    """
    Tip the die one cell in `direction`, rotating about the leading bottom edge.

    Rolling east: the west face becomes the new top, the top becomes the new
    east, the east becomes the new bottom, and the bottom becomes the new west.
    The axis perpendicular to travel (north/south here) is unchanged.
    """
    if direction == "E":
        return DieState(
            top=die.west,
            bottom=die.east,
            east=die.top,
            west=die.bottom,
            north=die.north,
            south=die.south,
        )
    if direction == "W":
        return DieState(
            top=die.east,
            bottom=die.west,
            east=die.bottom,
            west=die.top,
            north=die.north,
            south=die.south,
        )
    if direction == "N":
        return DieState(
            top=die.south,
            bottom=die.north,
            north=die.top,
            south=die.bottom,
            east=die.east,
            west=die.west,
        )
    if direction == "S":
        return DieState(
            top=die.north,
            bottom=die.south,
            north=die.bottom,
            south=die.top,
            east=die.east,
            west=die.west,
        )
    raise ValueError(f"Unknown direction: {direction}")


def apply_action(state: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply a single roll to a state dict and return the new state.

    Action: {"direction": "N"|"S"|"E"|"W"}
    State:  {"die": {...}, "position": [x, y], "board": {...}}

    Raises ValueError if the roll would leave the board or enter a blocked cell,
    so that a tool-use caller can observe the collision rather than silently
    producing an invalid state.
    """
    import copy

    new_state = copy.deepcopy(state)
    direction = action["direction"]
    dx, dy = DIRECTIONS[direction]

    x, y = new_state["position"]
    nx, ny = x + dx, y + dy

    board = new_state["board"]
    if not (0 <= nx < board["width"] and 0 <= ny < board["height"]):
        raise ValueError(f"Roll {direction} leaves the board at ({nx}, {ny})")
    if [nx, ny] in [list(c) for c in board.get("blocked", [])]:
        raise ValueError(f"Roll {direction} enters a blocked cell at ({nx}, {ny})")

    die = DieState(**new_state["die"])
    new_state["die"] = roll(die, direction).as_dict()
    new_state["position"] = [nx, ny]
    return new_state


def is_legal(state: Dict[str, Any], direction: str) -> bool:
    """Whether a roll in `direction` is legal from the current state."""
    try:
        apply_action(state, {"direction": direction})
        return True
    except ValueError:
        return False


def simulate(
    board: Dict[str, Any],
    start: Tuple[int, int],
    die: DieState,
    path: List[str],
) -> List[Dict[str, Any]]:
    """
    Roll along `path` and return the full trace.

    The trace has len(path) + 1 entries; entry 0 is the initial state before
    any roll. Each later entry records the direction taken to reach it and the
    bottom face that touches the board there.
    """
    state = {
        "board": board,
        "position": [start[0], start[1]],
        "die": die.as_dict(),
    }
    trace = [
        {
            "step": 0,
            "direction": None,
            "position": list(state["position"]),
            "die": dict(state["die"]),
            "bottom": state["die"]["bottom"],
        }
    ]
    for i, direction in enumerate(path):
        state = apply_action(state, {"direction": direction})
        trace.append(
            {
                "step": i + 1,
                "direction": direction,
                "position": list(state["position"]),
                "die": dict(state["die"]),
                "bottom": state["die"]["bottom"],
            }
        )
    return trace


def bottom_sum(trace: List[Dict[str, Any]]) -> int:
    """
    Sum of the bottom faces touching the board at each step of the path.

    The initial resting cell is excluded; only cells reached by a roll count,
    matching the MIRA "sum of the numbers on the bottom face that touches the
    path at each step" phrasing.
    """
    return sum(entry["bottom"] for entry in trace[1:])


# ============================================================================
# Board and path generation
# ============================================================================


def _random_path(
    rng: random.Random,
    board: Dict[str, Any],
    start: Tuple[int, int],
    length: int,
    max_tries: int = 2000,
) -> Optional[List[str]]:
    """
    Sample a legal, non-self-intersecting path of exactly `length` rolls.

    Avoiding revisits keeps the rendered path readable and keeps the question
    unambiguous when the path is drawn as a polyline.
    """
    for _ in range(max_tries):
        state = {
            "board": board,
            "position": [start[0], start[1]],
            "die": canonical_die().as_dict(),
        }
        visited = {tuple(state["position"])}
        path: List[str] = []
        stuck = False

        for _ in range(length):
            options = []
            for direction in DIRECTIONS:
                if not is_legal(state, direction):
                    continue
                dx, dy = DIRECTIONS[direction]
                nxt = (state["position"][0] + dx, state["position"][1] + dy)
                if nxt in visited:
                    continue
                options.append(direction)
            if not options:
                stuck = True
                break
            # Prefer continuing straight now and then so paths are not pure zigzag
            if path and path[-1] in options and rng.random() < 0.4:
                direction = path[-1]
            else:
                direction = rng.choice(options)
            state = apply_action(state, {"direction": direction})
            visited.add(tuple(state["position"]))
            path.append(direction)

        if not stuck and len(path) == length:
            return path
    return None


def generate_instance(
    seed: int = 0,
    variant: str = "top",
    num_rolls: int = 3,
    board_width: int = 5,
    board_height: int = 5,
    num_blocked: int = 0,
) -> Dict[str, Any]:
    """
    Build one puzzle instance.

    Args:
        seed: RNG seed.
        variant: "top", "sum", or "two".
        num_rolls: Path length (the reasoning-step count / difficulty level).
        board_width, board_height: Grid size in cells.
        num_blocked: Number of impassable cells to scatter on the board.

    Returns:
        Instance dict with board, start, die, path(s), trace(s), and answer.
    """
    if variant not in ("top", "sum", "two"):
        raise ValueError(f"Unknown variant: {variant}")

    rng = random.Random(seed)

    # Board large enough that a path of num_rolls fits comfortably
    span = max(3, min(7, num_rolls + 2))
    board_width = max(board_width, span)
    board_height = max(board_height, span)

    blocked: List[List[int]] = []
    board = {"width": board_width, "height": board_height, "blocked": blocked}

    start = (rng.randrange(board_width), rng.randrange(board_height))

    for _ in range(num_blocked):
        for _ in range(200):
            cell = [rng.randrange(board_width), rng.randrange(board_height)]
            if cell != [start[0], start[1]] and cell not in blocked:
                blocked.append(cell)
                break

    die = canonical_die()
    die.validate()

    if variant == "two":
        # Two equal-length paths from the same start; they must differ and, so
        # the question has a determinate answer, must not tie on bottom sum.
        path_a = _random_path(rng, board, start, num_rolls)
        if path_a is None:
            raise RuntimeError("Failed to sample path A")
        trace_a = simulate(board, start, die, path_a)
        sum_a = bottom_sum(trace_a)

        path_b = None
        trace_b = None
        sum_b = None
        for _ in range(400):
            candidate = _random_path(rng, board, start, num_rolls)
            if candidate is None or candidate == path_a:
                continue
            cand_trace = simulate(board, start, die, candidate)
            cand_sum = bottom_sum(cand_trace)
            if cand_sum != sum_a:
                path_b, trace_b, sum_b = candidate, cand_trace, cand_sum
                break
        if path_b is None:
            raise RuntimeError("Failed to sample a distinguishable path B")

        winner = "black" if sum_a > sum_b else "red"
        answer = {"winner": winner, "total": max(sum_a, sum_b)}

        return {
            "variant": variant,
            "board": board,
            "start": [start[0], start[1]],
            "initial_die": die.as_dict(),
            "paths": {"black": path_a, "red": path_b},
            "traces": {"black": trace_a, "red": trace_b},
            "sums": {"black": sum_a, "red": sum_b},
            "num_rolls": num_rolls,
            "answer": answer,
        }

    path = _random_path(rng, board, start, num_rolls)
    if path is None:
        raise RuntimeError("Failed to sample a path")
    trace = simulate(board, start, die, path)

    if variant == "top":
        answer = trace[-1]["die"]["top"]
    else:  # sum
        answer = bottom_sum(trace)

    return {
        "variant": variant,
        "board": board,
        "start": [start[0], start[1]],
        "initial_die": die.as_dict(),
        "path": path,
        "trace": trace,
        "num_rolls": num_rolls,
        "answer": answer,
    }


def generate_instance_json(**kwargs: Any) -> str:
    return json.dumps(generate_instance(**kwargs), indent=2)


# ============================================================================
# Rendering
# ============================================================================

# Pip layout in a unit square, keyed by face value.
_PIP_LAYOUT: Dict[int, List[Tuple[float, float]]] = {
    1: [(0.5, 0.5)],
    2: [(0.28, 0.72), (0.72, 0.28)],
    3: [(0.25, 0.75), (0.5, 0.5), (0.75, 0.25)],
    4: [(0.28, 0.28), (0.28, 0.72), (0.72, 0.28), (0.72, 0.72)],
    5: [(0.26, 0.26), (0.26, 0.74), (0.5, 0.5), (0.74, 0.26), (0.74, 0.74)],
    6: [
        (0.28, 0.24),
        (0.28, 0.5),
        (0.28, 0.76),
        (0.72, 0.24),
        (0.72, 0.5),
        (0.72, 0.76),
    ],
}


def _draw_pips(ax, value: int, x0: float, y0: float, size: float, color: str) -> None:
    """Draw the pip pattern for `value` inside the square at (x0, y0)."""
    radius = size * 0.075
    for px, py in _PIP_LAYOUT[value]:
        ax.add_patch(
            plt.Circle(
                (x0 + px * size, y0 + py * size), radius, color=color, zorder=6
            )
        )


# ----------------------------------------------------------------------------
# Isometric projection
#
# The board is drawn in an isometric 3D view, matching the MIRA figures: the
# die is a cube with three faces visible, sitting on a receding grid.
# ----------------------------------------------------------------------------

# Screen-space basis vectors for the three world axes.
#
# The two ground axes recede away from the viewer in opposite screen-x
# directions and both drop in screen-y; north (+y) recedes up-left, east (+x)
# recedes up-right, so the near corner of the board is the origin and the board
# opens away from the camera. The vertical axis rises.
#
# For the die to read as a cube rather than a flat slab, its vertical edges
# must project to roughly the same screen length as its horizontal ones -- that
# ratio is _ISO_Z[1] / hypot(*_ISO_X), and it is what governs how cubic the die
# looks. The constraint is that the cube's near-top corner (0,0,1) and its
# far-bottom corner (1,1,0) must not project onto the same point: they coincide
# exactly when the +z rise equals the two ground axes' combined y-drop, which
# collapses the solid. Keeping the ground drop shallow (0.36 each, so 0.72
# combined) leaves room for a 0.90 rise -- near-equal edge lengths, with the
# closest pair of projected corners still well separated. test_kinematics.py
# asserts both properties so a future tweak cannot silently reintroduce the
# collapse.
_ISO_X = (0.866, 0.36)   # world +x (east)  -> right and away
_ISO_Y = (-0.866, 0.36)  # world +y (north) -> left and away
_ISO_Z = (0.0, 0.90)     # world +z (up)    -> up


def iso(x: float, y: float, z: float = 0.0) -> Tuple[float, float]:
    """Project a world point onto the isometric screen plane."""
    return (
        x * _ISO_X[0] + y * _ISO_Y[0] + z * _ISO_Z[0],
        x * _ISO_X[1] + y * _ISO_Y[1] + z * _ISO_Z[1],
    )


def _shade(hex_color: str, factor: float) -> str:
    """Lighten (factor > 1) or darken (factor < 1) a hex colour."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    r, g, b = (min(255, max(0, int(c * factor))) for c in (r, g, b))
    return f"#{r:02X}{g:02X}{b:02X}"


def _iso_cell(ax, cx: int, cy: int, facecolor: str, edgecolor: str,
              linewidth: float, zorder: float) -> None:
    """Fill one board cell as an isometric parallelogram."""
    corners = [
        iso(cx, cy), iso(cx + 1, cy), iso(cx + 1, cy + 1), iso(cx, cy + 1)
    ]
    ax.add_patch(
        Polygon(corners, closed=True, facecolor=facecolor, edgecolor=edgecolor,
                linewidth=linewidth, zorder=zorder)
    )


def _draw_board(ax, board: Dict[str, Any]) -> None:
    """Draw the receding isometric grid, dark like the MIRA input images."""
    bw, bh = board["width"], board["height"]

    # Board surface
    surface = [iso(0, 0), iso(bw, 0), iso(bw, bh), iso(0, bh)]
    ax.add_patch(
        Polygon(surface, closed=True, facecolor=BOARD_SURFACE,
                edgecolor="none", zorder=0)
    )

    # Grid lines, drawn on top of the surface
    for x in range(bw + 1):
        p0, p1 = iso(x, 0), iso(x, bh)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=GRID_LINE,
                linewidth=0.9, alpha=0.55, zorder=1)
    for y in range(bh + 1):
        p0, p1 = iso(0, y), iso(bw, y)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=GRID_LINE,
                linewidth=0.9, alpha=0.55, zorder=1)

    for cx, cy in board.get("blocked", []):
        _iso_cell(ax, cx, cy, BLOCKED, _shade(BLOCKED, 1.4), 1.0, 2)

    # Outer edge: true black and twice the weight of the interior rules, so the
    # board's extent stays unambiguous now that its surface matches the page.
    ax.add_patch(
        Polygon(surface, closed=True, facecolor="none", edgecolor=BOARD_EDGE,
                linewidth=3.2, zorder=3)
    )

    # Frame the projected extent. A die standing on any cell rises one unit in
    # z, so include the whole board at that height too — otherwise a die on a
    # far corner is clipped.
    raised = [iso(x, y, 1.0) for x, y in
              ((0, 0), (bw, 0), (bw, bh), (0, bh))]
    xs = [p[0] for p in surface + raised]
    ys = [p[1] for p in surface + raised]
    pad = 0.5
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def _path_points(start: Tuple[int, int], path: List[str]) -> List[Tuple[float, float]]:
    """
    Cell-centre path points, projected to screen space.

    Drawn a hair above the surface (z = 0.02) so the line reads as lying on the
    board rather than z-fighting with the grid.
    """
    cells = [(start[0] + 0.5, start[1] + 0.5)]
    x, y = start
    for direction in path:
        dx, dy = DIRECTIONS[direction]
        x, y = x + dx, y + dy
        cells.append((x + 0.5, y + 0.5))
    return [iso(cx, cy, 0.02) for cx, cy in cells]


def _draw_path(
    ax,
    start: Tuple[int, int],
    path: List[str],
    color: str,
    upto: Optional[int] = None,
    linestyle: str = "-",
    zorder: float = 4,
    clear_start_cell: bool = False,
    clear_end_cell: bool = False,
) -> None:
    """
    Draw the path polyline, capped with a single arrowhead at its end.

    `zorder` lets the caller lift a path above the die. `clear_start_cell` and
    `clear_end_cell` pull the line back to the boundary of the die's cell
    instead of its centre — the die occupies that cell, so a line running to
    the centre disappears under the cube.
    """
    pts = _path_points(start, path)
    if upto is not None:
        pts = pts[: upto + 1]
    if len(pts) < 2:
        return

    # Pull the endpoint on the die's cell back to the boundary it rolls across,
    # so the line meets the cube's base rather than vanishing under its centre.
    # The step between adjacent cell centres is one cell, so the segment
    # midpoint lies exactly on that boundary; every roll stays depicted.
    if clear_start_cell:
        (sx0, sy0), (sx1, sy1) = pts[0], pts[1]
        pts = [((sx0 + sx1) / 2.0, (sy0 + sy1) / 2.0)] + list(pts[1:])
    if clear_end_cell:
        (ex0, ey0), (ex1, ey1) = pts[-2], pts[-1]
        pts = list(pts[:-1]) + [((ex0 + ex1) / 2.0, (ey0 + ey1) / 2.0)]


    # Stop the line short of the tip so the arrowhead caps it cleanly rather
    # than the stroke running through and past the head.
    head_len = 0.26
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    dx, dy = x1 - x0, y1 - y0
    norm = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux, uy = dx / norm, dy / norm

    trimmed = list(pts[:-1]) + [(x1 - ux * head_len, y1 - uy * head_len)]
    ax.plot(
        [p[0] for p in trimmed], [p[1] for p in trimmed],
        color=color, linewidth=3.4, linestyle=linestyle, alpha=0.95,
        zorder=zorder, solid_capstyle="butt", solid_joinstyle="round",
    )

    # A single arrowhead at the end of the run, as in the MIRA figures
    ax.add_patch(
        FancyArrow(
            x1 - ux * head_len, y1 - uy * head_len, ux * head_len, uy * head_len,
            width=0.0, head_width=0.26, head_length=head_len,
            length_includes_head=True, color=color, alpha=0.95,
            zorder=zorder + 1,
        )
    )


def _face_pips_3d(
    ax,
    origin: Tuple[float, float, float],
    u: Tuple[float, float, float],
    v: Tuple[float, float, float],
    value: int,
    color: str,
    scale: float,
    zorder: float,
) -> None:
    """
    Draw a face's pips onto an arbitrary 3D plane.

    `origin` is the face's lower-left corner in world space; `u` and `v` span
    the face. Pip positions come from the same 2D layout used elsewhere, so a
    face reads identically however it is oriented.
    """
    for px, py in _PIP_LAYOUT[value]:
        wx = origin[0] + u[0] * px + v[0] * py
        wy = origin[1] + u[1] * px + v[1] * py
        wz = origin[2] + u[2] * px + v[2] * py
        sx, sy = iso(wx, wy, wz)
        ax.add_patch(
            plt.Circle((sx, sy), 0.075 * scale, color=color, zorder=zorder)
        )


def _draw_die_on_cell(ax, pos: List[int], die: Dict[str, int]) -> None:
    """
    Draw the die as an isometric cube with three faces visible.

    In this projection the camera faces the board's near corner, so the visible
    faces are the top, the south face, and the west face. Showing three faces
    rather than one gives enough of the cube to reason about without exposing
    the full state.
    """
    x, y = pos
    inset = 0.04
    s = 1 - 2 * inset          # cube edge length
    x0, y0, z0 = x + inset, y + inset, 0.0

    # Corners of the cube in world space
    def c(dx: float, dy: float, dz: float) -> Tuple[float, float]:
        return iso(x0 + dx * s, y0 + dy * s, z0 + dz * s)

    top_face = [c(0, 0, 1), c(1, 0, 1), c(1, 1, 1), c(0, 1, 1)]
    south_face = [c(0, 0, 0), c(1, 0, 0), c(1, 0, 1), c(0, 0, 1)]
    west_face = [c(0, 0, 0), c(0, 1, 0), c(0, 1, 1), c(0, 0, 1)]

    # Lambert-ish shading: top brightest, south mid, west darkest
    for corners, shade in (
        (south_face, 0.86),
        (west_face, 0.72),
        (top_face, 1.0),
    ):
        ax.add_patch(
            Polygon(
                corners, closed=True,
                facecolor=_shade(DIE_BODY, shade),
                edgecolor=DIE_EDGE, linewidth=1.2,
                zorder=10, joinstyle="round",
            )
        )

    # Pips, each mapped onto its own face plane
    _face_pips_3d(
        ax, (x0, y0, z0 + s), (s, 0, 0), (0, s, 0),
        die["top"], PIP_COLOR, s, 11,
    )
    _face_pips_3d(
        ax, (x0, y0, z0), (s, 0, 0), (0, 0, s),
        die["south"], PIP_COLOR, s, 11,
    )
    _face_pips_3d(
        ax, (x0, y0, z0), (0, s, 0), (0, 0, s),
        die["west"], PIP_COLOR, s, 11,
    )


def render_state(
    instance: Dict[str, Any],
    step: int = 0,
    show_path: bool = True,
    figsize: Tuple[int, int] = (5, 5),
) -> plt.Figure:
    """
    Render the board at `step` rolls into the path.

    Step 0 is the initial state. The travelled portion of the path is drawn
    solid and the remainder dashed, so each image shows both progress and the
    plan still to execute.
    """
    board = instance["board"]
    fig, ax = plt.subplots(figsize=figsize)
    _draw_board(ax, board)

    start = (instance["start"][0], instance["start"][1])

    trace = (
        instance["traces"]["black"]
        if instance["variant"] == "two"
        else instance["trace"]
    )

    if instance["variant"] == "two":
        if show_path:
            for key, color in (("black", PATH_BLACK), ("red", PATH_RED)):
                _draw_path(ax, start, instance["paths"][key], color)
    else:
        if show_path:
            full = instance["path"]
            _draw_path(
                ax, start, full, PATH_BLACK, upto=step,
                clear_end_cell=step > 0,
            )
            if step < len(full):
                # The remaining path resumes from the cell reached so far. Take
                # that cell from the trace — deriving it from projected screen
                # coordinates is not possible, since the projection is not
                # invertible without the z term.
                rx, ry = trace[step]["position"]
                # Drawn below the die (which sits at zorder 10): the line now
                # stops at the cube's base, so letting the die occlude what
                # little overlaps reads as the die standing on the path.
                _draw_path(
                    ax, (rx, ry), full[step:], REMAINING_PATH, linestyle="--",
                    zorder=6, clear_start_cell=True,
                )

    entry = trace[min(step, len(trace) - 1)]

    # Start marker sits under the die so the origin stays visible. Drawn in the
    # path colour rather than its own teal, so it reads as one end of the route
    # instead of separate furniture -- matching the octahedron variant.
    sx, sy = start
    marker = [
        iso(sx + 0.05, sy + 0.05), iso(sx + 0.95, sy + 0.05),
        iso(sx + 0.95, sy + 0.95), iso(sx + 0.05, sy + 0.95),
    ]
    ax.add_patch(
        Polygon(
            marker, closed=True, facecolor="none", edgecolor=REMAINING_PATH,
            linewidth=2.0, linestyle=(0, (3, 2)), zorder=3,
        )
    )

    _draw_die_on_cell(ax, entry["position"], entry["die"])

    fig.tight_layout()
    return fig


def render_instance(instance: Dict[str, Any], **kwargs: Any) -> plt.Figure:
    """Render the initial state (the question image)."""
    return render_state(instance, step=0, **kwargs)


def plot_from_metadata(
    metadata_path: str,
    step: int = 0,
    figsize: Tuple[int, int] = (5, 5),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Load instance metadata from a JSON file and render it."""
    with open(metadata_path, "r") as f:
        instance = json.load(f)

    fig = render_state(instance, step=step, figsize=figsize)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig
