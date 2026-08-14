"""
Rolling Octahedron: an eight-faced die rolling on a triangular grid.

The cube variant in `generator.py` is the MIRA task; this is a harder sibling
that keeps the same idea — track a solid's hidden faces through a sequence of
rolls — on a lattice where each cell has three neighbours instead of four.

Why an octahedron, and why a triangular grid:

  A solid can only roll consistently on a grid whose cells its resting face
  tiles. That rules out most Platonic solids: a dodecahedron's pentagons do not
  tile the plane at all. Triangular faces do, which admits the tetrahedron, the
  octahedron, and the icosahedron.

  Of those, the tetrahedron is unusable here. Its rotation group locks to the
  lattice's cell parity, leaving exactly one orientation reachable per cell, so
  a model could read the answer off the position alone without tracking the
  solid — precisely the shortcut this task exists to rule out. The octahedron
  keeps four distinct bottom faces per cell, so the solid genuinely has to be
  carried mentally.

The kinematics here are not hand-derived. `_build_orientations()` rolls a real
octahedron in 3D — rotating about the physical edge it tips over — and records
the resulting orientation graph once at import. `test_kinematics.py` asserts the
group it produces has order 24, that every roll is reversible, and that position
does not determine the bottom face.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# matplotlib is imported lazily, inside the drawing functions that need it.
# The geometry and kinematics above them do not, and Blender's bundled Python
# has no matplotlib -- a module-level import would make this file unimportable
# there, taking the orientation graph with it. Keeping it lazy lets
# blender_render.py reuse the real kinematics instead of restating them.

# ============================================================================
# Geometry
# ============================================================================

# A regular octahedron has 6 vertices and 8 triangular faces.
_VERTS = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    dtype=float,
)
FACES: List[Tuple[int, int, int]] = [
    (0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
    (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5),
]

# Face values, chosen so opposite faces sum to 9 — the eight-faced analogue of a
# standard die's opposite-faces-sum-to-7 rule.
FACE_VALUES: Dict[int, int] = {0: 1, 6: 8, 1: 2, 7: 7, 2: 3, 4: 6, 3: 4, 5: 5}
OPPOSITE: Dict[int, int] = {0: 6, 6: 0, 1: 7, 7: 1, 2: 4, 4: 2, 3: 5, 5: 3}

# Dihedral angle: how far the solid turns as it tips over one edge.
_DIHEDRAL = math.pi - math.acos(-1 / 3)


def _centroid(verts: np.ndarray, face: Tuple[int, int, int]) -> np.ndarray:
    return verts[list(face)].mean(axis=0)


def _bottom_face(verts: np.ndarray) -> int:
    """Index of the face currently resting on the board."""
    return min(range(8), key=lambda i: _centroid(verts, FACES[i])[2])


def _settle(verts: np.ndarray) -> np.ndarray:
    """Rotate the solid so its lowest face lies flat on z = 0."""
    bi = _bottom_face(verts)
    f = FACES[bi]
    n = np.cross(verts[f[1]] - verts[f[0]], verts[f[2]] - verts[f[0]])
    n = n / np.linalg.norm(n)
    if n[2] > 0:
        n = -n
    axis = np.cross(n, np.array([0.0, 0.0, -1.0]))
    s = np.linalg.norm(axis)
    if s > 1e-9:
        axis = axis / s
        ang = math.acos(float(np.clip(np.dot(n, [0.0, 0.0, -1.0]), -1, 1)))
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)
        verts = (R @ verts.T).T
    verts = verts.copy()
    verts[:, 2] -= _centroid(verts, FACES[_bottom_face(verts)])[2]
    return verts


def _rotate_about_edge(verts: np.ndarray, p: np.ndarray, q: np.ndarray,
                       ang: float) -> np.ndarray:
    axis = q - p
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)
    return (R @ (verts - p).T).T + p


REFERENCE = _settle(_VERTS)


def _snap(x: float) -> float:
    """Round a lattice step, normalising -0.0 to 0.0 so it keys consistently."""
    return round(float(x), 3) + 0.0


def _legal_rolls(verts: np.ndarray) -> List[Tuple[Tuple[float, float], np.ndarray]]:
    """
    Every roll available from this pose, as (lattice step, new vertices).

    A roll tips the solid about one edge of its resting face. Candidates that
    would push it through the board, or leave it balanced rather than resting
    flat, are rejected.
    """
    bi = _bottom_face(verts)
    f = FACES[bi]
    origin = _centroid(verts, f)[:2]
    out = []
    for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
        for sign in (+1, -1):
            cand = _rotate_about_edge(verts, verts[a], verts[b], sign * _DIHEDRAL)
            if cand[:, 2].min() < -1e-6:
                continue
            nb = _bottom_face(cand)
            if nb == bi or abs(_centroid(cand, FACES[nb])[2]) > 1e-6:
                continue
            step = _centroid(cand, FACES[nb])[:2] - origin
            if np.linalg.norm(step) > 1e-6:
                out.append(((_snap(step[0]), _snap(step[1])), cand))
    return out


def _rotation_of(verts: np.ndarray) -> Tuple[float, ...]:
    """Signature identifying an orientation, via Kabsch against the reference."""
    A = REFERENCE - REFERENCE.mean(0)
    B = verts - verts.mean(0)
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return tuple(np.round(R.flatten(), 3) + 0.0)


def _build_orientations() -> Tuple[List[np.ndarray], Dict[int, Dict[Tuple[float, float], int]]]:
    """Enumerate every reachable orientation and the roll graph between them."""
    index: Dict[Tuple[float, ...], int] = {}
    poses: List[np.ndarray] = []

    def register(v: np.ndarray) -> int:
        sig = _rotation_of(v)
        if sig not in index:
            index[sig] = len(poses)
            poses.append(v)
        return index[sig]

    register(REFERENCE)
    i = 0
    while i < len(poses):
        for _, nxt in _legal_rolls(poses[i]):
            register(nxt)
        i += 1

    graph: Dict[int, Dict[Tuple[float, float], int]] = {}
    for k, pose in enumerate(poses):
        graph[k] = {step: index[_rotation_of(nxt)]
                    for step, nxt in _legal_rolls(pose)}
    return poses, graph


POSES, ROLL_GRAPH = _build_orientations()
NUM_ORIENTATIONS = len(POSES)



# Plain on-screen names for the six roll directions, following the cube's
# convention: describe what a viewer of the picture sees, never compass words.
# The angles are the projected screen directions of each lattice step, so these
# names stay true to the rendered image rather than to the internal lattice.
STEP_NAMES: Dict[Tuple[float, float], str] = {
    (0.789, -0.211): "right",
    (0.577, 0.577): "up",
    (-0.211, 0.789): "up-left",
    (-0.789, 0.211): "left",
    (-0.577, -0.577): "down",
    (0.211, -0.789): "down-right",
}


def step_name(step) -> str:
    """The on-screen name of a lattice step, as shown to a model."""
    key = (_snap(step[0]), _snap(step[1]))
    return STEP_NAMES.get(key, f"step {key}")


def face_layout(orientation: int) -> Dict[str, Any]:
    """
    The solid's faces described the way a viewer would describe them.

    Numbering the faces `face_1..face_7` would be useless to a reasoning model:
    the labels carry no spatial meaning, so nothing can be deduced from them.
    Instead the state is split into what the picture actually shows -- the face
    on the board, the face directly opposite it, the four turned toward the
    camera, and the rest hidden behind.
    """
    verts = POSES[orientation]
    resting = _bottom_face(verts)
    visible, hidden = [], []
    for fi in range(8):
        if fi in (resting, OPPOSITE[resting]):
            continue
        n, _ = _outward(verts, fi)
        (visible if float(n @ CAMERA) > 0.02 else hidden).append(FACE_VALUES[fi])
    return {
        "bottom": FACE_VALUES[resting],
        "opposite_the_bottom": FACE_VALUES[OPPOSITE[resting]],
        "visible_sides": sorted(visible),
        "hidden_sides": sorted(hidden),
    }


def bottom_value(orientation: int) -> int:
    """The face value resting on the board in this orientation."""
    return FACE_VALUES[_bottom_face(POSES[orientation])]


def visible_values(orientation: int, camera: np.ndarray) -> List[int]:
    """
    Face values a viewer at `camera` can read.

    Mirrors exactly what `render_state` labels, so the two cannot drift apart.
    """
    verts = POSES[orientation]
    centre = verts.mean(axis=0)
    out = []
    for fi in range(8):
        f = FACES[fi]
        c = _centroid(verts, f)
        n = np.cross(verts[f[1]] - verts[f[0]], verts[f[2]] - verts[f[0]])
        n = n / np.linalg.norm(n)
        if n @ (c - centre) < 0:
            n = -n
        if float(n @ camera) > 0.02:
            out.append(FACE_VALUES[fi])
    return out


# ============================================================================
# Board and puzzle generation
# ============================================================================


def build_board(radius: Tuple[float, float] = (4.6, 4.0)) -> Dict[Tuple[float, float], Dict[str, Any]]:
    """
    Walk the lattice outward from the origin, collecting reachable cells.

    Cells are keyed by their centre. Each records the orientation a solid would
    have arriving there, which fixes the triangle's up/down pointing — the two
    alternate, so the polygon cannot be derived from the centre alone.
    """
    cells: Dict[Tuple[float, float], Dict[str, Any]] = {}
    start_key = (0.0, 0.0)
    cells[start_key] = {"orientation": 0, "centre": (0.0, 0.0)}
    stack = [(0, (0.0, 0.0))]
    guard = 0
    while stack and guard < 8000:
        orient, pos = stack.pop()
        guard += 1
        if abs(pos[0]) > radius[0] or abs(pos[1]) > radius[1]:
            continue
        for step, nxt in ROLL_GRAPH[orient].items():
            q = (round(pos[0] + step[0], 2), round(pos[1] + step[1], 2))
            if q not in cells:
                cells[q] = {"orientation": nxt, "centre": q}
                stack.append((nxt, q))
    return cells


def cell_polygon(cell: Dict[str, Any]) -> np.ndarray:
    """The triangle a cell occupies, in world coordinates."""
    verts = POSES[cell["orientation"]]
    f = FACES[_bottom_face(verts)]
    pts = np.array([verts[i][:2] for i in f])
    pts = pts - pts.mean(0)
    return pts + np.array(cell["centre"])



def _silhouette_on_board(cell: Tuple[float, float], orientation: int,
                         board: Dict[Tuple[float, float], Dict[str, Any]],
                         scale: float = 0.92) -> bool:
    """
    True when the solid's base sits on the board.

    Only the base is tested. In this side-on view the body projects about four
    times a cell's screen height, so a real die standing on any cell rises well
    above the board behind it -- that is what a die looks like from the side,
    not an error. Requiring the whole silhouette to overlap a cell would force
    the path into the middle of a board large enough to swallow it.
    """
    def in_triangle(pt: np.ndarray, tri: np.ndarray) -> bool:
        a, b, c = tri
        v0, v1, v2 = c - a, b - a, pt - a
        d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
        d20, d21 = v2 @ v0, v2 @ v1
        den = d00 * d11 - d01 * d01
        if abs(den) < 1e-12:
            return False
        u = (d11 * d20 - d01 * d21) / den
        v = (d00 * d21 - d01 * d20) / den
        return u >= -1e-9 and v >= -1e-9 and u + v <= 1 + 1e-9

    verts = POSES[orientation]
    origin = _centroid(verts, FACES[_bottom_face(verts)])[:2]
    P = verts.copy()
    P[:, 0] -= origin[0]
    P[:, 1] -= origin[1]
    base = FACES[_bottom_face(P)]
    corners = [np.array(iso(cell[0] + P[k][0] * scale,
                            cell[1] + P[k][1] * scale)) for k in base]
    tris = [np.array([np.array(iso(x, y)) for x, y in cell_polygon(c)])
            for c in board.values()]
    return all(any(in_triangle(pt, t) for t in tris) for pt in corners)


def generate_instance(seed: int = 0, num_rolls: int = 5,
                      board_radius: Tuple[float, float] = (4.6, 4.0)) -> Dict[str, Any]:
    """
    Build one octahedron puzzle.

    The path avoids revisiting cells where it can, matching the cube generator:
    a path that doubles back draws arrows on top of each other and makes the
    question image unreadable.
    """
    rng = random.Random(seed)
    board = build_board(board_radius)

    # Keep the path clear of the rim. In this side-on view the solid stands
    # tall and projects nearly a full cell width past its own footprint, so a
    # ring-of-neighbours test is not enough: the board is diamond-shaped, and a
    # cell can have all its neighbours yet still sit where the silhouette hangs
    # over the edge. Test the projected silhouette against the board directly.
    interior = {k for k in board if _silhouette_on_board(k, board[k]["orientation"], board)}

    orient = 0
    pos = (0.0, 0.0)
    trace = [{"cell": list(pos), "orientation": orient, "bottom": bottom_value(orient)}]
    path: List[List[float]] = []
    visited = {(0.0, 0.0)}

    for _ in range(num_rolls):
        options = [(s, o) for s, o in ROLL_GRAPH[orient].items()
                   if (round(pos[0] + s[0], 2), round(pos[1] + s[1], 2)) in interior]
        fresh = [(s, o) for s, o in options
                 if (round(pos[0] + s[0], 2), round(pos[1] + s[1], 2)) not in visited]
        if fresh:
            options = fresh
        if not options:
            break
        step, orient = options[rng.randrange(len(options))]
        pos = (round(pos[0] + step[0], 2), round(pos[1] + step[1], 2))
        visited.add(pos)
        path.append([step[0], step[1]])
        trace.append({"cell": list(pos), "orientation": orient,
                      "bottom": bottom_value(orient)})

    steps = []
    for i, step in enumerate(path, 1):
        before, after = trace[i - 1], trace[i]
        steps.append({
            "step": i,
            "path": None,
            "lattice_step": step,
            "direction_name": step_name(step),
            "from_cell": before["cell"],
            "to_cell": after["cell"],
            "faces_before": face_layout(before["orientation"]),
            "faces_after": face_layout(after["orientation"]),
            "bottom_face": after["bottom"],
        })

    return {
        "solid": "octahedron",
        "faces": 8,
        "grid": "triangular",
        "seed": seed,
        "num_rolls": len(path),
        "steps": steps,
        "face_values": {str(k): v for k, v in FACE_VALUES.items()},
        "start": list(trace[0]["cell"]),
        "path": path,
        "trace": trace,
        "answer": trace[-1]["bottom"],
        "board_radius": list(board_radius),
    }


def replay(instance: Dict[str, Any]) -> List[int]:
    """Re-derive every bottom value from the stored path, for verification."""
    orient = 0
    out = [bottom_value(orient)]
    for step in instance["path"]:
        key = (_snap(step[0]), _snap(step[1]))
        if key not in ROLL_GRAPH[orient]:
            raise ValueError(f"stored step {key} is not a legal roll")
        orient = ROLL_GRAPH[orient][key]
        out.append(bottom_value(orient))
    return out


# ============================================================================
# Rendering
# ============================================================================

# Line weights, matching the cube renderer so the two variants read as one
# dataset. The ordering carries meaning: interior rules sit below the die's own
# edges, which sit below the markers, which sit below the board's border and
# the path. Previously these were five unrelated values picked ad hoc.
LW_GRID = 0.9        # interior lattice rules
LW_DIE_EDGE = 1.2    # edges of the solid
LW_MARKER = 2.0      # dashed destination outline
LW_BORDER = 3.2      # outer board border
LW_PATH = 3.4        # the route itself


# A deliberately side-on view: the camera sits about 17 degrees above the board
# rather than looking down on it. Two measurements fix this choice. Looking down
# steeply shrinks the faces angled away from the camera until their numbers no
# longer fit; looking along the board flattens the triangular lattice into an
# unreadable band. At this angle all four camera-facing faces keep a projected
# area of at least 0.45, and a lattice cell still projects 0.30 as tall as it is
# wide.
#
# Four is the ceiling, not a shortfall: an octahedron is convex, so exactly half
# its faces point away from any viewpoint. No angle shows more.
_ISO_X = (0.866, 0.30)
_ISO_Y = (-0.866, 0.30)
_ISO_Z = (0.0, 1.40)


def iso(x: float, y: float, z: float = 0.0) -> Tuple[float, float]:
    """Project a world point onto the isometric screen plane."""
    return (x * _ISO_X[0] + y * _ISO_Y[0] + z * _ISO_Z[0],
            x * _ISO_X[1] + y * _ISO_Y[1] + z * _ISO_Z[1])


def _camera() -> np.ndarray:
    """The view direction, as the null space of the projection."""
    M = np.array([[_ISO_X[0], _ISO_Y[0], _ISO_Z[0]],
                  [_ISO_X[1], _ISO_Y[1], _ISO_Z[1]]])
    cam = np.linalg.svd(M)[2][-1]
    return cam if cam[2] >= 0 else -cam


CAMERA = _camera()


def _outward(verts: np.ndarray, fi: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Outward normal of a face, measured from the solid's own centroid.

    Testing the normal against the face centroid alone would assume the solid
    is centred on the origin; it sits above the board instead, which flips the
    sign for downward faces and makes the resting face appear to point up.
    """
    f = FACES[fi]
    c = _centroid(verts, f)
    n = np.cross(verts[f[1]] - verts[f[0]], verts[f[2]] - verts[f[0]])
    n = n / np.linalg.norm(n)
    return (n if n @ (c - verts.mean(axis=0)) > 0 else -n), c




def _board_near_path(trace, radius, margin: int = 2):
    """
    The cells worth drawing: those near the path, but never so few that the
    solid ends up hanging over an edge of the cropped board.

    Generation works on a lattice far larger than any path needs, so the solid
    can never sit where its body would overhang. Drawing all of it would shrink
    the route to a scribble in an empty field, so the picture is cropped -- but
    cropping re-creates the overhang problem it was solving, because the drawn
    board now has new edges. The crop is therefore widened until every frame's
    silhouette is covered again.
    """
    full = build_board(radius)

    def crop(rings: int):
        """
        Grow outward from the path by whole lattice rings.

        Cropping against a screen-space rectangle looked tidier but cut across
        the lattice, leaving notches whose edges the border tracer then drew as
        heavy black lines through the middle of the board. Growing by adjacency
        keeps the kept set gap-free by construction.
        """
        keep = {}
        frontier = set()
        for entry in trace:
            key = (round(entry["cell"][0], 2), round(entry["cell"][1], 2))
            if key in full:
                keep[key] = full[key]
                frontier.add(key)
        for _ in range(rings):
            nxt = set()
            for key in frontier:
                for step in ROLL_GRAPH[full[key]["orientation"]]:
                    n = (round(key[0] + step[0], 2), round(key[1] + step[1], 2))
                    if n in full and n not in keep:
                        keep[n] = full[n]
                        nxt.add(n)
            frontier = nxt
        return keep

    rings = int(margin)
    while rings <= int(margin) + 4:
        keep = crop(rings)
        if all(_silhouette_on_board((round(t["cell"][0], 2), round(t["cell"][1], 2)),
                                    t["orientation"], keep) for t in trace):
            return keep
        rings += 1
    return full



def _draw_board_border(ax, board) -> None:
    """
    Stroke the board's outline in heavy black.

    Traces the convex hull of the projected cells rather than every unshared
    edge. Tracing unshared edges is exact, but the kept set is a lattice
    region, so its true boundary zigzags in and out; over half those edges then
    land well inside the drawn area and read as stray black lines across the
    board. The hull gives the straight outline a board is expected to have.
    """
    from matplotlib.patches import Polygon as MplPolygon

    from generator import BOARD_EDGE

    points = np.vstack([
        np.array([iso(px, py) for px, py in cell_polygon(cell)])
        for cell in board.values()
    ])

    # Monotone chain: sort, then sweep the lower and upper hulls.
    order = sorted(map(tuple, np.round(points, 6)))
    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2:
                (x1, y1), (x2, y2) = out[-2], out[-1]
                if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) > 0:
                    break
                out.pop()
            out.append(p)
        return out
    hull = half(order)[:-1] + half(order[::-1])[:-1]

    ax.add_patch(MplPolygon(hull, closed=True, facecolor="none",
                            edgecolor=BOARD_EDGE, linewidth=LW_BORDER,
                            zorder=4, joinstyle="round"))


def _draw_route(ax, points, color, dashed: bool, zorder: float = 7) -> None:
    """
    Draw a run of the path as a line capped with one arrowhead.

    Matplotlib's annotate-style arrows are sized in points, so on a board this
    wide they shrank to specks. Drawing in data units instead keeps the route
    the same visual weight whatever the figure size.
    """
    from matplotlib.patches import FancyArrow

    if len(points) < 2:
        return

    (x0, y0), (x1, y1) = points[-2], points[-1]
    dx, dy = x1 - x0, y1 - y0
    norm = math.hypot(dx, dy) or 1.0
    ux, uy = dx / norm, dy / norm

    # Size the head against the final segment. A fixed head swallowed short
    # segments whole -- the lattice has two step lengths, and on the shorter
    # one a 0.30 head left almost no line behind it.
    head = min(0.26, norm * 0.45)

    trimmed = list(points[:-1]) + [(x1 - ux * head, y1 - uy * head)]
    ax.plot([p[0] for p in trimmed], [p[1] for p in trimmed],
            color=color, linewidth=LW_PATH, zorder=zorder,
            linestyle=(0, (4, 2.5)) if dashed else "-",
            alpha=0.85 if dashed else 1.0,
            solid_capstyle="butt", solid_joinstyle="round")
    ax.add_patch(FancyArrow(
        x1 - ux * head, y1 - uy * head, ux * head, uy * head,
        width=0.0, head_width=head, head_length=head,
        length_includes_head=True, color=color,
        alpha=0.85 if dashed else 1.0, zorder=zorder + 1))


def _mark_cell(ax, board, cell, color) -> None:
    """Outline a cell, used to show the path's destination."""
    from matplotlib.patches import Polygon

    key = (round(cell[0], 2), round(cell[1], 2))
    if key not in board:
        return
    poly = cell_polygon(board[key])
    ax.add_patch(Polygon([iso(px, py) for px, py in poly], closed=True,
                         facecolor="none", edgecolor=color, linewidth=LW_MARKER,
                         linestyle=(0, (3, 2)), zorder=5))


def render_state(instance: Dict[str, Any], step: int = 0, dpi: int = 150):
    """
    Draw the board, the path, and the solid at `step`.

    step=0 is the question image: the whole path is shown with the solid at its
    start. Later steps show the solid partway along, with the path drawn up to
    that point.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    from generator import (BOARD_EDGE, BOARD_SURFACE, GRID_LINE, DIE_BODY,
                           DIE_EDGE, PATH_BLACK, PIP_COLOR, REMAINING_PATH,
                           _shade)

    trace = instance["trace"]
    # Draw only the board around the path. Generation uses a large lattice so
    # the solid never overhangs an edge, but rendering all of it leaves the
    # route covering a fraction of the frame and unreadable.
    board = _board_near_path(trace, tuple(instance["board_radius"]))
    at = trace[step]["cell"]
    orientation = trace[step]["orientation"]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_aspect("equal")
    ax.axis("off")

    for cell in board.values():
        poly = cell_polygon(cell)
        ax.add_patch(Polygon([iso(px, py) for px, py in poly], closed=True,
                             facecolor=BOARD_SURFACE, edgecolor=GRID_LINE,
                             linewidth=LW_GRID, alpha=0.55, zorder=1))

    # Outer border: the union of the cells has a ragged edge, so trace only the
    # boundary segments -- those belonging to exactly one cell. Drawing each
    # cell with a heavy stroke would blacken the interior rules too.
    _draw_board_border(ax, board)

    # Path: travelled portion solid, the rest dashed, so each frame shows both
    # progress and where the die is still going. Drawing only the travelled
    # part would erase the route the question is asking about.
    screen = [iso(t["cell"][0], t["cell"][1]) for t in trace]
    # Same scheme as the cube: the route already travelled is solid black, the
    # part still to come is dashed orange. The die projects wider than a single
    # lattice step, so a path drawn beneath it disappears exactly where it
    # matters; both runs go over the body.
    _draw_route(ax, screen[:step + 1], PATH_BLACK, dashed=False, zorder=60)
    if step < len(screen) - 1:
        _draw_route(ax, screen[step:], REMAINING_PATH, dashed=True, zorder=60)

    # Mark the START cell, as the cube does. Marking the destination instead
    # gave away half the answer: the route already ends there, and outlining it
    # told the reader where to look rather than making them follow the path.
    _mark_cell(ax, board, trace[0]["cell"], REMAINING_PATH)

    # The stored pose carries its own world position, so re-centre it before
    # placing it — otherwise the position counts twice and the solid drifts off
    # the board as the path lengthens.
    pose = POSES[orientation]
    origin = _centroid(pose, FACES[_bottom_face(pose)])[:2]
    P = pose.copy()
    P[:, 0] -= origin[0]
    P[:, 1] -= origin[1]

    # scale 1.0 makes the resting face exactly fill a cell; inset it slightly so
    # the solid sits inside its cell, as the cube does.
    scale = 0.92
    order = sorted(range(8), key=lambda fi: float(_outward(P, fi)[1] @ CAMERA))
    drawn = 0
    for fi in order:
        n, c = _outward(P, fi)
        toward = float(n @ CAMERA)
        # Lower-half faces are drawn too: without them the body has a gap on
        # the near side, since the silhouette is made of both halves.
        if toward <= 0.02:
            continue
        f = FACES[fi]
        poly = [iso(at[0] + P[k][0] * scale, at[1] + P[k][1] * scale,
                    P[k][2] * scale) for k in f]
        ax.add_patch(Polygon(poly, closed=True,
                             facecolor=_shade(DIE_BODY, 0.62 + 0.38 * toward),
                             edgecolor=DIE_EDGE, linewidth=LW_DIE_EDGE,
                             zorder=12 + drawn))
        # Every face turned toward the camera carries its number. An earlier
        # cut also required the face to point upward, which silently blanked
        # the lower-half faces in all 24 orientations.
        s = iso(at[0] + c[0] * scale, at[1] + c[1] * scale, c[2] * scale)
        ax.text(s[0], s[1], str(FACE_VALUES[fi]), ha="center", va="center",
                fontsize=13, fontweight="bold", color=PIP_COLOR,
                zorder=40 + drawn)
        drawn += 1

    ax.autoscale_view()
    return fig
