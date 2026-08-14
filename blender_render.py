"""
Blender renderer for the Rolling Dice dataset.

A drop-in alternative to the matplotlib renderers in `generator.py` and
`octahedron.py`: it reads the same `metadata.json` and writes the same
`initial.png` / `cot_NN.png` filenames into the same puzzle directories, so
every downstream consumer (the rollout scripts, the judges,
`evaluate_responses.py`) keeps working against Blender output unchanged.

Run it through Blender, never through a bare python:

    blender --background --python blender_render.py -- --puzzle output/top/level_05/puzzle_0001
    blender --background --python blender_render.py -- --output-dir output --variant top

`render_dataset.py` wraps that invocation if you would rather not type it.

Why the visuals are constrained rather than free
------------------------------------------------
These images are benchmark stimuli, not illustration. Three properties are
load-bearing, and the port preserves each one deliberately:

  * **Exactly three cube faces are visible.** A fourth would hand the model
    state it is supposed to carry mentally, which is the capability the task
    probes. The matplotlib version got this by drawing only three polygons; here
    an orthographic camera on the cube's near corner gives it by construction,
    so the invariant holds for free rather than by omission.
  * **Face values stay readable.** Pips are real geometry (shallow spheres set
    into the faces) and the octahedron's numbers are extruded text, both in
    black against a pale body, since reading them off the picture is half the
    task.
  * **Travelled path solid, remainder dashed.** Each frame has to show both
    progress and the plan still to run, or the model loses the route.

The camera angles are not re-invented. They are the exact null spaces of the
two matplotlib projection bases -- 29.5 degrees of elevation for the cube, 16.86
for the octahedron -- recovered by `_camera_from_basis`, so a Blender frame
looks at the board from the same place the 2D renderer did. `--check-angles`
prints them against the source bases.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

try:
    import bpy
    import bmesh
    from mathutils import Euler, Matrix, Vector
except ImportError:  # pragma: no cover - allows --check-angles outside Blender
    bpy = None


# ============================================================================
# Palette
#
# Taken from generator.py so the two renderers agree. Hex is converted to
# linear-light because Blender's shader sockets are linear, while the source
# values are sRGB -- assigning them raw would wash every surface out.
# ============================================================================

BOARD_SURFACE = "#FFFFFF"
GRID_LINE = "#5A5A5A"
BOARD_EDGE = "#000000"
DIE_BODY = "#DED8C6"
DIE_EDGE = "#1A1A1A"
PIP_COLOR = "#000000"
PATH_BLACK = "#14181C"
PATH_RED = "#E85D75"
REMAINING_PATH = "#C2620A"
BLOCKED = "#1B263B"


def srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def rgba(hex_color: str, alpha: float = 1.0) -> Tuple[float, float, float, float]:
    """Hex sRGB -> linear RGBA, as Blender's shader inputs expect."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return (srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), alpha)


# ============================================================================
# Camera
# ============================================================================

# The screen-space bases from the two matplotlib renderers, kept here so the
# derived camera can be checked against its source rather than trusted.
CUBE_BASIS = ((0.866, 0.36), (-0.866, 0.36), (0.0, 0.90))
OCT_BASIS = ((0.866, 0.30), (-0.866, 0.30), (0.0, 1.40))


def _camera_from_basis(basis) -> Tuple[float, float]:
    """
    Elevation and azimuth (degrees) of the viewpoint a 2D iso basis implies.

    A parallel projection collapses exactly the direction its 2x3 matrix sends
    to zero, so the view direction is that matrix's null space -- no fitting or
    guesswork. Signs are chosen so the camera sits above the board.
    """
    (xx, xy), (yx, yy), (zx, zy) = basis
    # Null space of [[xx, yx, zx], [xy, yy, zy]] via the cross product of rows.
    r0 = (xx, yx, zx)
    r1 = (xy, yy, zy)
    d = (
        r0[1] * r1[2] - r0[2] * r1[1],
        r0[2] * r1[0] - r0[0] * r1[2],
        r0[0] * r1[1] - r0[1] * r1[0],
    )
    n = math.sqrt(sum(c * c for c in d)) or 1.0
    d = tuple(c / n for c in d)
    if d[2] < 0:
        d = tuple(-c for c in d)
    elevation = math.degrees(math.asin(d[2]))
    azimuth = math.degrees(math.atan2(d[1], d[0]))
    return elevation, azimuth


CUBE_ELEVATION, CUBE_AZIMUTH = _camera_from_basis(CUBE_BASIS)
OCT_ELEVATION, OCT_AZIMUTH = _camera_from_basis(OCT_BASIS)


def setup_camera(target: Sequence[float], elevation: float, azimuth: float,
                 ortho_scale: float) -> Any:
    """
    Place an orthographic camera looking at `target` from the given angles.

    Orthographic, not perspective: the source renders are parallel projections,
    and perspective would make a die's apparent size depend on where on the
    board it stands -- the same face reading differently cell to cell.
    """
    el, az = math.radians(elevation), math.radians(azimuth)
    distance = 40.0
    loc = (
        target[0] + distance * math.cos(el) * math.cos(az),
        target[1] + distance * math.cos(el) * math.sin(az),
        target[2] + distance * math.sin(el),
    )

    cam_data = bpy.data.cameras.new("Camera")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = ortho_scale
    cam = bpy.data.objects.new("Camera", cam_data)
    cam.location = loc
    bpy.context.collection.objects.link(cam)

    direction = Vector(target) - Vector(loc)
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    return cam


# ============================================================================
# Scene plumbing
# ============================================================================


def reset_scene() -> None:
    """Empty the file. Renders run in a loop, so leftovers would accumulate."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def make_material(name: str, color, roughness: float = 0.55,
                  shadeless: bool = False) -> Any:
    """
    A Principled BSDF material.

    `shadeless` swaps in a pure emission shader, used for the path and the grid:
    those are diagram marks rather than objects, and letting light fall on them
    would make a route dim as it crossed the board's shaded side.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")

    if shadeless:
        shader = nodes.new("ShaderNodeEmission")
        shader.inputs["Color"].default_value = color
        shader.inputs["Strength"].default_value = 1.0
    else:
        shader = nodes.new("ShaderNodeBsdfPrincipled")
        shader.inputs["Base Color"].default_value = color
        shader.inputs["Roughness"].default_value = roughness
        # Metallic/specular left at defaults; a dataset die wants a matte read,
        # not highlights that could be mistaken for pips.
        if "Specular IOR Level" in shader.inputs:
            shader.inputs["Specular IOR Level"].default_value = 0.25
        elif "Specular" in shader.inputs:
            shader.inputs["Specular"].default_value = 0.25

    links.new(shader.outputs[0], out.inputs["Surface"])
    return mat


def setup_world(strength: float = 1.0) -> None:
    """A plain white world, so the board's white surface stays white."""
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs["Strength"].default_value = strength


def setup_lighting(target: Sequence[float], azimuth: float) -> None:
    """
    A key/fill pair placed relative to the camera.

    The key sits off the camera's azimuth so the die's three visible faces pick
    up distinct brightnesses -- that tonal separation is what makes the solid
    read as a cube rather than a flat hexagon, and the 2D renderer faked it with
    hardcoded shade factors.
    """
    key_data = bpy.data.lights.new("Key", type="SUN")
    key_data.energy = 3.0
    key_data.angle = math.radians(12.0)
    key = bpy.data.objects.new("Key", key_data)
    key.location = (target[0] + 12, target[1] + 6, target[2] + 18)
    key.rotation_euler = (
        Vector(target) - Vector(key.location)
    ).to_track_quat("-Z", "Y").to_euler()
    bpy.context.collection.objects.link(key)

    fill_data = bpy.data.lights.new("Fill", type="SUN")
    fill_data.energy = 1.4
    fill = bpy.data.objects.new("Fill", fill_data)
    az = math.radians(azimuth)
    fill.location = (
        target[0] + 16 * math.cos(az),
        target[1] + 16 * math.sin(az),
        target[2] + 8,
    )
    fill.rotation_euler = (
        Vector(target) - Vector(fill.location)
    ).to_track_quat("-Z", "Y").to_euler()
    bpy.context.collection.objects.link(fill)


def setup_render(width: int, height: int, samples: int, engine: str,
                 transparent: bool = False) -> None:
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = transparent

    if engine == "cycles":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        # Prefer GPU when the build exposes one; falls back silently to CPU.
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.get_devices()
            for dev in prefs.devices:
                dev.use = True
            scene.cycles.device = "GPU"
        except Exception:
            scene.cycles.device = "CPU"
    else:
        # EEVEE's identifier changed in Blender 4.2 (EEVEE Next).
        for ident in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
            try:
                scene.render.engine = ident
                break
            except TypeError:
                continue
        eevee = getattr(scene, "eevee", None)
        if eevee is not None and hasattr(eevee, "taa_render_samples"):
            eevee.taa_render_samples = max(16, samples // 4)


# ============================================================================
# Mesh helpers
# ============================================================================


def new_mesh_object(name: str, verts, faces, material) -> Any:
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.data.materials.append(material)
    bpy.context.collection.objects.link(obj)
    return obj


def add_tube(name: str, p0, p1, radius: float, material, z: float) -> Any:
    """
    A capsule-ended cylinder between two ground points, used for path segments
    and grid rules.

    Drawn as geometry rather than as a 2D stroke because a Blender scene has no
    line primitive that survives an orthographic render at arbitrary
    resolutions; a real tube also keeps its width consistent with the die.
    """
    a = Vector((p0[0], p0[1], z))
    b = Vector((p1[0], p1[1], z))
    d = b - a
    length = d.length
    if length < 1e-9:
        return None

    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=length,
                                        location=tuple((a + b) / 2.0), vertices=12)
    obj = bpy.context.active_object
    obj.name = name
    obj.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
    obj.data.materials.clear()
    obj.data.materials.append(material)
    return obj


def add_cone(name: str, tip, direction, length: float, radius: float,
             material, z: float) -> Any:
    """An arrowhead: a cone whose tip sits at `tip`, pointing along `direction`."""
    d = Vector((direction[0], direction[1], 0.0))
    if d.length < 1e-9:
        return None
    d.normalize()
    base = Vector((tip[0], tip[1], z)) - d * length
    centre = base + d * (length / 2.0)

    bpy.ops.mesh.primitive_cone_add(radius1=radius, radius2=0.0, depth=length,
                                    location=tuple(centre), vertices=16)
    obj = bpy.context.active_object
    obj.name = name
    obj.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
    obj.data.materials.clear()
    obj.data.materials.append(material)
    return obj


def dash_segments(points: Sequence[Sequence[float]], dash: float, gap: float):
    """
    Chop a polyline into dashes of world-space length.

    Walking the whole polyline rather than dashing each segment separately keeps
    the rhythm continuous across corners; per-segment dashing restarts the
    pattern at every turn and reads as a change of meaning.
    """
    out: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    carry = 0.0
    drawing = True
    for i in range(len(points) - 1):
        ax, ay = points[i][0], points[i][1]
        bx, by = points[i + 1][0], points[i + 1][1]
        seg = math.hypot(bx - ax, by - ay)
        t = 0.0
        while t < seg:
            want = (dash if drawing else gap) - carry
            step = min(want, seg - t)
            if drawing and step > 1e-6:
                t0, t1 = t / seg, (t + step) / seg
                out.append((
                    (ax + (bx - ax) * t0, ay + (by - ay) * t0),
                    (ax + (bx - ax) * t1, ay + (by - ay) * t1),
                ))
            t += step
            carry += step
            if carry >= (dash if drawing else gap) - 1e-9:
                drawing = not drawing
                carry = 0.0
    return out


def draw_route(points: Sequence[Sequence[float]], color: str, dashed: bool,
               z: float, width: float = 0.075, name: str = "route") -> None:
    """
    Draw a route as tubes plus a single arrowhead at its end.

    The line is pulled back from the tip so the head caps it instead of poking
    through, matching the 2D renderer's `head_len` trim.
    """
    if len(points) < 2:
        return
    mat = make_material(f"{name}_mat", rgba(color), shadeless=True)

    head_len = 0.30
    head_radius = 0.16
    x0, y0 = points[-2][0], points[-2][1]
    x1, y1 = points[-1][0], points[-1][1]
    dx, dy = x1 - x0, y1 - y0
    norm = math.hypot(dx, dy) or 1.0
    ux, uy = dx / norm, dy / norm

    # Shorten the last segment by the head length, so the shaft stops where the
    # cone begins.
    trimmed = [tuple(p[:2]) for p in points[:-1]]
    trimmed.append((x1 - ux * head_len, y1 - uy * head_len))

    if dashed:
        for i, (a, b) in enumerate(dash_segments(trimmed, dash=0.34, gap=0.22)):
            add_tube(f"{name}_dash_{i}", a, b, width, mat, z)
    else:
        for i in range(len(trimmed) - 1):
            add_tube(f"{name}_seg_{i}", trimmed[i], trimmed[i + 1], width, mat, z)
            # A sphere at each interior joint, so corners do not show a notch
            # where two cylinders meet at an angle.
            if 0 < i + 1 < len(trimmed) - 1:
                bpy.ops.mesh.primitive_uv_sphere_add(
                    radius=width, location=(trimmed[i + 1][0], trimmed[i + 1][1], z),
                    segments=12, ring_count=8)
                joint = bpy.context.active_object
                joint.data.materials.clear()
                joint.data.materials.append(mat)

    add_cone(f"{name}_head", (x1, y1), (ux, uy), head_len, head_radius, mat, z)


# ============================================================================
# Cube variant
# ============================================================================

# Pip layout in a unit square, matching generator._PIP_LAYOUT exactly so a face
# reads the same in both renderers.
PIP_LAYOUT: Dict[int, List[Tuple[float, float]]] = {
    1: [(0.5, 0.5)],
    2: [(0.28, 0.72), (0.72, 0.28)],
    3: [(0.25, 0.75), (0.5, 0.5), (0.75, 0.25)],
    4: [(0.28, 0.28), (0.28, 0.72), (0.72, 0.28), (0.72, 0.72)],
    5: [(0.26, 0.26), (0.26, 0.74), (0.5, 0.5), (0.74, 0.26), (0.74, 0.74)],
    6: [(0.28, 0.24), (0.28, 0.5), (0.28, 0.76),
        (0.72, 0.24), (0.72, 0.5), (0.72, 0.76)],
}

DIRECTIONS = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}


def build_cube_board(board: Dict[str, Any]) -> None:
    """The board: a white slab, grid rules as thin tubes, blocked cells inset."""
    bw, bh = board["width"], board["height"]
    thickness = 0.12

    surface_mat = make_material("board", rgba(BOARD_SURFACE), roughness=0.9)
    verts = [
        (0, 0, 0), (bw, 0, 0), (bw, bh, 0), (0, bh, 0),
        (0, 0, -thickness), (bw, 0, -thickness),
        (bw, bh, -thickness), (0, bh, -thickness),
    ]
    faces = [(0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1),
             (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    new_mesh_object("board", verts, faces, surface_mat)

    grid_mat = make_material("grid", rgba(GRID_LINE), shadeless=True)
    for x in range(bw + 1):
        add_tube(f"grid_x{x}", (x, 0), (x, bh), 0.012, grid_mat, 0.004)
    for y in range(bh + 1):
        add_tube(f"grid_y{y}", (0, y), (bw, y), 0.012, grid_mat, 0.004)

    # Outer border: heavier, and true black, so the board's extent is
    # unambiguous now that the surface matches the page.
    edge_mat = make_material("board_edge", rgba(BOARD_EDGE), shadeless=True)
    corners = [(0, 0), (bw, 0), (bw, bh), (0, bh)]
    for i in range(4):
        add_tube(f"border_{i}", corners[i], corners[(i + 1) % 4],
                 0.035, edge_mat, 0.006)

    blocked_mat = make_material("blocked", rgba(BLOCKED), roughness=0.7)
    for cx, cy in board.get("blocked", []):
        new_mesh_object(
            f"blocked_{cx}_{cy}",
            [(cx + 0.04, cy + 0.04, 0.01), (cx + 0.96, cy + 0.04, 0.01),
             (cx + 0.96, cy + 0.96, 0.01), (cx + 0.04, cy + 0.96, 0.01)],
            [(0, 1, 2, 3)],
            blocked_mat,
        )


def build_die(position: Sequence[int], die: Dict[str, int]) -> Any:
    """
    A real cube with pips on all six faces.

    Pips are spheres pressed slightly into each face rather than painted
    circles, so they catch the lighting and stay legible at the angles the
    faces are seen from. Only three faces can be seen at once -- that is the
    camera's doing, not a choice made here, which is exactly why the invariant
    is safe: no drawing decision can accidentally expose a fourth.
    """
    inset = 0.04
    s = 1.0 - 2 * inset
    cx = position[0] + 0.5
    cy = position[1] + 0.5
    cz = s / 2.0

    bpy.ops.mesh.primitive_cube_add(size=s, location=(cx, cy, cz))
    cube = bpy.context.active_object
    cube.name = "die"

    # Bevel the edges. A razor-sharp cube renders with aliased silhouettes and
    # reads as a flat hexagon at this scale; a small bevel catches a highlight
    # along each edge and separates the three visible faces.
    bevel = cube.modifiers.new("Bevel", "BEVEL")
    bevel.width = 0.035
    bevel.segments = 3
    bevel.limit_method = "ANGLE"

    body_mat = make_material("die_body", rgba(DIE_BODY), roughness=0.6)
    cube.data.materials.append(body_mat)

    pip_mat = make_material("pip", rgba(PIP_COLOR), roughness=0.45)
    half = s / 2.0
    pip_radius = 0.075 * s

    # Face definitions: (value, origin corner, u axis, v axis, outward normal),
    # with the local frame of each face expressed in the cube's own space. The
    # pip layout is applied in that frame, so a face reads identically however
    # it is oriented -- the same trick the 2D renderer uses in _face_pips_3d.
    faces = [
        (die["top"],    (-half, -half,  half), (s, 0, 0), (0, s, 0), (0, 0, 1)),
        (die["bottom"], (-half,  half, -half), (s, 0, 0), (0, -s, 0), (0, 0, -1)),
        (die["north"],  (-half,  half,  half), (s, 0, 0), (0, 0, -s), (0, 1, 0)),
        (die["south"],  (-half, -half, -half), (s, 0, 0), (0, 0, s), (0, -1, 0)),
        (die["east"],   ( half, -half, -half), (0, s, 0), (0, 0, s), (1, 0, 0)),
        (die["west"],   (-half,  half, -half), (0, -s, 0), (0, 0, s), (-1, 0, 0)),
    ]

    for value, origin, u, v, normal in faces:
        for i, (px, py) in enumerate(PIP_LAYOUT[value]):
            lx = origin[0] + u[0] * px + v[0] * py
            ly = origin[1] + u[1] * px + v[1] * py
            lz = origin[2] + u[2] * px + v[2] * py
            # Sink the sphere so only a shallow cap stands proud of the face.
            depth = pip_radius * 0.55
            loc = (cx + lx - normal[0] * depth,
                   cy + ly - normal[1] * depth,
                   cz + lz - normal[2] * depth)
            bpy.ops.mesh.primitive_uv_sphere_add(radius=pip_radius, location=loc,
                                                 segments=16, ring_count=10)
            pip = bpy.context.active_object
            pip.name = f"pip_{value}_{i}"
            pip.data.materials.clear()
            pip.data.materials.append(pip_mat)
            bpy.ops.object.shade_smooth()

    return cube


def cube_path_points(start: Sequence[int], path: Sequence[str]):
    """Cell-centre points along a path, in world coordinates."""
    pts = [(start[0] + 0.5, start[1] + 0.5)]
    x, y = start[0], start[1]
    for d in path:
        dx, dy = DIRECTIONS[d]
        x, y = x + dx, y + dy
        pts.append((x + 0.5, y + 0.5))
    return pts


def render_cube_state(meta: Dict[str, Any], step: int, out_path: Path,
                      args: argparse.Namespace, path_key: str = None) -> None:
    """Build and render one cube frame."""
    reset_scene()
    setup_world()

    board = meta["board"]
    bw, bh = board["width"], board["height"]

    if meta["variant"] == "two" and path_key is not None:
        path = meta["paths"][path_key]
        trace = meta["traces"][path_key]
    elif meta["variant"] == "two":
        path = meta["paths"]["black"]
        trace = meta["traces"]["black"]
    else:
        path = meta["path"]
        trace = meta["trace"]

    build_cube_board(board)

    entry = trace[min(step, len(trace) - 1)]
    start = meta["start"]

    if meta["variant"] == "two" and path_key is None:
        # Question image for the two-path variant: both routes, both solid, in
        # their own colours -- the question names them by colour.
        for key, color in (("black", PATH_BLACK), ("red", PATH_RED)):
            pts = cube_path_points(start, meta["paths"][key])
            draw_route(pts, color, dashed=False, z=0.02, name=f"route_{key}")
    else:
        full_pts = cube_path_points(start, path)
        if step > 0:
            # Stop the travelled run at the die's cell boundary rather than its
            # centre, so the line meets the cube's base instead of vanishing
            # under it. The midpoint of the last step lies on that boundary.
            done = list(full_pts[:step + 1])
            ax, ay = done[-2]
            bx, by = done[-1]
            done[-1] = ((ax + bx) / 2.0, (ay + by) / 2.0)
            draw_route(done, PATH_BLACK, dashed=False, z=0.02, name="route_done")
        if step < len(path):
            remaining = list(full_pts[step:])
            if len(remaining) >= 2:
                ax, ay = remaining[0]
                bx, by = remaining[1]
                remaining[0] = ((ax + bx) / 2.0, (ay + by) / 2.0)
                draw_route(remaining, REMAINING_PATH, dashed=True, z=0.02,
                           name="route_todo")

    # Start marker, in the remaining-path colour so it reads as one end of the
    # route rather than separate furniture.
    marker_mat = make_material("start_marker", rgba(REMAINING_PATH), shadeless=True)
    sx, sy = start[0], start[1]
    mc = [(sx + 0.06, sy + 0.06), (sx + 0.94, sy + 0.06),
          (sx + 0.94, sy + 0.94), (sx + 0.06, sy + 0.94)]
    for i in range(4):
        for j, (a, b) in enumerate(dash_segments([mc[i], mc[(i + 1) % 4]],
                                                 dash=0.16, gap=0.11)):
            add_tube(f"marker_{i}_{j}", a, b, 0.028, marker_mat, 0.012)

    build_die(entry["position"], entry["die"])

    target = (bw / 2.0, bh / 2.0, 0.0)
    # Fit the board's diagonal, plus room for a die standing at a far corner.
    ortho = math.hypot(bw, bh) * 0.80 + 1.2
    setup_camera(target, CUBE_ELEVATION, CUBE_AZIMUTH, ortho)
    setup_lighting(target, CUBE_AZIMUTH)
    setup_render(args.width, args.height, args.samples, args.engine,
                 args.transparent)

    bpy.context.scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)


# ============================================================================
# Octahedron variant
#
# Geometry and kinematics are imported from octahedron.py rather than restated.
# That module derives its orientation graph by rolling a real solid at import,
# and `test_kinematics.py` asserts properties of it -- a second copy here could
# drift from the ground truth the dataset was generated against.
# ============================================================================


def load_octahedron_module():
    """Import octahedron.py from the project directory Blender was pointed at."""
    project = Path(__file__).resolve().parent
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    import octahedron
    return octahedron


def build_octahedron_board(board, oct_) -> None:
    """Triangular lattice cells as flat polygons, with a heavy hull border."""
    surface_mat = make_material("oct_board", rgba(BOARD_SURFACE), roughness=0.9)
    grid_mat = make_material("oct_grid", rgba(GRID_LINE), shadeless=True)

    all_pts = []
    for key, cell in board.items():
        poly = oct_.cell_polygon(cell)
        pts = [(float(px), float(py)) for px, py in poly]
        all_pts.extend(pts)
        new_mesh_object(
            f"cell_{key[0]}_{key[1]}",
            [(px, py, 0.0) for px, py in pts],
            [(0, 1, 2)],
            surface_mat,
        )
        for i in range(3):
            add_tube(f"rule_{key[0]}_{key[1]}_{i}", pts[i], pts[(i + 1) % 3],
                     0.012, grid_mat, 0.004)

    # Border: the convex hull of the cells, matching the 2D renderer. The kept
    # region's true boundary zigzags, and tracing it exactly puts heavy black
    # lines through the middle of the board.
    edge_mat = make_material("oct_edge", rgba(BOARD_EDGE), shadeless=True)
    hull = convex_hull(all_pts)
    for i in range(len(hull)):
        add_tube(f"oct_border_{i}", hull[i], hull[(i + 1) % len(hull)],
                 0.035, edge_mat, 0.006)


def convex_hull(points):
    """Monotone-chain hull, as used by octahedron._draw_board_border."""
    pts = sorted(set((round(x, 6), round(y, 6)) for x, y in points))
    if len(pts) < 3:
        return pts

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

    return half(pts)[:-1] + half(pts[::-1])[:-1]


def build_octahedron(cell, orientation: int, oct_) -> None:
    """
    The solid at `cell`, in the pose the kinematics recorded.

    The stored pose carries its own world position, so it is re-centred on the
    origin before being placed -- otherwise the position counts twice and the
    solid drifts off the board as the path lengthens.
    """
    import numpy as np

    pose = oct_.POSES[orientation]
    origin = oct_._centroid(pose, oct_.FACES[oct_._bottom_face(pose)])[:2]
    P = pose.copy()
    P[:, 0] -= origin[0]
    P[:, 1] -= origin[1]

    scale = 0.92
    verts = [(float(cell[0] + P[k][0] * scale),
              float(cell[1] + P[k][1] * scale),
              float(P[k][2] * scale)) for k in range(len(P))]

    body_mat = make_material("oct_body", rgba(DIE_BODY), roughness=0.6)
    obj = new_mesh_object("octahedron", verts, oct_.FACES, body_mat)

    # Edges as their own dark geometry: a bevel on eight triangles meeting at
    # six points pinches badly, and the faces need a hard line between them or
    # adjacent numbers appear to sit on one surface.
    edge_mat = make_material("oct_die_edge", rgba(DIE_EDGE), shadeless=True)
    seen = set()
    for f in oct_.FACES:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            va, vb = Vector(verts[a]), Vector(verts[b])
            d = vb - va
            bpy.ops.mesh.primitive_cylinder_add(
                radius=0.016, depth=d.length, location=tuple((va + vb) / 2.0),
                vertices=8)
            e = bpy.context.active_object
            e.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
            e.data.materials.clear()
            e.data.materials.append(edge_mat)

    # Numbers on the faces turned toward the camera. Every camera-facing face
    # carries its value, including the lower-half ones: an earlier 2D cut also
    # required the face to point upward, which silently blanked those in all 24
    # orientations.
    text_mat = make_material("oct_text", rgba(PIP_COLOR), shadeless=True)
    camera = oct_.CAMERA
    for fi in range(8):
        n, c = oct_._outward(P, fi)
        if float(n @ camera) <= 0.02:
            continue
        add_face_number(
            oct_.FACE_VALUES[fi],
            (float(cell[0] + c[0] * scale),
             float(cell[1] + c[1] * scale),
             float(c[2] * scale)),
            (float(n[0]), float(n[1]), float(n[2])),
            text_mat,
        )


def add_face_number(value: int, centre, normal, material) -> Any:
    """
    Place a face's number, lying in that face's plane and lifted clear of it.

    The glyph is rotated so its own +Z matches the face normal and then rolled
    so its up axis is as close to world up as that plane allows -- otherwise
    numbers on the lower faces render upside down and become unreadable.
    """
    curve = bpy.data.curves.new(f"num_{value}", type="FONT")
    curve.body = str(value)
    curve.align_x = "CENTER"
    curve.align_y = "CENTER"
    curve.size = 0.34
    curve.extrude = 0.012

    obj = bpy.data.objects.new(f"num_{value}", curve)
    obj.data.materials.append(material)
    bpy.context.collection.objects.link(obj)

    n = Vector(normal).normalized()
    world_up = Vector((0.0, 0.0, 1.0))
    if abs(n.dot(world_up)) > 0.999:
        world_up = Vector((0.0, 1.0, 0.0))
    right = world_up.cross(n).normalized()
    up = n.cross(right).normalized()
    obj.matrix_world = Matrix((
        (right.x, up.x, n.x, centre[0] + n.x * 0.02),
        (right.y, up.y, n.y, centre[1] + n.y * 0.02),
        (right.z, up.z, n.z, centre[2] + n.z * 0.02),
        (0.0, 0.0, 0.0, 1.0),
    ))
    return obj


def render_octahedron_state(meta: Dict[str, Any], step: int, out_path: Path,
                            args: argparse.Namespace) -> None:
    """Build and render one octahedron frame."""
    oct_ = load_octahedron_module()

    reset_scene()
    setup_world()

    trace = meta["trace"]
    board = oct_._board_near_path(trace, tuple(meta["board_radius"]))
    at = trace[step]["cell"]
    orientation = trace[step]["orientation"]

    build_octahedron_board(board, oct_)

    # Route: travelled solid, remainder dashed, both lifted above the board.
    pts = [(t["cell"][0], t["cell"][1]) for t in trace]
    if step > 0:
        draw_route(pts[:step + 1], PATH_BLACK, dashed=False, z=0.02,
                   width=0.055, name="oct_done")
    if step < len(pts) - 1:
        draw_route(pts[step:], REMAINING_PATH, dashed=True, z=0.02,
                   width=0.055, name="oct_todo")

    # Mark the START cell. Marking the destination gives away half the answer:
    # the route already ends there.
    marker_mat = make_material("oct_marker", rgba(REMAINING_PATH), shadeless=True)
    key = (round(trace[0]["cell"][0], 2), round(trace[0]["cell"][1], 2))
    if key in board:
        poly = [(float(px), float(py)) for px, py in oct_.cell_polygon(board[key])]
        for i in range(3):
            for j, (a, b) in enumerate(
                    dash_segments([poly[i], poly[(i + 1) % 3]], dash=0.16, gap=0.11)):
                add_tube(f"oct_marker_{i}_{j}", a, b, 0.026, marker_mat, 0.014)

    build_octahedron(at, orientation, oct_)

    xs = [c[0] for c in [cell["centre"] for cell in board.values()]]
    ys = [c[1] for c in [cell["centre"] for cell in board.values()]]
    target = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, 0.35)
    ortho = max(max(xs) - min(xs), 3.0) * 1.30 + 1.5

    setup_camera(target, OCT_ELEVATION, OCT_AZIMUTH, ortho)
    setup_lighting(target, OCT_AZIMUTH)
    setup_render(args.width, args.height, args.samples, args.engine,
                 args.transparent)

    bpy.context.scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)


# ============================================================================
# Driving the render
# ============================================================================


def render_puzzle(puzzle_dir: Path, args: argparse.Namespace) -> int:
    """
    Render every frame of one puzzle.

    Writes the same filenames the matplotlib renderer does, so the puzzle
    directory stays a drop-in input for the rest of the pipeline. `--suffix`
    writes alongside instead of over, for comparing the two renderers.
    """
    meta_path = puzzle_dir / "metadata.json"
    if not meta_path.exists():
        print(f"  skip {puzzle_dir}: no metadata.json")
        return 0

    with open(meta_path) as f:
        meta = json.load(f)

    variant = meta.get("variant", "top")
    suffix = args.suffix or ""

    def out(name: str) -> Path:
        stem, ext = os.path.splitext(name)
        return puzzle_dir / f"{stem}{suffix}{ext}"

    written = 0

    if variant == "octahedron":
        if not args.cot_only:
            render_octahedron_state(meta, 0, out("initial.png"), args)
            written += 1
        if not args.initial_only:
            for i in range(1, len(meta["trace"])):
                render_octahedron_state(meta, i, out(f"cot_{i - 1:02d}.png"), args)
                written += 1
    elif variant == "two":
        if not args.cot_only:
            render_cube_state(meta, 0, out("initial.png"), args, path_key=None)
            written += 1
        if not args.initial_only:
            for tag in ("black", "red"):
                for i in range(1, len(meta["paths"][tag]) + 1):
                    render_cube_state(meta, i, out(f"cot_{tag}_{i - 1:02d}.png"),
                                      args, path_key=tag)
                    written += 1
    else:
        if not args.cot_only:
            render_cube_state(meta, 0, out("initial.png"), args)
            written += 1
        if not args.initial_only:
            for i in range(1, len(meta["path"]) + 1):
                render_cube_state(meta, i, out(f"cot_{i - 1:02d}.png"), args)
                written += 1

    print(f"  {puzzle_dir}: {written} frame(s)")
    return written


def find_puzzles(output_dir: Path, variant: str, level: int) -> List[Path]:
    """Every puzzle directory under output/, filtered by variant and level."""
    puzzles: List[Path] = []
    variants = [variant] if variant and variant != "all" else \
        [d.name for d in sorted(output_dir.iterdir()) if d.is_dir()]
    for v in variants:
        vdir = output_dir / v
        if not vdir.is_dir():
            continue
        for level_dir in sorted(vdir.glob("level_*")):
            if level is not None and level_dir.name != f"level_{level:02d}":
                continue
            puzzles.extend(sorted(p for p in level_dir.glob("puzzle_*") if p.is_dir()))
    return puzzles


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render Rolling Dice puzzles in Blender",
        epilog="Run through Blender: blender --background --python "
               "blender_render.py -- [options]",
    )
    p.add_argument("--puzzle", type=str, default=None,
                   help="Render a single puzzle directory")
    p.add_argument("--output-dir", type=str, default="output",
                   help="Dataset root to walk")
    p.add_argument("--variant", type=str, default="all",
                   help="top, sum, two, octahedron, or all")
    p.add_argument("--level", type=int, default=None, help="Only this level")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after this many puzzles")
    p.add_argument("--initial-only", action="store_true",
                   help="Only the question image")
    p.add_argument("--cot-only", action="store_true",
                   help="Only the per-step frames")
    p.add_argument("--suffix", type=str, default=None,
                   help="Append to filenames, e.g. _blender, to write "
                        "alongside the matplotlib renders instead of over them")
    p.add_argument("--width", type=int, default=1000)
    p.add_argument("--height", type=int, default=1000)
    p.add_argument("--samples", type=int, default=128,
                   help="Cycles samples; EEVEE uses a quarter of this")
    p.add_argument("--engine", choices=("cycles", "eevee"), default="cycles")
    p.add_argument("--transparent", action="store_true",
                   help="Render on a transparent background")
    p.add_argument("--check-angles", action="store_true",
                   help="Print the camera angles derived from the 2D bases "
                        "and exit; works outside Blender")
    return p.parse_args(argv)


def main() -> None:
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    args = parse_args(argv)

    if args.check_angles:
        print("Camera angles derived from the matplotlib projection bases:")
        print(f"  cube:       elevation {CUBE_ELEVATION:7.3f}  "
              f"azimuth {CUBE_AZIMUTH:8.3f}")
        print(f"  octahedron: elevation {OCT_ELEVATION:7.3f}  "
              f"azimuth {OCT_AZIMUTH:8.3f}")
        print("\nThe octahedron figure should match the README's stated "
              "'about 17 degrees'.")
        return

    if bpy is None:
        sys.exit("This script must run inside Blender:\n"
                 "  blender --background --python blender_render.py -- --help")

    if args.puzzle:
        puzzles = [Path(args.puzzle)]
    else:
        root = Path(args.output_dir)
        if not root.is_dir():
            sys.exit(f"No such output directory: {root}")
        puzzles = find_puzzles(root, args.variant, args.level)

    if args.limit:
        puzzles = puzzles[: args.limit]

    if not puzzles:
        sys.exit("No puzzles matched.")

    print(f"Rendering {len(puzzles)} puzzle(s) with "
          f"{args.engine} at {args.width}x{args.height}")

    total = 0
    for puzzle in puzzles:
        total += render_puzzle(puzzle, args)

    print(f"\nDone: {total} image(s) from {len(puzzles)} puzzle(s).")


if __name__ == "__main__":
    main()
