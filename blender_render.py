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


def set_input(shader, names, value) -> bool:
    """
    Set the first socket that exists from `names`.

    Principled BSDF renamed several sockets in Blender 4.x ("Specular" became
    "Specular IOR Level", clearcoat became "Coat Weight"). Probing by name keeps
    one script working across versions instead of pinning it to one.
    """
    for n in names:
        if n in shader.inputs:
            shader.inputs[n].default_value = value
            return True
    return False


def make_material(name: str, color, roughness: float = 0.55,
                  shadeless: bool = False, coat: float = 0.0,
                  sheen: float = 0.0) -> Any:
    """
    A Principled BSDF material.

    `shadeless` swaps in a pure emission shader, used for diagram marks -- the
    grid rules and the start marker. Those denote rather than depict, and
    letting light fall on them would make a rule dim as it crossed the board's
    shaded side, reading as a change of meaning where none exists.

    `coat` adds a thin clear layer, which is what makes the die body look like a
    moulded object rather than flat-shaded paper: the coat carries a tight
    highlight that moves across the three visible faces and separates them even
    where the diffuse tones are close.
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
        # Matte base: a dataset die wants an even read, not broad highlights
        # that could be mistaken for pips.
        set_input(shader, ("Specular IOR Level", "Specular"), 0.3)
        if coat:
            set_input(shader, ("Coat Weight", "Clearcoat"), coat)
            set_input(shader, ("Coat Roughness", "Clearcoat Roughness"), 0.12)
        if sheen:
            set_input(shader, ("Sheen Weight", "Sheen"), sheen)

    links.new(shader.outputs[0], out.inputs["Surface"])
    return mat


def setup_world(strength: float = 1.0) -> None:
    """
    A bright, slightly graded environment.

    Pure uniform white lights every surface identically, which flattens the
    solid -- the very cue the render exists to provide. A gradient that is
    brighter overhead than at the horizon gives surfaces a direction-dependent
    ambient term, so faces angled differently pick up different light even
    where no lamp reaches them.

    The gradient is *neutral* -- white grading to a dimmer white, never a
    coloured sky. An earlier version graded toward a bluish grey, which tinted
    the white board: ambient light carries its own colour onto every diffuse
    surface, so a blue-grey environment makes a white floor render blue-grey.
    Brightness may vary here; hue may not.
    """
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputWorld")
    bg = nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = strength

    # Neutral vertical gradient: full white overhead, slightly dimmer below.
    # This is what *lights* the scene. What the camera sees behind the board is
    # forced to pure white separately below, so the page never renders grey.
    tex = nodes.new("ShaderNodeTexCoord")
    sep = nodes.new("ShaderNodeSeparateXYZ")
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0.88, 0.88, 0.88, 1.0)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    # A Light Path node splits "what lights the scene" from "what the camera
    # sees". Camera rays get flat white; every other ray keeps the gradient, so
    # the backdrop is pure #FFFFFF while surfaces still receive graded ambient
    # light. Without this the gradient's dim lower half renders as a grey page.
    lp = nodes.new("ShaderNodeLightPath")
    mix = nodes.new("ShaderNodeMixRGB")
    mix.inputs["Color2"].default_value = (1.0, 1.0, 1.0, 1.0)

    links.new(tex.outputs["Generated"], sep.inputs["Vector"])
    links.new(sep.outputs["Z"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], mix.inputs["Color1"])
    links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
    links.new(mix.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], out.inputs["Surface"])


def setup_lighting(target: Sequence[float], azimuth: float,
                   scale: float = 1.0) -> None:
    """
    A three-point rig: area key, sun fill, and a low rim.

    An AREA key rather than a bare sun, because area lights cast penumbral
    shadows: the die's contact shadow softens with distance from the base, which
    is the strongest available cue that the solid is standing *on* the board
    rather than floating above it. The first port had no shadow at all and the
    die read as pasted on.

    The key sits off the camera's azimuth so the three visible faces pick up
    distinct brightnesses -- that tonal separation is what makes the solid read
    as a cube, and the 2D renderer had to fake it with hardcoded shade factors.
    """
    def aim(obj, at):
        obj.rotation_euler = (
            Vector(at) - Vector(obj.location)
        ).to_track_quat("-Z", "Y").to_euler()

    az = math.radians(azimuth)
    # Key: offset ~50 degrees from the camera azimuth, high and to one side.
    kaz = az + math.radians(50.0)
    key_data = bpy.data.lights.new("Key", type="AREA")
    key_data.energy = 900.0 * scale * scale
    key_data.size = 9.0 * scale
    key_data.shape = "DISK"
    key = bpy.data.objects.new("Key", key_data)
    key.location = (target[0] + 11 * scale * math.cos(kaz),
                    target[1] + 11 * scale * math.sin(kaz),
                    target[2] + 15 * scale)
    aim(key, target)
    bpy.context.collection.objects.link(key)

    # Fill: broad and weak, from the camera side, to open the shadowed faces
    # without erasing the tonal separation the key just created.
    fill_data = bpy.data.lights.new("Fill", type="SUN")
    fill_data.energy = 1.1
    fill_data.angle = math.radians(30.0)
    fill = bpy.data.objects.new("Fill", fill_data)
    fill.location = (target[0] + 14 * scale * math.cos(az),
                     target[1] + 14 * scale * math.sin(az),
                     target[2] + 7 * scale)
    aim(fill, target)
    bpy.context.collection.objects.link(fill)

    # Rim: from behind, low, to put a bright edge on the solid's far side so it
    # separates from a white board at the silhouette.
    rim_data = bpy.data.lights.new("Rim", type="SUN")
    rim_data.energy = 1.6
    rim_data.angle = math.radians(8.0)
    rim = bpy.data.objects.new("Rim", rim_data)
    rim.location = (target[0] - 12 * scale * math.cos(az),
                    target[1] - 12 * scale * math.sin(az),
                    target[2] + 5 * scale)
    aim(rim, target)
    bpy.context.collection.objects.link(rim)


def setup_render(width: int, height: int, samples: int, engine: str,
                 transparent: bool = False) -> None:
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA" if transparent else "RGB"
    scene.render.film_transparent = transparent

    # Colour management: 'Standard', not Blender's default filmic/AgX tone map.
    # AgX deliberately desaturates and rolls off highlights for photographic
    # look -- on a diagram it turns the white board a muddy grey and pulls the
    # orange route toward brown, changing the very colours the palette fixed.
    try:
        scene.view_settings.view_transform = "Standard"
    except TypeError:
        pass
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    if engine == "cycles":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        # Bright, low-variance scene: few bounces are needed, and capping them
        # cuts render time sharply with no visible difference.
        scene.cycles.max_bounces = 4
        scene.cycles.diffuse_bounces = 3
        scene.cycles.glossy_bounces = 2
        scene.cycles.transmission_bounces = 2
        scene.cycles.use_fast_gi = True
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
        if eevee is not None:
            if hasattr(eevee, "taa_render_samples"):
                eevee.taa_render_samples = max(32, samples // 2)
            # Soft shadows and ambient occlusion are what give EEVEE the
            # contact cue Cycles gets from ray tracing. Names differ between
            # EEVEE Legacy and EEVEE Next, so each is probed.
            for attr, val in (("use_gtao", True), ("gtao_distance", 0.6),
                              ("use_soft_shadows", True),
                              ("use_shadow_jitter_viewport", True),
                              ("shadow_ray_count", 2),
                              ("shadow_step_count", 6)):
                if hasattr(eevee, attr):
                    try:
                        setattr(eevee, attr, val)
                    except (AttributeError, TypeError):
                        pass


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


def add_ribbon(name: str, points: Sequence[Sequence[float]], width: float,
               material, z: float) -> Any:
    """
    A flat ribbon following a polyline, mitred at the corners.

    This replaces a chain of cylinders. Cylinders were the literal translation
    of a 2D stroke, but they notch visibly at every turn and need a sphere
    patching each joint; a single mitred strip is one clean object, sits flush
    on the board, and keeps a constant apparent width from this camera because
    it is a flat surface rather than a tube whose silhouette narrows on turns.
    """
    pts = [Vector((p[0], p[1], 0.0)) for p in points]
    if len(pts) < 2:
        return None

    half = width / 2.0
    left: List[Vector] = []
    right: List[Vector] = []

    def perp(a: Vector, b: Vector) -> Vector:
        d = (b - a)
        d.z = 0.0
        if d.length < 1e-9:
            return Vector((0.0, 0.0, 0.0))
        d.normalize()
        return Vector((-d.y, d.x, 0.0))

    for i, p in enumerate(pts):
        if i == 0:
            n = perp(pts[0], pts[1])
        elif i == len(pts) - 1:
            n = perp(pts[-2], pts[-1])
        else:
            # Mitre: average the two edge normals and lengthen to keep the
            # ribbon's width constant through the corner.
            n0, n1 = perp(pts[i - 1], p), perp(p, pts[i + 1])
            n = (n0 + n1)
            if n.length < 1e-9:
                n = n0
            else:
                n.normalize()
                cosang = max(0.35, n.dot(n0))  # clamp so sharp turns stay sane
                n = n / cosang
        left.append(p + n * half)
        right.append(p - n * half)

    verts = [(v.x, v.y, z) for v in left] + [(v.x, v.y, z) for v in right]
    n = len(left)
    faces = [(i, i + 1, n + i + 1, n + i) for i in range(n - 1)]
    return new_mesh_object(name, verts, faces, material)


def add_contact_shadow(name: str, centre, size: float, z: float = 0.003,
                       strength: float = 0.30, spread: float = 0.55,
                       offset=(0.0, 0.0)) -> Any:
    """
    A soft dark patch on the board, standing in for the die's cast shadow.

    The playing surface is emissive so it renders exactly white everywhere, and
    an emissive surface cannot receive a real shadow -- the two requirements
    genuinely conflict. Painting the contact shadow as its own decal resolves
    it: the floor stays pure #FFFFFF except directly beneath the solid, which
    is the one place a shadow carries information (that the die is standing on
    the board rather than floating above it).

    A radial gradient with no hard rim, so it reads as ambient occlusion
    tightening under the body rather than as a drawn ellipse.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    # Emission shading a grey-to-white gradient, not a transparency blend.
    # Alpha blending here produced a hard opaque quad in EEVEE (blend mode and
    # shadow settings differ between engines and Blender versions); emitting
    # the shadow's *colour* against the white board needs no alpha at all, so
    # it looks the same in both engines.
    emit = nodes.new("ShaderNodeEmission")

    grad = nodes.new("ShaderNodeTexGradient")
    grad.gradient_type = "SPHERICAL"
    texco = nodes.new("ShaderNodeTexCoord")
    ramp = nodes.new("ShaderNodeValToRGB")
    # The spherical gradient is 1 at the centre and falls to 0 at the edge, so
    # position 0 is the patch rim (white, invisible against the board) and
    # position 1 is directly under the solid (darkest).
    shade = max(0.0, min(1.0, 1.0 - strength))
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    ramp.color_ramp.elements[1].position = max(0.05, min(0.95, spread))
    ramp.color_ramp.elements[1].color = (shade, shade, shade, 1.0)
    # Ease the falloff so the rim blends into the board instead of banding.
    for el in ramp.color_ramp.elements:
        el.color = tuple(el.color)
    ramp.color_ramp.interpolation = "EASE"

    links.new(texco.outputs["Object"], grad.inputs["Vector"])
    links.new(grad.outputs["Color"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], emit.inputs["Color"])
    links.new(emit.outputs[0], out.inputs["Surface"])

    # Built at the origin and then positioned: the gradient is driven by Object
    # coordinates, so the mesh must be centred on its own origin or the falloff
    # is sampled off-centre and the patch renders as a flat slab.
    half = size / 2.0
    obj = new_mesh_object(
        name,
        [(-half, -half, 0.0), (half, -half, 0.0),
         (half, half, 0.0), (-half, half, 0.0)],
        [(0, 1, 2, 3)],
        mat,
    )
    obj.location = (centre[0] + offset[0], centre[1] + offset[1], z)
    # Never let the decal catch light or cast its own shadow.
    if hasattr(obj, "visible_shadow"):
        obj.visible_shadow = False
    return obj


def draw_route(points: Sequence[Sequence[float]], color: str, dashed: bool,
               z: float, width: float = 0.14, name: str = "route") -> None:
    """
    Draw a route as flat ribbons capped with a single arrowhead.

    The line is pulled back from the tip so the head caps it instead of poking
    through, matching the 2D renderer's `head_len` trim. Route marks are
    emissive: they annotate the scene rather than inhabit it, so they must read
    identically over the lit and shadowed halves of the board.
    """
    if len(points) < 2:
        return
    mat = make_material(f"{name}_mat", rgba(color), shadeless=True)

    head_len = 0.30
    head_width = 0.34
    x0, y0 = points[-2][0], points[-2][1]
    x1, y1 = points[-1][0], points[-1][1]
    dx, dy = x1 - x0, y1 - y0
    norm = math.hypot(dx, dy) or 1.0
    ux, uy = dx / norm, dy / norm
    # Size the head against the final segment: a fixed head swallows a short
    # step whole, and the octahedron lattice has two different step lengths.
    head = min(head_len, norm * 0.45)

    # Shorten the last segment by the head length, so the shaft stops where the
    # arrowhead begins.
    trimmed = [tuple(p[:2]) for p in points[:-1]]
    trimmed.append((x1 - ux * head, y1 - uy * head))

    if dashed:
        for i, (a, b) in enumerate(dash_segments(trimmed, dash=0.34, gap=0.22)):
            add_ribbon(f"{name}_dash_{i}", [a, b], width, mat, z)
    else:
        add_ribbon(f"{name}_shaft", trimmed, width, mat, z)

    # Arrowhead as a flat triangle, matching the ribbon it caps.
    tipx, tipy = x1, y1
    bx, by = x1 - ux * head, y1 - uy * head
    px, py = -uy, ux
    new_mesh_object(
        f"{name}_head",
        [(tipx, tipy, z),
         (bx + px * head_width / 2.0, by + py * head_width / 2.0, z),
         (bx - px * head_width / 2.0, by - py * head_width / 2.0, z)],
        [(0, 1, 2)],
        mat,
    )


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
    """The board: a white slab, grid rules as flat inlays, blocked cells inset."""
    bw, bh = board["width"], board["height"]
    thickness = 0.18

    # The playing surface is emissive white, so it renders exactly #FFFFFF
    # everywhere -- no shading falloff across the board, no cast shadow, and no
    # tint picked up from the environment. A lit diffuse floor cannot do that:
    # it necessarily darkens away from the key light, which greys the far half
    # of the board and defeats the palette's white surface.
    #
    # The die is unaffected. It is still fully lit and still casts its shadow
    # onto everything that is lit; only this one surface refuses to receive it.
    top_mat = make_material("board_top", rgba(BOARD_SURFACE), shadeless=True)
    # The side walls stay lit, so the slab still reads as a solid object with
    # thickness rather than a glowing sheet of paper.
    side_mat = make_material("board_side", rgba(BOARD_SURFACE), roughness=0.85,
                             sheen=0.1)

    verts = [
        (0, 0, 0), (bw, 0, 0), (bw, bh, 0), (0, bh, 0),
        (0, 0, -thickness), (bw, 0, -thickness),
        (bw, bh, -thickness), (0, bh, -thickness),
    ]
    # Face 0 is the top; the rest are the underside and the four walls.
    faces = [(0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1),
             (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    slab = new_mesh_object("board", verts, faces, top_mat)
    slab.data.materials.append(side_mat)
    for poly in slab.data.polygons:
        poly.material_index = 0 if poly.index == 0 else 1

    slab_bevel = slab.modifiers.new("Bevel", "BEVEL")
    slab_bevel.width = 0.02
    slab_bevel.segments = 2
    slab_bevel.limit_method = "ANGLE"

    # Grid rules as flat inlays rather than tubes: a tube on a white board
    # catches a highlight along its crown and reads heavier than intended, and
    # its apparent width changes with the viewing angle. A flat strip lying on
    # the surface keeps the even, drawn weight the 2D renderer had.
    grid_mat = make_material("grid", rgba(GRID_LINE), shadeless=True)
    for x in range(bw + 1):
        add_ribbon(f"grid_x{x}", [(x, 0), (x, bh)], 0.022, grid_mat, 0.005)
    for y in range(bh + 1):
        add_ribbon(f"grid_y{y}", [(0, y), (bw, y)], 0.022, grid_mat, 0.005)

    # Outer border: heavier, and true black, so the board's extent is
    # unambiguous now that the surface matches the page.
    #
    # Inset by half its own width so the ribbon lies wholly on the flat top
    # rather than draping over the slab's bevelled lip, where the rounding cut
    # it off and left the border broken along the near edges.
    edge_mat = make_material("board_edge", rgba(BOARD_EDGE), shadeless=True)
    b = 0.03
    corners = [(b, b), (bw - b, b), (bw - b, bh - b), (b, bh - b), (b, b)]
    add_ribbon("border", corners, 0.06, edge_mat, 0.009)

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
    # reads as a flat hexagon at this scale; a rounded edge catches a highlight
    # along each corner and separates the three visible faces even where their
    # diffuse tones are close. This is the single change that most makes the
    # die look like an object rather than a shaded polygon.
    bevel = cube.modifiers.new("Bevel", "BEVEL")
    bevel.width = 0.045
    bevel.segments = 6
    bevel.limit_method = "ANGLE"
    bevel.angle_limit = math.radians(30.0)
    bevel.harden_normals = True

    # Smooth shading with an angle split, so the bevel reads as a rounded edge
    # while the faces stay flat. Without this the bevel facets show as bands.
    for poly in cube.data.polygons:
        poly.use_smooth = True
    if hasattr(cube.data, "use_auto_smooth"):       # Blender < 4.1
        cube.data.use_auto_smooth = True
        cube.data.auto_smooth_angle = math.radians(30.0)
    else:                                            # Blender >= 4.1
        smooth = cube.modifiers.new("SmoothByAngle", "NODES")
        try:
            ng = bpy.data.node_groups.get("Smooth by Angle")
            if ng is None:
                bpy.ops.object.modifier_remove(modifier=smooth.name)
            else:
                smooth.node_group = ng
        except Exception:
            pass

    # A clear coat gives the body a moulded, injection-plastic read.
    body_mat = make_material("die_body", rgba(DIE_BODY), roughness=0.42,
                             coat=0.35)
    cube.data.materials.append(body_mat)

    pip_mat = make_material("pip", rgba(PIP_COLOR), roughness=0.35, coat=0.2)
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
            # A pip that sits fully on the surface reads as a sticker; one sunk
            # this far catches its own small shadow at the rim, which is what
            # makes it look moulded into the body.
            depth = pip_radius * 0.62
            loc = (cx + lx - normal[0] * depth,
                   cy + ly - normal[1] * depth,
                   cz + lz - normal[2] * depth)
            bpy.ops.mesh.primitive_uv_sphere_add(radius=pip_radius, location=loc,
                                                 segments=24, ring_count=14)
            pip = bpy.context.active_object
            pip.name = f"pip_{value}_{i}"
            pip.data.materials.clear()
            pip.data.materials.append(pip_mat)
            for poly in pip.data.polygons:
                poly.use_smooth = True

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
          (sx + 0.94, sy + 0.94), (sx + 0.06, sy + 0.94), (sx + 0.06, sy + 0.06)]
    for j, (a, b) in enumerate(dash_segments(mc, dash=0.16, gap=0.11)):
        add_ribbon(f"marker_{j}", [a, b], 0.05, marker_mat, 0.012)

    # Contact shadow first, so the die is built over it. Offset away from the
    # key light, which sits 50 degrees off the camera azimuth.
    px, py = entry["position"]
    add_contact_shadow("die_shadow", (px + 0.5, py + 0.5), size=2.1,
                       strength=0.32, spread=0.42, offset=(-0.12, -0.12))

    build_die(entry["position"], entry["die"])

    target = (bw / 2.0, bh / 2.0, 0.0)
    # Fit the board's diagonal, plus room for a die standing at a far corner.
    ortho = math.hypot(bw, bh) * 0.80 + 1.2
    setup_camera(target, CUBE_ELEVATION, CUBE_AZIMUTH, ortho)
    setup_lighting(target, CUBE_AZIMUTH, scale=max(1.0, max(bw, bh) / 6.0))
    setup_render(args.width or 1000, args.height or 1000,
                 args.samples, args.engine, args.transparent)

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
    # Emissive white, for the same reason as the cube board: a lit floor
    # darkens away from the key light and takes the solid's shadow, both of
    # which grey a surface the palette specifies as white.
    surface_mat = make_material("oct_board", rgba(BOARD_SURFACE), shadeless=True)
    grid_mat = make_material("oct_grid", rgba(GRID_LINE), shadeless=True)

    all_pts = []
    cells_pts = []
    for key, cell in board.items():
        poly = oct_.cell_polygon(cell)
        pts = [(float(px), float(py)) for px, py in poly]
        all_pts.extend(pts)
        cells_pts.append(pts)
        new_mesh_object(
            f"cell_{key[0]}_{key[1]}",
            [(px, py, 0.0) for px, py in pts],
            [(0, 1, 2)],
            surface_mat,
        )

    # Lattice rules as flat inlays, one ribbon per shared edge. Deduplicated:
    # every interior edge belongs to two triangles, and drawing it twice
    # doubles its apparent weight against the single-drawn boundary edges.
    seen = set()
    for pts in cells_pts:
        for i in range(3):
            a, b = pts[i], pts[(i + 1) % 3]
            key = tuple(sorted((tuple(round(c, 4) for c in a),
                                tuple(round(c, 4) for c in b))))
            if key in seen:
                continue
            seen.add(key)
            add_ribbon(f"rule_{len(seen)}", [a, b], 0.022, grid_mat, 0.005)

    # Border: the convex hull of the cells, matching the 2D renderer. The kept
    # region's true boundary zigzags, and tracing it exactly puts heavy black
    # lines through the middle of the board.
    edge_mat = make_material("oct_edge", rgba(BOARD_EDGE), shadeless=True)
    hull = convex_hull(all_pts)
    add_ribbon("oct_border", list(hull) + [hull[0]], 0.06, edge_mat, 0.007)


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

    body_mat = make_material("oct_body", rgba(DIE_BODY), roughness=0.42,
                             coat=0.35)
    obj = new_mesh_object("octahedron", verts, oct_.FACES, body_mat)

    # A small bevel rounds the edges so each catches a highlight. Unlike the
    # cube this stays modest: eight triangles meet at six points, and a wide
    # bevel pinches badly at those vertices.
    bevel = obj.modifiers.new("Bevel", "BEVEL")
    bevel.width = 0.018
    bevel.segments = 3
    bevel.limit_method = "ANGLE"
    bevel.angle_limit = math.radians(20.0)
    bevel.harden_normals = True

    # Edges still get their own dark geometry on top of the bevel: adjacent
    # faces of an octahedron meet at a shallow angle, so lighting alone leaves
    # neighbouring faces nearly the same tone and their numbers appear to sit
    # on one continuous surface. The drawn edge is what keeps the faces
    # countable, which the task depends on.
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
                radius=0.014, depth=d.length, location=tuple((va + vb) / 2.0),
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
            # Direction from the board toward the viewer. This is exactly
            # octahedron.CAMERA, which is why face visibility above and glyph
            # facing here stay consistent by construction.
            view=(float(camera[0]), float(camera[1]), float(camera[2])),
        )


def add_face_number(value: int, centre, normal, material,
                    view: "Vector" = None) -> Any:
    """
    Place a face's number, anchored to that face but turned to face the camera.

    The label is **billboarded**: it takes the camera's own rotation rather
    than a frame built from the face normal. Deriving the frame from the face
    is the obvious approach and it does not work -- an octahedron's faces are
    tilted in three axes at once, so a basis that is correctly right-handed can
    still present the glyph edge-on, rolled, or seen from its reverse side,
    which renders it mirrored. A mirrored 5 reads as a 2, silently corrupting
    the value the task asks the model to read off the picture.

    Billboarding sidesteps all of it: every digit is upright and unmirrored by
    construction, which is what the 2D renderer got for free by drawing text at
    a projected centroid. The label is nudged along the face normal so it sits
    clear of the surface it belongs to.
    """
    curve = bpy.data.curves.new(f"num_{value}", type="FONT")
    curve.body = str(value)
    curve.align_x = "CENTER"
    curve.align_y = "CENTER"
    curve.size = 0.34
    curve.extrude = 0.004

    obj = bpy.data.objects.new(f"num_{value}", curve)
    obj.data.materials.append(material)
    bpy.context.collection.objects.link(obj)

    n = Vector(normal).normalized()
    # Lift the glyph off its face, along that face's own normal, so it is never
    # buried in the surface or z-fighting with it.
    #
    # The offset is along the normal and nothing else. An earlier version also
    # pulled the label toward the solid's centre, meaning to stop the topmost
    # digit overhanging the silhouette; it instead dragged every label off its
    # own face and onto its neighbour, which is far worse than a slight
    # crop -- the digit must stay on the face whose value it reports.
    lift = 0.06
    obj.location = (centre[0] + n.x * lift,
                    centre[1] + n.y * lift,
                    centre[2] + n.z * lift)

    cam = bpy.context.scene.camera
    if cam is not None:
        obj.rotation_euler = cam.rotation_euler.copy()
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
                   width=0.10, name="oct_done")
    if step < len(pts) - 1:
        draw_route(pts[step:], REMAINING_PATH, dashed=True, z=0.02,
                   width=0.10, name="oct_todo")

    # Mark the START cell. Marking the destination gives away half the answer:
    # the route already ends there.
    marker_mat = make_material("oct_marker", rgba(REMAINING_PATH), shadeless=True)
    key = (round(trace[0]["cell"][0], 2), round(trace[0]["cell"][1], 2))
    if key in board:
        poly = [(float(px), float(py)) for px, py in oct_.cell_polygon(board[key])]
        for j, (a, b) in enumerate(
                dash_segments(poly + [poly[0]], dash=0.16, gap=0.11)):
            add_ribbon(f"oct_marker_{j}", [a, b], 0.048, marker_mat, 0.014)

    # Contact shadow under the resting face. Tighter than the cube's: the
    # octahedron touches the board on one small triangle, so a broad patch
    # would read as a shadow belonging to something else.
    add_contact_shadow("oct_shadow", (at[0], at[1]), size=1.5,
                       strength=0.30, spread=0.38, offset=(-0.06, -0.06))

    xs = [c[0] for c in [cell["centre"] for cell in board.values()]]
    ys = [c[1] for c in [cell["centre"] for cell in board.values()]]
    target = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, 0.35)
    ortho = max(max(xs) - min(xs), 3.0) * 1.30 + 1.5

    # Camera before the solid: the face labels are rolled against the camera's
    # up axis so they read upright on screen, which needs the camera in place.
    setup_camera(target, OCT_ELEVATION, OCT_AZIMUTH, ortho)
    build_octahedron(at, orientation, oct_)
    setup_lighting(target, OCT_AZIMUTH, scale=max(1.0, ortho / 6.0))
    # Wide by default: the cropped lattice is a band, matching the 10x6.5 figure
    # the 2D renderer used.
    setup_render(args.width or 1500, args.height or 975,
                 args.samples, args.engine, args.transparent)

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

    # Resolve the per-variant resolution defaults here, so the renderers and
    # the log agree on one number rather than each filling in its own.
    if args.width is None or args.height is None:
        dw, dh = (1500, 975) if variant == "octahedron" else (1000, 1000)
        args = argparse.Namespace(**vars(args))
        args.width = args.width or dw
        args.height = args.height or dh
    print(f"  {puzzle_dir.name}: {variant} at {args.width}x{args.height}")

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
    # Left unset by default so each variant gets the aspect its 2D renderer
    # used: the cube board is square, while the octahedron's cropped lattice is
    # a wide band and squeezing it into a square wastes most of the frame.
    p.add_argument("--width", type=int, default=None,
                   help="Default: 1000 (cube) / 1500 (octahedron)")
    p.add_argument("--height", type=int, default=None,
                   help="Default: 1000 (cube) / 975 (octahedron)")
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

    size = ("per variant" if args.width is None or args.height is None
            else f"{args.width}x{args.height}")
    print(f"Rendering {len(puzzles)} puzzle(s) with {args.engine} at {size}")

    total = 0
    for puzzle in puzzles:
        total += render_puzzle(puzzle, args)

    print(f"\nDone: {total} image(s) from {len(puzzles)} puzzle(s).")


if __name__ == "__main__":
    main()
