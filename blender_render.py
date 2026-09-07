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
# Warmer and a touch deeper than generator.py's #DED8C6. That value was chosen
# against the 2D renderer's flat fill; in a lit 3D render the solid's brightest
# face washes out toward the white board and its faintest sits close to the grey
# grid, so the body could read as part of the board rather than as an object on
# it. This cream lifts contrast against the white surface (1.44:1, up from
# 1.42:1) while staying pale enough for black pips and numbers to carry.
DIE_BODY = "#E8D5AC"
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
_OCT_BASIS_ELEVATION, OCT_AZIMUTH = _camera_from_basis(OCT_BASIS)

# The octahedron is rendered from a little higher than its 2D basis implies.
#
# The 16.86 degrees the basis encodes was chosen for a flat drawing, where the
# lattice only had to be readable as line work. In a shaded 3D render it is too
# shallow: a lattice cell projects just 0.25 as tall as it is wide, so the board
# collapses toward a band and the route is hard to follow across it. Measured
# across elevations, 26 degrees raises that to 0.38 -- half again as legible --
# while the smallest camera-facing face only drops from 0.251 to 0.240 in
# projected area, so all four faces keep room for their number.
#
# Still four visible faces: an octahedron is convex, so exactly half its faces
# point away from any viewpoint. No angle shows more, and test_blender_scene.py
# asserts the count rather than trusting this comment.
#
# Raising this further trades board legibility for face legibility: at 40-45
# degrees the two faces angled away from the camera widen noticeably (they keep
# only ~28% of their width at 26 degrees), at the cost of looking down on the
# board rather than across it. `--elevation` overrides it per run.
#
# It was briefly set to the cube's 29.496 so the two solids were seen from one
# point. That is tidier in principle and worse in practice here: it flattens
# the lattice further and buys nothing the shallower angle does not already
# give, so it is back to the value the rest of the framing was tuned around.
OCT_ELEVATION = 26.0

# Orthographic width the octahedron is framed at, for every puzzle.
#
# Set to what frames the widest cropped board in the dataset, so nothing is cut
# off and, more importantly, the die is drawn at one constant size -- deriving
# it per puzzle made the solid change scale between frames a reader is meant to
# compare, varying between 7.12 and 8.90.
#
# Sized for the margin=2 lattice in octahedron._board_near_path, whose widest
# board measures 5.69 units. The two move together: widening the margin without
# widening this crops the lattice, and widening this without the margin leaves
# the die small in a lot of empty space -- at margin=4 and ortho 11.80 the die
# fell from 49% of the frame's width to 36%, which is what made it hard to
# read.
OCT_ORTHO_SCALE = 8.90


# Where setup_camera last placed the camera, and what it was aimed at. Read
# back instead of recovering the direction from the camera object.
#
# `rotation_euler` is assigned but `matrix_world` is only recomputed when the
# dependency graph next evaluates, so querying the object's orientation during
# scene construction returns the identity -- which silently reports the view as
# straight down the +Z axis and mislabels which faces are visible. Recording
# the placement avoids depending on evaluation order at all.
_CAMERA_PLACEMENT: Dict[str, Tuple[float, float, float]] = {}


def view_direction() -> Tuple[float, float, float]:
    """
    Direction from the scene toward the camera, as a unit vector.

    Same convention as `octahedron.CAMERA`: it points *out* of the board at the
    viewer, so a face is visible when its outward normal has a positive dot
    product with it.
    """
    loc = _CAMERA_PLACEMENT.get("location")
    target = _CAMERA_PLACEMENT.get("target")
    if loc is None or target is None:
        return (0.0, 0.0, 1.0)
    d = Vector(loc) - Vector(target)
    if d.length < 1e-9:
        return (0.0, 0.0, 1.0)
    d.normalize()
    return (d.x, d.y, d.z)


# Camera height for `--natural`. Matches the cube's own elevation, which is
# where the request came from: the cube is viewed from slightly further above,
# and its faces then sit at a consistent +/-26 degrees on screen, which is what
# reads as ordinary perspective rather than as a diagram.
#
# It is only a small lift over the default 26. Going higher does not help this
# solid -- measured across the 24 orientations, face-aligned digits tilt a
# median 37 degrees at 26 and 51 at 45, because an octahedron's triangles never
# square up to the view the way a cube's quads do.
OCT_NATURAL_ELEVATION = 29.496


def elevation_used(args) -> float:
    """The octahedron camera elevation this run is using."""
    explicit = getattr(args, "elevation", None)
    if explicit:
        return explicit
    if getattr(args, "natural", False):
        return OCT_NATURAL_ELEVATION
    return OCT_ELEVATION


def camera_frame() -> Tuple["Vector", "Vector"]:
    """
    The camera's right and up axes in world space.

    Derived from the recorded placement for the same reason as
    `view_direction`: the camera object's own matrix is not yet valid while the
    scene is still being built.
    """
    view = Vector(view_direction())          # scene -> camera
    world_up = Vector((0.0, 0.0, 1.0))
    if abs(view.dot(world_up)) > 0.999:
        world_up = Vector((0.0, 1.0, 0.0))
    right = world_up.cross(view).normalized()
    up = view.cross(right).normalized()
    return right, up


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

    # Record the placement so view_direction()/camera_frame() can report the
    # view without reading back a matrix the depsgraph has not refreshed yet.
    _CAMERA_PLACEMENT["location"] = tuple(float(v) for v in loc)
    _CAMERA_PLACEMENT["target"] = tuple(float(v) for v in target)
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

    # Strength scales the whole background, which would dim the page along with
    # the lighting. Divide it back out on the camera-ray branch so the visible
    # backdrop stays pure white however weak the ambient light is set: the two
    # jobs of the world -- what the camera sees, and how much it lights the
    # scene -- have to be tuned independently.
    compensate = nodes.new("ShaderNodeMixRGB")
    compensate.blend_type = "MULTIPLY"
    compensate.inputs["Fac"].default_value = 1.0
    inv_strength = 1.0 / strength if strength > 1e-6 else 1.0
    compensate.inputs["Color2"].default_value = (
        inv_strength, inv_strength, inv_strength, 1.0)
    mix.inputs["Color2"].default_value = (1.0, 1.0, 1.0, 1.0)

    links.new(tex.outputs["Generated"], sep.inputs["Vector"])
    links.new(sep.outputs["Z"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], mix.inputs["Color1"])
    links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
    # Compensate BEFORE the camera/lighting split, on the white the camera sees,
    # so only the backdrop is scaled back up and the ambient stays as dim as
    # `strength` asks. Applying it after the mix scaled both branches, which
    # silently undid every reduction in ambient.
    compensate.inputs["Color1"].default_value = (1.0, 1.0, 1.0, 1.0)
    links.new(compensate.outputs["Color"], mix.inputs["Color2"])
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
    # Key offset from the camera azimuth.
    #
    # Solved against the solid's own normals rather than picked. One light
    # cannot give four faces four tones here: swing it far enough to part the
    # two side faces and one of them drops to zero and is lit only by ambient.
    # 35 degrees keeps every face on the lit side of the key, and the
    # counter-fill below supplies the difference the key cannot.
    kaz = az + math.radians(55.0)
    key_data = bpy.data.lights.new("Key", type="AREA")
    # Sized so the brightest face stays inside the palette instead of blowing
    # out. At 900 the face most squarely lit clipped to pure white -- measured
    # on the octahedron's top face, which meets the key at 0.81 -- so the solid
    # lost its cream body colour exactly where the number sits.
    # Lower than before, because the key now strikes the front face more
    # squarely: at 380 it clipped that face to 255,249,209 and the digit lost
    # its background.
    # Strong relative to the ambient. The two are one setting, not two: with a
    # bright world the faces all sat within 4 luminance of each other, because
    # ambient arrives equally from every direction and cannot separate them.
    # Dropping the world to 0.35 and raising the key threefold makes the key the
    # thing that shapes the solid -- measured, the closest pair of faces goes
    # from 4 apart to 13.
    key_data.energy = 720.0 * scale * scale
    key_data.size = 9.0 * scale
    key_data.shape = "DISK"
    key = bpy.data.objects.new("Key", key_data)
    # Lower than it was (15), so the light rakes across the side faces instead
    # of falling mostly on the top one.
    # Placed by an explicit elevation (18 degrees) rather than a height, because
    # that is the quantity the separation depends on: solved against the
    # solid's own normals, 55 degrees round at 18 up is what parts the two
    # side faces, which sit symmetrically about the view axis and therefore
    # take the same light from anything more overhead.
    kel = math.radians(18.0)
    kdist = 14.0 * scale
    key.location = (target[0] + kdist * math.cos(kel) * math.cos(kaz),
                    target[1] + kdist * math.cos(kel) * math.sin(kaz),
                    target[2] + kdist * math.sin(kel))
    aim(key, target)
    bpy.context.collection.objects.link(key)

    # Fill: broad and weak, from the camera side, to open the shadowed faces
    # without erasing the tonal separation the key just created.
    fill_data = bpy.data.lights.new("Fill", type="SUN")
    # Counter-fill, not an even fill: it comes from the opposite side of the
    # camera to the key, so the face the key leaves darkest is the one it
    # lifts. That is what gives four distinct tones instead of three plus a
    # near-duplicate -- measured, the closest pair of faces goes from 0.04
    # apart to 0.16.
    fill_data.energy = 0.55
    fill_data.angle = math.radians(30.0)
    fill = bpy.data.objects.new("Fill", fill_data)
    faz = az - math.radians(70.0)
    fill.location = (target[0] + 14 * scale * math.cos(faz),
                     target[1] + 14 * scale * math.sin(faz),
                     target[2] + 7 * scale)
    aim(fill, target)
    bpy.context.collection.objects.link(fill)

    # Rim: from behind, low, to put a bright edge on the solid's far side so it
    # separates from a white board at the silhouette.
    rim_data = bpy.data.lights.new("Rim", type="SUN")
    # Kept very low deliberately. A rim light is aimed back at the camera, so a
    # face turned away from the viewer meets it almost head-on and blows out
    # long before the faces the viewer is reading do. That is what left the
    # octahedron's rear-upper face pure white, losing both its body colour and
    # its number -- a face the task needs legible.
    #
    # Measured on that face: 1.6 clipped to 255,255,255 and stayed clipped even
    # at 0.45; 0.30 reached 255,255,238; 0.20 held at 251,239,216. 0.15 keeps a
    # margin below the clip while still parting the silhouette from the white
    # board behind it.
    rim_data.energy = 0.15
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

    # One quad per segment, with a round joint filling each corner.
    #
    # A single mitred strip is tidier but fails on this lattice: its corner
    # vertices are pushed out by 1/cos(half-angle), and the triangular grid
    # turns 60 degrees at a time, so on a route that keeps turning the same way
    # the strip inflates until it folds back through itself and the line reads
    # as broken. Building each segment separately cannot fold, whatever the
    # route does; the joints keep the corners from showing a notch.
    verts: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, ...]] = []
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        n = perp(a, b)
        if n.length < 1e-9:
            continue
        base = len(verts)
        verts.extend([
            (a.x + n.x * half, a.y + n.y * half, z),
            (b.x + n.x * half, b.y + n.y * half, z),
            (b.x - n.x * half, b.y - n.y * half, z),
            (a.x - n.x * half, a.y - n.y * half, z),
        ])
        faces.append((base, base + 1, base + 2, base + 3))

    # Round each interior joint, so corners read as a continuous turn rather
    # than as two strips meeting at a notch.
    for i in range(1, len(pts) - 1):
        p = pts[i]
        base = len(verts)
        steps = 10
        verts.append((p.x, p.y, z))
        for s in range(steps + 1):
            t = 2.0 * math.pi * s / steps
            verts.append((p.x + math.cos(t) * half, p.y + math.sin(t) * half, z))
        for s in range(steps):
            faces.append((base, base + 1 + s, base + 2 + s))

    if not faces:
        return None
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

# ============================================================================
# Numbers on the octahedron
#
# The cube carries its values as sunk spheres and never had a placement problem,
# because a sphere looks the same from every angle. Numerals do not: they have a
# top and a reading direction, and an octahedron's faces are tilted in three axes
# at once. Placing each digit as its own 3D object meant solving orientation,
# handedness and foreshortening by hand, and every attempt left some face
# mirrored, rolled or squashed to an unreadable sliver.
#
# Painting the digits into the body's texture removes the problem instead of
# fighting it. They are drawn once into an image, each face is given its own
# square of that image, and Blender's texture sampling puts them on the surface.
# A number can then no longer be detached, mirrored or rolled: it is part of the
# face. The UV square is aligned to the *camera's* axes rather than the face's
# edges, so the digit reads upright on screen -- measured across all 24
# orientations, aligning to the face's own edges leaves 75% of visible faces
# between 37 and 60 degrees off level, which no choice of corner can fix.
# ============================================================================

# How much of a tile the digit's ink occupies. The UV mapping is scaled against
# this same number, so the glyph is guaranteed to land inside the triangle it
# belongs to rather than overhanging an edge.
GLYPH_FRACTION = 0.30

# How large a digit appears on a face. Larger values map the face to a larger
# region of the tile, so the digit -- which stays a fixed share of that tile --
# ends up *smaller* on the face. The relationship is inverse.
#
# This is the knob to turn for the drawn size; GLYPH_FRACTION only sets how
# much of the atlas tile the ink occupies, which is a resolution choice.
GLYPH_SCALE = 0.58


# Digits come from a real font, baked to outlines ahead of time.
#
# They were hand-built from arcs and lines here for a while, on the reasoning
# that a font file might not exist wherever this runs. The reasoning was sound
# and the result was not: every digit had to be shaped by hand, and each fix
# traded one flaw for another -- the 6's spine met its bowl at a kink, and
# correcting that made it lean like a b.
#
# `bake_digits.py` writes `digit_outlines.json` from the font matplotlib ships
# with itself, so the shapes are properly designed and reproducible, and this
# module needs nothing at render time -- Blender's bundled Python has no
# matplotlib, which is what ruled out reading the font here directly.
_GLYPH_FAMILY = "Avenir"
_GLYPH_WEIGHT = "normal"
_OUTLINE_FILE = "digit_outlines.json"
_SYMBOL_FILE = "symbol_outlines.json"

# Marks a face can carry. Digits are the default; symbols are an alternative
# set for when the values need telling apart at a glance rather than reading.
#
# A symbol also cannot be misread by rotation, which digits can: a 6 and a 9
# are the same shape turned round, and on a solid that tumbles, that is a real
# hazard. `bake_symbols.py` writes them, in the order face values 1..8 take.
SYMBOL_ORDER = ["heart", "arrow", "triangle", "moon", "star", "house",
                "circle", "square"]

_DIGIT_OUTLINES: Dict[int, List[List[Tuple[float, float]]]] = {}
_SYMBOL_OUTLINES: Dict[int, List[List[Tuple[float, float]]]] = {}

# Which set the renderer is currently drawing with.
USE_SYMBOLS = False

# Faces that carry a symbol while the rest carry digits, as a set of face
# values. Empty means the choice is uniform and USE_SYMBOLS decides it.
#
# Mixing is per die rather than per face at random each frame: the marking is a
# property of the solid, so a face that shows a heart in one frame has to show
# a heart in every frame of that puzzle, or the pictures would contradict each
# other about what the die is.
MIXED_SYMBOL_FACES: set = set()


def choose_mixed_faces(ratio: float, seed: int, values) -> set:
    """
    Pick which face values carry symbols, for a given ratio and seed.

    Deterministic in the seed, so every frame of a puzzle marks the same faces,
    and two runs of the same puzzle agree.
    """
    import random as _random

    vals = sorted(values)
    n = max(0, min(len(vals), int(round(len(vals) * ratio))))
    return set(_random.Random(seed).sample(vals, n))


def _load_digit_outlines() -> Dict[int, List[List[Tuple[float, float]]]]:
    """The digit set, read once."""
    global _DIGIT_OUTLINES
    if _DIGIT_OUTLINES:
        return _DIGIT_OUTLINES
    path = Path(__file__).resolve().parent / _OUTLINE_FILE
    with open(path) as f:
        raw = json.load(f)
    _DIGIT_OUTLINES = {int(k): [[(p[0], p[1]) for p in poly] for poly in v]
                       for k, v in raw.items()}
    return _DIGIT_OUTLINES


def _load_symbol_outlines() -> Dict[int, List[List[Tuple[float, float]]]]:
    """The symbol set, read once, keyed by the face value each stands for."""
    global _SYMBOL_OUTLINES
    if _SYMBOL_OUTLINES:
        return _SYMBOL_OUTLINES
    path = Path(__file__).resolve().parent / _SYMBOL_FILE
    with open(path) as f:
        raw = json.load(f)
    _SYMBOL_OUTLINES = {
        i + 1: [[(p[0], p[1]) for p in poly] for poly in raw[name]]
        for i, name in enumerate(SYMBOL_ORDER) if name in raw
    }
    return _SYMBOL_OUTLINES


def outlines_for(value: int) -> List[List[Tuple[float, float]]]:
    """
    The contours marking `value`, from whichever set that face uses.

    A face is a symbol when it is in MIXED_SYMBOL_FACES, or when the whole die
    is symbols. Both files hold the same thing -- closed contours filled
    even-odd -- so everything downstream treats them identically.
    """
    if value in MIXED_SYMBOL_FACES or (USE_SYMBOLS and not MIXED_SYMBOL_FACES):
        return _load_symbol_outlines().get(value, [])
    return _load_digit_outlines().get(value, [])


def _load_outlines() -> Dict[int, List[List[Tuple[float, float]]]]:
    """
    Every mark, each from the set its face uses.

    Kept as a dict so callers that want the whole set -- the net sheet, the
    tests -- do not have to know how the choice is made.
    """
    keys = set(_load_digit_outlines()) | set(_load_symbol_outlines())
    return {v: outlines_for(v) for v in sorted(keys)}
    path = Path(__file__).resolve().parent / _OUTLINE_FILE
    with open(path) as f:
        raw = json.load(f)
    _DIGIT_OUTLINES = {int(k): [[(p[0], p[1]) for p in poly] for poly in v]
                       for k, v in raw.items()}
    return _DIGIT_OUTLINES


_COVERAGE_CACHE: Dict[Tuple[int, int], Dict[Tuple[int, int], float]] = {}


def _digit_coverage(value: int, size: int):
    """
    Per-pixel ink coverage for `value`, as a dict of (x, y) -> 0..1.

    Coverage rather than a yes/no mask, because a binary mask renders curves as
    a visible staircase at the size a face gets on screen. Each pixel is
    sampled on a subgrid and the fraction of samples inside the glyph becomes
    its alpha, so edge pixels land part-way between ink and body.

    Insideness uses the even-odd rule across all of a digit's contours, which
    is what keeps the counters open -- the hole in a 6, both holes in an 8.
    """
    # Cached: the atlas is rebuilt for every frame, but a digit's coverage
    # depends only on the value and the size, so it is computed once per run.
    # The set a face uses is part of the key: the same value is a different
    # shape depending on it, and a cache that ignored that would paint a digit
    # where a symbol belongs.
    is_symbol = value in MIXED_SYMBOL_FACES or (USE_SYMBOLS
                                                and not MIXED_SYMBOL_FACES)
    key = (value, size, is_symbol)
    if key in _COVERAGE_CACHE:
        return _COVERAGE_CACHE[key]

    contours = outlines_for(value)
    if not contours:
        return {}

    xs = [p[0] for c in contours for p in c]
    ys = [p[1] for c in contours for p in c]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    gw, gh = x1 - x0, y1 - y0
    if gw <= 0 or gh <= 0:
        return {}

    # Fit the mark into the box, preserving its proportions and centring it.
    #
    # Symbols get more of the box than digits. A numeral is read by its
    # skeleton and survives being small, while a shape is read by its
    # silhouette -- a circle and a heart at digit size collapse toward the same
    # blob on a face turned away from the camera.
    fill = 1.02 if is_symbol else 0.92
    scale = min(size / gw, size / gh) * fill
    ox = (size - gw * scale) / 2.0
    oy = (size - gh * scale) / 2.0

    # Scanline fill rather than a point test per sample.
    #
    # Testing every subsample against every edge was correct and far too slow:
    # 32 seconds for one glyph at the size the atlas uses, and the atlas is
    # built per frame. Crossings only change along a row, so each scanline is
    # solved once and the spans between crossings are filled directly.
    def spans(y: float):
        """x-intervals of ink along the horizontal line at `y`, even-odd."""
        xs = []
        for c in contours:
            n = len(c)
            for i in range(n):
                ax, ay = c[i]
                bx, by = c[(i + 1) % n]
                if (ay > y) != (by > y):
                    xs.append(ax + (y - ay) / (by - ay) * (bx - ax))
        xs.sort()
        return [(xs[i], xs[i + 1]) for i in range(0, len(xs) - 1, 2)]

    SUB = 3                                   # 3x3 samples per pixel
    step = 1.0 / SUB
    offset = step / 2.0
    per = SUB * SUB

    counts: Dict[Tuple[int, int], int] = {}
    for py in range(size):
        for sy in range(SUB):
            gy = (py + offset + sy * step - oy) / scale + y0
            for sx0, sx1 in spans(gy):
                # Convert this ink interval back to pixel space and add its
                # subsample hits, one row of the subgrid at a time.
                a = (sx0 - x0) * scale + ox
                b = (sx1 - x0) * scale + ox
                lo = max(0, int(math.floor(a)))
                hi = min(size - 1, int(math.ceil(b)))
                for px in range(lo, hi + 1):
                    for sx in range(SUB):
                        gx = px + offset + sx * step
                        if a <= gx <= b:
                            counts[(px, py)] = counts.get((px, py), 0) + 1
    coverage = {k: v / per for k, v in counts.items() if v}
    _COVERAGE_CACHE[key] = coverage
    return coverage


def make_numbered_material(name: str, base_hex: str, values: Dict[int, int],
                           tile: int = 512, masks: Dict[int, Any] = None):
    """One material whose texture carries every face's number, in a grid atlas."""
    # 512 rather than 256: the glyph occupies a third of a tile, so the smaller
    # size left a digit only ~87px across to carry curves, and the stroke edges
    # showed even after anti-aliasing. This doubles it to ~174px.
    # Digits sit on alternate cells, with a blank cell between them.
    #
    # A face maps to more than its own tile -- measured spanning -0.36..1.36 in
    # u and 0.01..1.49 in v -- so it samples across the tile border. Packed
    # tight, that border is a neighbouring digit, and EXTEND plus Linear
    # filtering smears it onto the face: a hard black wedge appeared at the
    # vertex where four faces meet, and disconnecting the texture removed it
    # entirely. Shrinking the digit only reduced the bleed (85 stray pixels at
    # fraction 0.30, 43 at 0.24, 23 at 0.20) because the smear happens at any
    # size. A blank cell on every side means the overrun samples flat body
    # colour, which is what it should have been all along.
    used = max(1, len(values))
    span = int(math.ceil(math.sqrt(used)))
    cols = span * 2
    rows = int(math.ceil(used / span)) * 2

    img = bpy.data.images.new(name, width=tile * cols, height=tile * rows)
    # The buffer is written in the space the image is sampled in. Writing
    # scene-linear values into an sRGB image has Blender convert them a second
    # time, which shifts every colour.
    img.colorspace_settings.name = "sRGB"
    w, h = img.size
    px = [0.0] * (w * h * 4)

    def srgb_bytes(hex_color: str):
        c = hex_color.lstrip("#")
        return tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    base = srgb_bytes(base_hex)
    for i in range(w * h):
        px[i * 4 + 0] = base[0]
        px[i * 4 + 1] = base[1]
        px[i * 4 + 2] = base[2]
        px[i * 4 + 3] = 1.0

    ink = srgb_bytes(PIP_COLOR)
    for slot, (_face_index, value) in enumerate(sorted(values.items())):
        # Every other cell, in both directions.
        gx, gy = (slot % span) * 2, (slot // span) * 2
        cx, cy = gx * tile, gy * tile
        # Drawn centred with a wide margin: the UV square fits the face's
        # projection, which is narrower than the tile on steeply angled faces,
        # so ink near a tile edge could fall outside the face.
        #
        # This fraction is what sets the digit's size on the solid. Kept modest
        # so a number sits within its face with clear space around it rather
        # than crowding the edges, which on a triangular face makes it harder to
        # tell which face a digit belongs to.
        # The face's UV square is larger than the tile -- measured at 1.71x1.48
        # at the sizes used here -- so a face samples well past its own tile.
        # With EXTEND that repeats whichever pixel sits on the tile edge, which
        # is a neighbour's ink, and those bleed onto the face as dark smears.
        #
        # Drawing the digit small and centred keeps a wide margin of flat
        # background on every side, so the overflow lands on that margin rather
        # than on another digit. The margin has to be wide enough to cover the
        # overflow, which is what GLYPH_FRACTION really buys.
        inner = int(tile * GLYPH_FRACTION)
        pad = (tile - inner) // 2

        # Ink outside the face's own UV triangle is discarded rather than
        # drawn.
        #
        # A face maps to more than its tile, so with EXTEND sampling it reaches
        # past the tile border and picks up whatever sits there. Keeping the
        # digit small was not enough -- shrinking it only reduced the bleed
        # (85 stray pixels at 0.30, 43 at 0.24, 23 at 0.20) because the
        # filtering smears across the border whatever the size. Masking to the
        # triangle removes the cause instead: there is no ink outside the face
        # to be smeared in the first place. Confirmed as the cause by
        # disconnecting the texture, which took the artefact from 85 pixels to
        # none.
        tri = (masks or {}).get(_face_index, None)

        def in_triangle(u: float, v: float) -> bool:
            if tri is None:
                return True
            (ax, ay), (bx, by), (cxx, cyy) = tri
            d = lambda px, py, qx, qy, rx, ry: ((px - rx) * (qy - ry)
                                                - (qx - rx) * (py - ry))
            d1 = d(u, v, ax, ay, bx, by)
            d2 = d(u, v, bx, by, cxx, cyy)
            d3 = d(u, v, cxx, cyy, ax, ay)
            neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
            pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
            return not (neg and pos)

        # Blend by coverage rather than stamping solid pixels, so the curves
        # keep smooth edges instead of a staircase.
        for (dx, dy), a in _digit_coverage(value, inner).items():
            if not in_triangle((pad + dx) / tile, (pad + dy) / tile):
                continue
            x, y = cx + pad + dx, cy + pad + dy
            if 0 <= x < w and 0 <= y < h:
                o = (y * w + x) * 4
                for ch in range(3):
                    px[o + ch] = ink[ch] * a + px[o + ch] * (1.0 - a)

    img.pixels = px
    img.pack()

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.42
    set_input(bsdf, ("Specular IOR Level", "Specular"), 0.3)
    set_input(bsdf, ("Coat Weight", "Clearcoat"), 0.35)
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.interpolation = "Linear"
    tex.extension = "EXTEND"

    # Clamp each face's UVs to its own tile before sampling.
    #
    # A face's UV square is bigger than its tile -- 1.71x1.48 at these sizes --
    # so without this it samples across the tile border and picks up the
    # neighbouring digit, which appeared as dark smears on the side faces.
    # Clamping folds everything outside the tile back onto that tile's own flat
    # margin, so a face can only ever show its own number.
    uvmap = nodes.new("ShaderNodeUVMap")
    sep_uv = nodes.new("ShaderNodeSeparateXYZ")
    comb_uv = nodes.new("ShaderNodeCombineXYZ")
    links.new(uvmap.outputs["UV"], sep_uv.inputs["Vector"])

    # Each tile spans 1/cols by 1/rows, so clamping is done per axis against
    # the tile the face was assigned.
    for axis, count, out_sock in (("X", cols, "X"), ("Y", rows, "Y")):
        scale_up = nodes.new("ShaderNodeMath")
        scale_up.operation = "MULTIPLY"
        scale_up.inputs[1].default_value = float(count)
        links.new(sep_uv.outputs[axis], scale_up.inputs[0])

        floor_n = nodes.new("ShaderNodeMath")
        floor_n.operation = "FLOOR"
        links.new(scale_up.outputs[0], floor_n.inputs[0])

        frac = nodes.new("ShaderNodeMath")
        frac.operation = "SUBTRACT"
        links.new(scale_up.outputs[0], frac.inputs[0])
        links.new(floor_n.outputs[0], frac.inputs[1])

        clamped = nodes.new("ShaderNodeClamp")
        clamped.inputs["Min"].default_value = 0.002
        clamped.inputs["Max"].default_value = 0.998
        links.new(frac.outputs[0], clamped.inputs["Value"])

        back = nodes.new("ShaderNodeMath")
        back.operation = "ADD"
        links.new(clamped.outputs[0], back.inputs[0])
        links.new(floor_n.outputs[0], back.inputs[1])

        down = nodes.new("ShaderNodeMath")
        down.operation = "DIVIDE"
        down.inputs[1].default_value = float(count)
        links.new(back.outputs[0], down.inputs[0])
        links.new(down.outputs[0], comb_uv.inputs[out_sock])

    links.new(comb_uv.outputs["Vector"], tex.inputs["Vector"])
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])

    # Darken the ink back to black wherever the texture is dark.
    #
    # The body is a lit surface, so a face turned away from the key light
    # renders its digits mid-grey -- measured at RGB 103 on the front face,
    # against the black they are drawn in. Since reading these digits is the
    # task, they must not fade with the lighting. Mixing toward a pure-black
    # shader by the texture's own darkness keeps the ink solid on every face
    # while leaving the body itself fully lit.
    ink_shader = nodes.new("ShaderNodeEmission")
    ink_shader.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    ink_shader.inputs["Strength"].default_value = 0.0
    mix = nodes.new("ShaderNodeMixShader")
    # Fac 0 -> lit body, Fac 1 -> flat black. The texture is near-white on the
    # body and near-black on the ink, so invert it to drive the mix.
    inv = nodes.new("ShaderNodeInvert")
    links.new(tex.outputs["Color"], inv.inputs["Color"])

    # Threshold the inverted texture before it drives the mix.
    #
    # Inverting alone does not separate ink from body cleanly: the body is a
    # cream (#E8D5AC), so it inverts to roughly 0.2 rather than 0, and the
    # anti-aliased edge of a stroke lands somewhere between. Feeding those
    # middling values straight in makes the stroke partly transparent, and the
    # lit body shows through it -- which is why digits had pale specks inside
    # them. A steep ramp sends anything body-coloured to 0 and anything
    # ink-coloured to 1, while leaving a narrow band for the anti-aliased edge
    # so the glyph keeps its smooth outline.
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "LINEAR"
    ramp.color_ramp.elements[0].position = 0.45
    ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    ramp.color_ramp.elements[1].position = 0.62
    ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    links.new(inv.outputs["Color"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], mix.inputs["Fac"])
    links.new(bsdf.outputs[0], mix.inputs[1])
    links.new(ink_shader.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], out.inputs["Surface"])
    return mat, cols, rows


def unwrap_faces_to_atlas(obj, face_slots: Dict[int, int], cols: int,
                          rows: int, natural: bool = False) -> None:
    """
    Give each face its own square of the atlas.

    Two modes. By default the tile is squared to the camera, so every digit
    reads upright on screen whatever the face's roll. With `natural`, the tile
    is squared to the face's own edges instead, so a digit sits in the surface
    the way it does on a real die and tilts with it -- what the cube gets for
    free, since its pips are placed in each face's own frame and a sphere has
    no orientation to give away.

    The trade is measured, not assumed: face-aligned leaves a median tilt of
    37 degrees across the 24 orientations, and only 25% of visible faces within
    30 degrees of level. Raising the camera does not rescue it -- at 45 degrees
    the median is worse, 51 -- because these triangles never sit square to the
    view the way a cube's quads do. So `natural` buys realism and costs
    legibility, which is why it is opt-in rather than the default.

    Built by hand with `bmesh` rather than run through an unwrapping operator:
    the mapping is one triangle per tile, so there is nothing to solve, and a
    hand-built mapping cannot seam or flip the way a projected one can.

    Each face's corners are projected onto the camera's right and up axes and
    that projection is fitted into the tile. Because the tile's u runs along the
    camera's right and its v along its up, the digit lands upright on screen
    whatever the face's own roll is -- no corner has to be chosen, so there is no
    tie to break and no winding order to fall through to.
    """
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    uv_layer = bm.loops.layers.uv.verify()
    cam_right, cam_up = camera_frame()

    bm.faces.ensure_lookup_table()
    for face in bm.faces:
        slot = face_slots.get(face.index)
        if slot is None:
            continue
        # Same alternate-cell layout the atlas is painted with, so a face
        # lands on its own digit and its overrun falls on the blank cells
        # around it.
        span = max(1, cols // 2)
        cx, cy = (slot % span) * 2, (slot // span) * 2

        if natural:
            # Axes in the face's own plane, so the digit lies in the surface
            # and follows its tilt. The in-plane "up" is chosen as the
            # direction of the corner opposite the most level edge, which
            # keeps the glyph standing on that edge rather than on a vertex.
            fn = face.normal.normalized()
            loops = list(face.loops)
            # Pick the edge that looks most horizontal, measured as an angle on
            # screen rather than as a slope ratio.
            #
            # A ratio ranks a near-vertical edge as "flat" once its run is
            # tiny, which is how the front face ended up with its digit rolled
            # 60 degrees while its neighbours sat within 21. An angle has no
            # such blind spot, and the apex test keeps the glyph standing on
            # the chosen edge rather than hanging from it.
            best_axis, best_ang = None, None
            for i in range(3):
                a = loops[(i + 1) % 3].vert.co
                b = loops[(i + 2) % 3].vert.co
                d = b - a
                rise = (loops[i].vert.co - (a + b) / 2.0).dot(cam_up)
                if rise <= 1e-6:
                    continue            # apex below its baseline: upside down
                ang = abs(math.degrees(math.atan2(d.dot(cam_up),
                                                  d.dot(cam_right))))
                ang = min(ang, 180.0 - ang)       # direction, not sign
                if best_ang is None or ang < best_ang:
                    best_ang, best_axis = ang, d
            if best_axis is None or best_axis.length < 1e-9:
                axis_u = cam_right - fn * cam_right.dot(fn)
            else:
                axis_u = best_axis.copy()
            axis_u = axis_u - fn * axis_u.dot(fn)
            if axis_u.length < 1e-9:
                axis_u = cam_right - fn * cam_right.dot(fn)
            axis_u.normalize()
            axis_v = fn.cross(axis_u).normalized()
            if axis_v.dot(cam_up) < 0:
                axis_v, axis_u = -axis_v, -axis_u
        else:
            axis_u, axis_v = cam_right, cam_up

        pts = [(l.vert.co.dot(axis_u), l.vert.co.dot(axis_v))
               for l in face.loops]
        us = [p[0] for p in pts]
        vs = [p[1] for p in pts]
        span = max(max(us) - min(us), max(vs) - min(vs)) or 1.0
        # Centre on the face's centroid, not on its bounding box. A triangle's
        # box centre lies outside the triangle on the side away from its apex,
        # which slid every digit toward the base of the solid.
        mid_u = sum(us) / len(us)
        mid_v = sum(vs) / len(vs)
        # Scale so the glyph's square lands inside the projected triangle.
        #
        # Fitting the face's bounding box to the tile does not do that: a
        # triangle covers only half its box, so a square centred in the box
        # overhangs two of the three edges, and measured across all 24
        # orientations half of every visible face had its digit escaping. What
        # bounds a centred square is the triangle's inradius -- the largest
        # circle that fits -- so the mapping is scaled against that instead.
        ax, ay = pts[0]
        bx, by = pts[1]
        cxx, cyy = pts[2]
        side_a = math.hypot(bx - cxx, by - cyy)
        side_b = math.hypot(ax - cxx, ay - cyy)
        side_c = math.hypot(ax - bx, ay - by)
        perim = side_a + side_b + side_c
        area = abs((bx - ax) * (cyy - ay) - (cxx - ax) * (by - ay)) / 2.0
        inradius = (2.0 * area / perim) if perim > 1e-9 else 0.0

        # A square of side s fits inside a circle of radius r when s <= r*sqrt2.
        # The mapping is scaled so the tile's *ink area* covers that square.
        #
        # GLYPH_SCALE, not GLYPH_FRACTION, is what sets how large the digit
        # looks on a face. The two used to be the same number, which made the
        # setting inert: shrinking the glyph in the tile also shrank the region
        # of the tile a face mapped to, so the digit kept the same share of the
        # face. Separating them lets the drawn size be tuned on its own, with
        # GLYPH_FRACTION left to control only how much of the tile the ink uses
        # -- that is, its resolution.
        if inradius > 1e-9:
            fit = (GLYPH_SCALE / math.sqrt(2.0)) / inradius
        else:
            fit = 1.30 / span

        # No aspect correction here.
        #
        # Undoing each face's foreshortening was tried, to keep a circle round
        # and a square square. It fixes the two faces turned toward the camera
        # and wrecks the rest: measured across all 24 orientations it left 38%
        # of visible faces outside a 0.6-1.6 aspect, against a projection that
        # is at least consistent. A symbol on an oblique face is squashed
        # because the face is, which reads as perspective rather than as a
        # drawing error.

        for loop, (pu, pv) in zip(face.loops, pts):
            u = 0.5 + (pu - mid_u) * fit
            v = 0.5 + (pv - mid_v) * fit
            loop[uv_layer].uv = ((cx + u) / cols, (cy + v) / rows)

    bm.to_mesh(mesh)
    bm.free()


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


def _paint_cube_symbols(cube, die: Dict[str, int]) -> None:
    """
    Paint the cube's faces with symbols from the shared atlas.

    Each face gets a tile, laid out in the face's own plane with axes taken
    from the camera, so a symbol stands the same way up on all six. A cube's
    faces are square and its normals axis-aligned, so this is simpler than the
    octahedron's version -- but it is the same idea, and it has to be done the
    same way: assigning tile corners in the polygon's own winding turns each
    face's mark differently, because the six faces are wound independently of
    how the camera sees them.
    """
    values = {i: v for i, v in enumerate(
        [die["top"], die["bottom"], die["north"], die["south"],
         die["east"], die["west"]])}

    mat, cols, rows = make_numbered_material("die_body", DIE_BODY, values)
    cube.data.materials.clear()
    cube.data.materials.append(mat)

    # Which face carries which value, by its outward normal -- not by polygon
    # index, which depends on how Blender happens to wind the primitive.
    by_normal = {
        (0, 0, 1): 0, (0, 0, -1): 1, (0, 1, 0): 2,
        (0, -1, 0): 3, (1, 0, 0): 4, (-1, 0, 0): 5,
    }
    slots = {i: slot for slot, i in enumerate(sorted(values))}

    bm = bmesh.new()
    bm.from_mesh(cube.data)
    uv = bm.loops.layers.uv.verify()
    bm.faces.ensure_lookup_table()

    cam_right, cam_up = camera_frame()
    span = max(1, cols // 2)

    for poly in bm.faces:
        n = poly.normal
        key = (round(n.x), round(n.y), round(n.z))
        idx = by_normal.get(key)
        if idx is None:
            continue
        gx, gy = (slots[idx] % span) * 2, (slots[idx] // span) * 2

        # Axes along the face's own edges, not from the camera.
        #
        # Deriving them from the camera's up leaves every mark tilted against
        # the face it sits on: the projection turns each of the three visible
        # faces differently, so a mark squared to the screen is square to
        # nothing you can see. A cube's faces are square and axis-aligned, so
        # taking the axes from the edges is both exact and what a real die
        # does -- a pip is aligned to its face, not to the viewer.
        #
        # The octahedron cannot do this: its triangles have no edge that reads
        # as horizontal, which is why it squares its marks to the camera
        # instead.
        n_vec = Vector((float(key[0]), float(key[1]), float(key[2])))
        edge = None
        for loop in poly.loops:
            d = (loop.link_loop_next.vert.co - loop.vert.co)
            if d.length > 1e-6:
                edge = d.normalized()
                break
        axis_u = edge if edge is not None else Vector((1.0, 0.0, 0.0))
        axis_v = n_vec.cross(axis_u).normalized()

        # Stand the mark upright as the viewer sees it: of the face's four
        # possible quarter turns, take the one whose v points most nearly up
        # the screen. Without this each face keeps whichever edge bmesh
        # happened to list first, and the marks sit at four different angles.
        best = None
        for _ in range(4):
            score = axis_v.dot(cam_up)
            if best is None or score > best[0]:
                best = (score, axis_u.copy(), axis_v.copy())
            axis_u, axis_v = axis_v, -axis_u
        _, axis_u, axis_v = best

        # Right-handed about the outward normal, or the mark is mirrored.
        if axis_u.cross(axis_v).dot(n_vec) < 0:
            axis_u = -axis_u

        pts = [(l.vert.co.dot(axis_u), l.vert.co.dot(axis_v), l)
               for l in poly.loops]
        us = [p[0] for p in pts]
        vs = [p[1] for p in pts]

        # One scale for both axes, about the face's own centre -- the same
        # fitting the octahedron uses.
        #
        # Normalising u and v separately was tried, each against its own span,
        # with a further stretch meant to undo the projection's foreshortening.
        # That is what bent the marks: dividing by two different spans discards
        # the face's proportions outright, so a square became a rhombus however
        # the stretch was tuned. Scaling both axes by one number keeps the
        # face's shape and lets the projection do what it does to it, which is
        # what makes the octahedron's symbols read correctly.
        mid_u = sum(us) / len(us)
        mid_v = sum(vs) / len(vs)
        half = max(max(us) - min(us), max(vs) - min(vs)) / 2.0 or 1.0
        # Larger numerator, smaller mark: it widens the window of the tile the
        # face maps to, and the ink is a fixed share of that tile. 0.40 leaves
        # the mark at about 0.37 of the face.
        fit = 0.40 / half

        for pu, pv, loop in pts:
            u_ = 0.5 + (pu - mid_u) * fit
            v_ = 0.5 + (pv - mid_v) * fit
            loop[uv].uv = ((gx + u_) / cols, (gy + v_) / rows)

    bm.to_mesh(cube.data)
    bm.free()


def build_die(position: Sequence[int], die: Dict[str, int],
              symbols: bool = False) -> Any:
    """
    A real cube, marked with pips or with symbols.

    Pips are the default: spheres pressed slightly into each face rather than
    painted circles, so they catch the lighting and stay legible at the angles
    the faces are seen from. With `symbols` the faces are painted from the same
    baked outlines the octahedron uses, so a cube and an octahedron marked with
    symbols look alike.

    Only three faces can be seen at once -- that is the camera's doing, not a
    choice made here, which is exactly why the invariant is safe: no drawing
    decision can accidentally expose a fourth.
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

    if symbols:
        _paint_cube_symbols(cube, die)
        return cube

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

    # Select the mark set before anything reads it, as the octahedron does.
    # Without this the cube would ask for symbols and be handed digits, since
    # the atlas is built from whichever set this flag names.
    global USE_SYMBOLS
    want = bool(getattr(args, "symbols", False))
    if want != USE_SYMBOLS:
        USE_SYMBOLS = want
        _COVERAGE_CACHE.clear()

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

    build_die(entry["position"], entry["die"],
              symbols=bool(getattr(args, "symbols", False)))

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


def build_octahedron(cell, orientation: int, oct_, natural: bool = False) -> None:
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

    # Every face carries its number, painted into the body's own texture. All
    # eight are numbered, not just the ones facing the camera: which of them the
    # picture shows is then settled by the renderer occluding the solid, the way
    # it would for a real die, rather than by this code predicting the view.
    face_values = {fi: oct_.FACE_VALUES[fi] for fi in range(len(oct_.FACES))}
    slots = {fi: slot for slot, fi in enumerate(sorted(face_values))}

    # Built in two passes: unwrap first, then paint the atlas with each face's
    # UV triangle in hand, so the ink can be masked to the face it belongs to.
    # Painting first would mean drawing the digits blind to where the faces
    # actually land, which is what let ink spill onto a neighbour.
    placeholder = make_material("oct_body_tmp", rgba(DIE_BODY))
    obj = new_mesh_object("octahedron", verts, oct_.FACES, placeholder)
    _span = int(math.ceil(math.sqrt(max(1, len(face_values)))))
    cols = _span * 2
    rows = int(math.ceil(len(face_values) / _span)) * 2
    unwrap_faces_to_atlas(obj, slots, cols, rows, natural=natural)

    masks = {}
    uv_layer = obj.data.uv_layers.active
    for poly in obj.data.polygons:
        slot = slots.get(poly.index)
        if slot is None:
            continue
        gspan = max(1, cols // 2)
        tx, ty = (slot % gspan) * 2, (slot // gspan) * 2
        masks[poly.index] = [
            (uv_layer.data[i].uv[0] * cols - tx,
             uv_layer.data[i].uv[1] * rows - ty)
            for i in poly.loop_indices
        ]

    body_mat, cols, rows = make_numbered_material(
        "oct_body", DIE_BODY, face_values, masks=masks)
    obj.data.materials.clear()
    obj.data.materials.append(body_mat)

    # Rounded edges, as on the cube. The cube reads as a real object mostly
    # because of this: a wide, many-segment bevel carries a moving highlight
    # along every edge, which is what tells the eye the body is solid and
    # moulded rather than a shaded polygon.
    #
    # The octahedron kept a much smaller bevel out of caution -- eight triangles
    # meet at six points and a wide bevel pinches there -- but 0.018 was small
    # enough that the edges caught no light at all, so the solid stayed flat
    # while the cube next to it looked real. 0.032 is the widest that still
    # clears those vertices.
    bevel = obj.modifiers.new("Bevel", "BEVEL")
    # Wider without drawn edges: in `natural` mode the bevel is the only thing
    # parting one face from the next, so it has to carry a visible highlight
    # along every edge by itself.
    bevel.width = 0.040 if natural else 0.032
    bevel.segments = 8 if natural else 6
    bevel.limit_method = "ANGLE"
    bevel.angle_limit = math.radians(20.0)
    bevel.harden_normals = True

    # Smooth shading with an angle split, exactly as the cube does it: the
    # bevel then reads as a rounded edge instead of a band of flat facets,
    # while the faces themselves stay flat.
    for poly in obj.data.polygons:
        poly.use_smooth = True
    if hasattr(obj.data, "use_auto_smooth"):        # Blender < 4.1
        obj.data.use_auto_smooth = True
        obj.data.auto_smooth_angle = math.radians(20.0)
    else:                                            # Blender >= 4.1
        ng = bpy.data.node_groups.get("Smooth by Angle")
        if ng is not None:
            smooth = obj.modifiers.new("SmoothByAngle", "NODES")
            smooth.node_group = ng

    # Edges still get their own dark geometry on top of the bevel: adjacent
    # faces of an octahedron meet at a shallow angle, so lighting alone leaves
    # neighbouring faces nearly the same tone and their numbers appear to sit
    # on one continuous surface. The drawn edge is what keeps the faces
    # countable, which the task depends on.
    #
    # They are lit rather than emissive now, and thinner. As flat black strokes
    # they read as ink drawn over a photograph -- the one part of the solid that
    # never responded to the light. A dark, lit material still separates the
    # faces while sitting in the same scene as the body.
    #
    # In `natural` mode they are skipped entirely, leaving the die a single
    # object: one mesh, one material, its numbers in its own texture. A real
    # die has no drawn edges -- the bevel and the light do that work -- so the
    # strokes are exactly the part that gives away a diagram. The faces stay
    # countable there because the wider bevel catches a different highlight on
    # each one.
    if natural:
        return

    edge_mat = make_material("oct_die_edge", rgba(DIE_EDGE), roughness=0.5)
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
                radius=0.009, depth=d.length, location=tuple((va + vb) / 2.0),
                vertices=10)
            e = bpy.context.active_object
            e.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
            e.data.materials.clear()
            e.data.materials.append(edge_mat)
            for poly in e.data.polygons:
                poly.use_smooth = True


def render_octahedron_state(meta: Dict[str, Any], step: int, out_path: Path,
                            args: argparse.Namespace) -> None:
    """Build and render one octahedron frame."""
    oct_ = load_octahedron_module()

    # Select the mark set before anything reads it. The coverage cache is keyed
    # by (value, size) alone, so it has to be dropped when the set changes or a
    # run could paint digits from a previous frame's cache onto symbol faces.
    global USE_SYMBOLS, MIXED_SYMBOL_FACES
    want = bool(getattr(args, "symbols", False))

    # A ratio mixes the two sets; without one the die is uniformly digits or
    # uniformly symbols. The faces are chosen per puzzle rather than per frame,
    # so the marking stays a property of the die: seeded with the puzzle's own
    # seed so two runs agree and every frame of one puzzle matches.
    ratio = getattr(args, "symbol_ratio", None)
    if ratio is not None:
        seed = int(meta.get("seed", 0)) + int(getattr(args, "mix_seed", 0))
        faces = choose_mixed_faces(ratio, seed, oct_.FACE_VALUES.values())
    else:
        faces = set()

    if want != USE_SYMBOLS or faces != MIXED_SYMBOL_FACES:
        USE_SYMBOLS = want
        MIXED_SYMBOL_FACES = faces
        _COVERAGE_CACHE.clear()

    reset_scene()
    setup_world(0.35)

    trace = meta["trace"]
    board = oct_._board_near_path(trace, tuple(meta["board_radius"]))
    at = trace[step]["cell"]
    orientation = trace[step]["orientation"]

    build_octahedron_board(board, oct_)

    # Route: travelled solid, remainder dashed, lying on the board.
    #
    # It stays at board level and is instead composited over the solid further
    # down, by rendering the two in separate passes. Lifting the route above the
    # solid was tried first and does place it in front, but under a parallel
    # projection height also slides a point across the screen -- 2.67 units at
    # the height needed to clear the apex -- so the route no longer ran over the
    # cells it names, and it covered the front face's number.
    route_z = 0.02
    pts = [(t["cell"][0], t["cell"][1]) for t in trace]

    def stop_short(run, frac=0.42):
        """
        Pull a run back from the cell the solid stands on.

        The solid occupies that cell and rises out of it, so a route drawn all
        the way to the centre emerges from under the body as a stub -- with the
        run arriving from the camera's side it reads as a black block sitting
        against the die rather than as a path going beneath it. Ending the run
        partway into the last step leaves the route clearly headed for the cell
        without colliding with what is standing there.
        """
        if len(run) < 2:
            return run
        (ax, ay), (bx, by) = run[-2], run[-1]
        return list(run[:-1]) + [(ax + (bx - ax) * frac, ay + (by - ay) * frac)]

    if step > 0:
        draw_route(stop_short(pts[:step + 1]), PATH_BLACK, dashed=False,
                   z=route_z, width=0.10, name="oct_done")
    if step < len(pts) - 1:
        # The remaining run starts at the solid's cell, so trim its near end
        # for the same reason -- reversed, since it leaves rather than arrives.
        todo = list(reversed(stop_short(list(reversed(pts[step:])), frac=0.42)))
        draw_route(todo, REMAINING_PATH, dashed=True, z=route_z,
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
    # A fixed framing, not one derived from this puzzle's board.
    #
    # The board is cropped to the neighbourhood of each puzzle's path, so its
    # width varies -- measured 4.32 to 5.69 across the set, which put the
    # ortho scale between 7.12 and 8.90 and made the die noticeably larger in
    # some frames than others. Since these are meant to be compared with each
    # other, the scale is pinned to what frames the widest board in the set,
    # so a die is the same size in every picture.
    ortho = OCT_ORTHO_SCALE

    # Camera before the solid: the face labels are rolled against the camera's
    # up axis so they read upright on screen, which needs the camera in place.
    setup_camera(target, elevation_used(args), OCT_AZIMUTH, ortho)
    build_octahedron(at, orientation, oct_,
                     natural=getattr(args, "natural", False))
    setup_lighting(target, OCT_AZIMUTH, scale=max(1.0, ortho / 6.0))
    # Wide by default: the cropped lattice is a band, matching the 10x6.5 figure
    # the 2D renderer used.
    setup_render(args.width or 1500, args.height or 975,
                 args.samples, args.engine, args.transparent)

    # Two passes, composited: the scene as it stands, then the route alone drawn
    # over it.
    #
    # The route lies on the board, which is where it belongs, but the solid
    # stands more than a cell tall in this view -- so a route running toward the
    # camera passes behind it and vanishes, and one puzzle's route read as
    # scattered fragments. Raising the route instead was tried and fails
    # differently: under a parallel projection height also shifts a point
    # sideways, so it stopped lining up with its own cells and covered the front
    # face's number.
    #
    # Compositing keeps the geometry honest and settles only the occlusion,
    # which is the single thing that was wrong. It is what the 2D renderer did
    # by giving the route a higher zorder than the die.
    # The route is left where it lies, and the renderer decides what the solid
    # hides. No compositing pass.
    #
    # Two attempts to force the route in front of the body were tried and both
    # were worse than the problem. Painting the whole route over the scene drew
    # arrows straight across the faces -- the exact "route runs through the die"
    # reading. Restricting it by depth did not help either: the body is about
    # 1.8 units deep along the view, so pieces that the camera can see perfectly
    # well still count as "behind" its centre, and those were the ones landing
    # on a face's number. Adding a screen-overlap test narrowed it but not
    # enough, because a route passing the solid genuinely does overlap its
    # silhouette.
    #
    # Rendered plainly, what the solid hides is a short piece of one arrow near
    # its base, and the route stays readable across every frame -- while the
    # numbers, which are what the task asks a model to read, are never crossed.
    scene = bpy.context.scene
    scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)


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
    p.add_argument("--symbols", action="store_true",
                   help="Mark the octahedron's faces with symbols (heart, "
                        "arrow, triangle, moon, star, house, circle, square) "
                        "instead of digits. Quicker to tell apart at a "
                        "glance, and unlike a 6 and a 9 they cannot be "
                        "confused by rotation.")
    p.add_argument("--symbol-ratio", type=float, default=None,
                   metavar="R",
                   help="Mix the two markings: this fraction of the faces "
                        "carry symbols and the rest digits (0.0-1.0). Which "
                        "faces is drawn from --mix-seed, and is fixed for a "
                        "puzzle so every frame of it marks the same faces.")
    p.add_argument("--mix-seed", type=int, default=0,
                   help="Seed choosing which faces are symbols under "
                        "--symbol-ratio (default 0)")
    p.add_argument("--natural", action="store_true",
                   help="Octahedron: numbers aligned to the faces' own edges "
                        "rather than squared to the camera, no drawn edges, "
                        "and a higher camera. Looks like a real die; the "
                        "digits tilt with their faces, so they are less "
                        "uniformly legible than the default.")
    p.add_argument("--elevation", type=float, default=None,
                   help="Octahedron camera height in degrees (default 26). "
                        "Higher widens the faces angled away from the camera, "
                        "at the cost of looking down on the board; 40-45 is "
                        "the useful upper end.")
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
