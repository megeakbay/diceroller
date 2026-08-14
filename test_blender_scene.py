"""
Scene checks that must run inside Blender.

`test_kinematics.py` covers the maths; these cover the parts that only exist
once a scene is built -- geometry, camera, and the invariants the benchmark
depends on. Run:

    blender --background --python test_blender_scene.py

The important one is `check_three_visible_faces`. Showing three of the cube's
faces is a deliberate design constraint, not an accident of drawing: a fourth
would hand the model state it is supposed to carry mentally, and an earlier
version of the 2D renderer defeated the benchmark exactly that way by drawing a
six-face readout. Here the camera enforces it, so the check confirms the camera
is placed as intended rather than that someone drew the right polygons.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

import blender_render as br  # noqa: E402

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok: {name}")
    else:
        FAILURES.append(f"{name} {detail}".strip())
        print(f"  FAIL: {name} {detail}")


def camera_direction() -> Vector:
    """The direction the scene camera looks along."""
    cam = bpy.context.scene.camera
    return (cam.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))).normalized()


def check_three_visible_faces() -> None:
    """
    Exactly three of the cube's six faces may face the camera.

    Tested on the axis-aligned face normals rather than on the rendered pixels,
    since the die is axis-aligned in every state: a face is visible when its
    outward normal opposes the view direction.
    """
    print("\nthree-visible-faces invariant")
    br.reset_scene()
    br.setup_world()

    board = {"width": 7, "height": 7, "blocked": []}
    br.build_cube_board(board)
    br.build_die([3, 3], {"top": 1, "bottom": 6, "north": 2,
                          "south": 5, "east": 3, "west": 4})
    target = (3.5, 3.5, 0.0)
    br.setup_camera(target, br.CUBE_ELEVATION, br.CUBE_AZIMUTH, 8.0)

    view = camera_direction()
    normals = {
        "top": Vector((0, 0, 1)), "bottom": Vector((0, 0, -1)),
        "north": Vector((0, 1, 0)), "south": Vector((0, -1, 0)),
        "east": Vector((1, 0, 0)), "west": Vector((-1, 0, 0)),
    }
    visible = [n for n, v in normals.items() if v.dot(view) < -1e-6]
    check("exactly three faces face the camera",
          len(visible) == 3, f"got {sorted(visible)}")
    check("top is one of them", "top" in visible, f"visible={sorted(visible)}")
    # The camera looks at the board's near corner, so the two side faces seen
    # are the ones the 2D renderer drew: south and west.
    check("the two side faces are south and west",
          set(visible) == {"top", "south", "west"}, f"got {sorted(visible)}")


def check_camera_matches_2d() -> None:
    """The derived angles must reproduce the projections they came from."""
    print("\ncamera derivation")
    el, az = br._camera_from_basis(br.CUBE_BASIS)
    check("cube elevation ~29.5 deg", abs(el - 29.496) < 0.01, f"{el:.3f}")
    el2, _ = br._camera_from_basis(br.OCT_BASIS)
    check("octahedron elevation ~16.9 deg (README says ~17)",
          abs(el2 - 16.859) < 0.01, f"{el2:.3f}")
    check("both share the same azimuth", abs(az - (-135.0)) < 0.01, f"{az:.3f}")

    # A cube corner must not collapse onto another under this projection --
    # the constraint generator.py balances its basis against.
    br.reset_scene()
    br.setup_camera((0, 0, 0), br.CUBE_ELEVATION, br.CUBE_AZIMUTH, 8.0)
    view = camera_direction()
    # Build a screen basis and project the eight corners.
    up = Vector((0.0, 0.0, 1.0))
    right = view.cross(up).normalized()
    scr_up = right.cross(view).normalized()
    pts = []
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = Vector((dx, dy, dz))
                pts.append((round(w.dot(right), 4), round(w.dot(scr_up), 4)))
    check("no two cube corners project to one point",
          len(set(pts)) == 8, f"{len(set(pts))} distinct of 8")


def check_die_geometry() -> None:
    """Pips must land on the faces, and the die must stand on the board."""
    print("\ndie geometry")
    br.reset_scene()
    br.setup_world()
    die = {"top": 1, "bottom": 6, "north": 2, "south": 5, "east": 3, "west": 4}
    cube = br.build_die([2, 4], die)

    check("die object created", cube is not None)
    lo = min((cube.matrix_world @ v.co).z for v in cube.data.vertices)
    check("die rests on the board (z=0)", abs(lo) < 1e-6, f"lowest z={lo:.6f}")

    # 21 pips on a standard die: 1+2+3+4+5+6.
    pips = [o for o in bpy.data.objects if o.name.startswith("pip_")]
    check("all 21 pips present", len(pips) == 21, f"got {len(pips)}")

    # Every pip centre must sit within the cube's bounds (they are sunk into
    # the faces, so none may float outside the body).
    s = 1.0 - 2 * 0.04
    cx, cy, cz = 2 + 0.5, 4 + 0.5, s / 2.0
    outside = []
    for p in pips:
        d = p.location - Vector((cx, cy, cz))
        if max(abs(d.x), abs(d.y), abs(d.z)) > s / 2.0 + 1e-6:
            outside.append(p.name)
    check("no pip floats outside the die body", not outside, str(outside[:3]))

    check("die is bevelled", any(m.type == "BEVEL" for m in cube.modifiers))


def check_route_geometry() -> None:
    """Routes must be built, lie flat, and be emissive."""
    print("\nroute geometry")
    br.reset_scene()
    br.setup_world()
    mat = br.make_material("t", br.rgba(br.PATH_BLACK), shadeless=True)
    obj = br.add_ribbon("r", [(0, 0), (2, 0), (2, 2)], 0.14, mat, 0.02)

    check("ribbon built", obj is not None)
    zs = {round((obj.matrix_world @ v.co).z, 6) for v in obj.data.vertices}
    check("ribbon is flat and lifted off the board", zs == {0.02}, str(zs))

    nodes = mat.node_tree.nodes
    check("route material is emissive (reads over light and shadow alike)",
          any(n.type == "EMISSION" for n in nodes))

    body = br.make_material("b", br.rgba(br.DIE_BODY))
    check("die material is a Principled BSDF (it is lit, not emissive)",
          any(n.type == "BSDF_PRINCIPLED" for n in body.node_tree.nodes))


def check_colour_management() -> None:
    """The white board must stay white: no filmic/AgX tone map."""
    print("\ncolour management")
    br.reset_scene()
    br.setup_render(256, 256, 16, "cycles")
    vs = bpy.context.scene.view_settings
    check("view transform is Standard, not AgX/Filmic",
          vs.view_transform == "Standard", vs.view_transform)
    check("no creative look applied", vs.look in ("None", ""), vs.look)


def check_full_frames_render() -> None:
    """Both variants must build and render a frame end to end."""
    print("\nend-to-end render")
    import argparse
    import json
    import tempfile

    root = Path(__file__).resolve().parent / "output"
    args = argparse.Namespace(
        width=240, height=240, samples=4, engine="eevee", transparent=False,
        suffix=None, initial_only=True, cot_only=False,
    )

    for variant, renderer in (("top", br.render_cube_state),
                              ("octahedron", br.render_octahedron_state)):
        found = sorted((root / variant).glob("level_*/puzzle_*/metadata.json"))
        if not found:
            print(f"  skip {variant}: no generated puzzles")
            continue
        meta = json.loads(found[0].read_text())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "frame.png"
            try:
                renderer(meta, 0, out, args)
                ok = out.exists() and out.stat().st_size > 0
                check(f"{variant}: renders a frame", ok,
                      f"{out.stat().st_size if out.exists() else 0} bytes")
            except Exception as exc:  # noqa: BLE001
                check(f"{variant}: renders a frame", False, f"{type(exc).__name__}: {exc}")


def main() -> None:
    print("Blender scene checks")
    print("=" * 60)
    check_camera_matches_2d()
    check_three_visible_faces()
    check_die_geometry()
    check_route_geometry()
    check_colour_management()
    check_full_frames_render()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
