"""Sanity checks for die kinematics. Run: python test_kinematics.py"""
from generator import (
    DIRECTIONS,
    DieState,
    bottom_sum,
    canonical_die,
    generate_instance,
    roll,
    simulate,
)

OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}


def test_invariants():
    """Every reachable orientation must stay a valid die."""
    die = canonical_die()
    die.validate()
    seen = set()
    frontier = [die]
    while frontier:
        d = frontier.pop()
        key = tuple(d.as_dict().values())
        if key in seen:
            continue
        seen.add(key)
        d.validate()
        for direction in DIRECTIONS:
            frontier.append(roll(d, direction))
    # A die has 24 distinct orientations in 3D
    assert len(seen) == 24, f"expected 24 orientations, got {len(seen)}"
    print(f"  ok: 24 distinct orientations, all valid (opposite faces sum to 7)")


def test_roll_is_reversible():
    """Rolling one way then back must restore the original orientation."""
    die = canonical_die()
    for direction in DIRECTIONS:
        there = roll(die, direction)
        back = roll(there, OPPOSITE[direction])
        assert back.as_dict() == die.as_dict(), f"{direction} not reversible"
    print("  ok: every roll is reversed by its opposite")


def test_four_rolls_cycle():
    """Four rolls in the same direction returns the die to its start."""
    for direction in DIRECTIONS:
        die = canonical_die()
        for _ in range(4):
            die = roll(die, direction)
        assert die.as_dict() == canonical_die().as_dict(), f"{direction}x4 != identity"
    print("  ok: four identical rolls form the identity")


def test_known_case():
    """Hand-checked: roll east from T1 B6 N2 S5 E3 W4."""
    die = roll(canonical_die(), "E")
    # west(4) comes up, top(1) goes east, east(3) goes down, bottom(6) goes west
    assert die.as_dict() == {
        "top": 4, "bottom": 3, "north": 2, "south": 5, "east": 1, "west": 6
    }, die.as_dict()
    print("  ok: single east roll matches hand calculation")


def test_perpendicular_axis_fixed():
    """Rolling E/W must not disturb the north-south axis, and vice versa."""
    die = canonical_die()
    for direction in ("E", "W"):
        r = roll(die, direction)
        assert (r.north, r.south) == (die.north, die.south)
    for direction in ("N", "S"):
        r = roll(die, direction)
        assert (r.east, r.west) == (die.east, die.west)
    print("  ok: the axis perpendicular to travel is preserved")


def test_bottom_sum_excludes_start():
    board = {"width": 5, "height": 5, "blocked": []}
    trace = simulate(board, (0, 0), canonical_die(), ["E", "E"])
    assert len(trace) == 3
    expected = trace[1]["bottom"] + trace[2]["bottom"]
    assert bottom_sum(trace) == expected
    print(f"  ok: bottom_sum counts only rolled-onto cells ({expected})")


def test_answers_consistent():
    """Regenerate each variant and confirm the stored answer is reproducible."""
    for variant in ("top", "sum", "two"):
        for seed in range(25):
            inst = generate_instance(seed=seed, variant=variant, num_rolls=4)
            if variant == "two":
                sums = inst["sums"]
                assert sums["black"] != sums["red"], "two-path answer must not tie"
                winner = "black" if sums["black"] > sums["red"] else "red"
                assert inst["answer"]["winner"] == winner
                assert inst["answer"]["total"] == max(sums.values())
            else:
                re_trace = simulate(
                    inst["board"],
                    tuple(inst["start"]),
                    DieState(**inst["initial_die"]),
                    inst["path"],
                )
                if variant == "top":
                    assert inst["answer"] == re_trace[-1]["die"]["top"]
                else:
                    assert inst["answer"] == bottom_sum(re_trace)
        print(f"  ok: '{variant}' answers reproducible over 25 seeds")


def test_blocked_cells_respected():
    for seed in range(20):
        inst = generate_instance(seed=seed, variant="top", num_rolls=4, num_blocked=3)
        blocked = [tuple(c) for c in inst["board"]["blocked"]]
        for entry in inst["trace"]:
            assert tuple(entry["position"]) not in blocked, "path crosses a blocked cell"
    print("  ok: generated paths never enter blocked cells")


def test_projection_reads_as_a_cube():
    """
    The die must project as a cube, not a flat slab.

    Two properties matter. First, vertical edges should project to roughly the
    same screen length as horizontal ones, or the die looks squashed. Second,
    no two of the cube's eight corners may land on the same screen point -- the
    near-top and far-bottom corners coincide exactly when the +z rise equals the
    ground axes' combined y-drop, which collapses the solid.
    """
    import math
    from generator import iso

    def screen_len(a, b):
        return math.hypot(b[0] - a[0], b[1] - a[1])

    origin = iso(0, 0, 0)
    horizontal = screen_len(origin, iso(1, 0, 0))
    vertical = screen_len(origin, iso(0, 0, 1))

    ratio = vertical / horizontal
    assert 0.85 <= ratio <= 1.15, (
        f"vertical edges project at {ratio:.2f}x the horizontal edge length; "
        f"outside 0.85-1.15 the die stops reading as a cube"
    )

    corners = [
        iso(dx, dy, dz)
        for dx in (0, 1)
        for dy in (0, 1)
        for dz in (0, 1)
    ]
    closest = min(
        screen_len(corners[i], corners[j])
        for i in range(len(corners))
        for j in range(i + 1, len(corners))
    )
    assert closest > 0.1 * horizontal, (
        f"two cube corners project {closest:.3f} apart (edge is {horizontal:.3f}); "
        f"the solid is collapsing into a flat slab"
    )
    print(
        f"  ok: cube projects with vertical/horizontal edge ratio {ratio:.2f}, "
        f"closest corner pair {closest / horizontal:.2f} edges apart"
    )


# ----------------------------------------------------------------------------
# Octahedron variant (eight faces, triangular grid)
# ----------------------------------------------------------------------------


def test_octahedron_rotation_group():
    """The reachable orientation set must be exactly 24 — |rotations(octahedron)|."""
    import octahedron as oct_

    assert oct_.NUM_ORIENTATIONS == 24, (
        f"expected the 24 rotations of an octahedron, got {oct_.NUM_ORIENTATIONS}"
    )
    counts = {len(v) for v in oct_.ROLL_GRAPH.values()}
    assert counts == {3}, f"every pose must offer 3 rolls, saw {counts}"
    print("  ok: octahedron reaches exactly 24 orientations, 3 rolls from each")


def test_octahedron_rolls_reversible():
    """Rolling back over the same lattice step must restore the orientation."""
    import octahedron as oct_

    for orient, rolls in oct_.ROLL_GRAPH.items():
        for step, nxt in rolls.items():
            back = (oct_._snap(-step[0]), oct_._snap(-step[1]))
            assert back in oct_.ROLL_GRAPH[nxt], (
                f"no reverse step {back} from orientation {nxt}"
            )
            assert oct_.ROLL_GRAPH[nxt][back] == orient, (
                f"rolling {step} then {back} did not return to {orient}"
            )
    print("  ok: every octahedron roll is undone by the reverse step")


def test_octahedron_opposite_faces_sum_to_nine():
    import octahedron as oct_

    for a, b in oct_.OPPOSITE.items():
        assert oct_.FACE_VALUES[a] + oct_.FACE_VALUES[b] == 9, (
            f"faces {a}/{b} sum to {oct_.FACE_VALUES[a] + oct_.FACE_VALUES[b]}, not 9"
        )
    assert sorted(oct_.FACE_VALUES.values()) == list(range(1, 9))
    print("  ok: octahedron shows 1-8 once each, opposite faces summing to 9")


def test_octahedron_position_does_not_determine_bottom():
    """
    The task is only meaningful if the solid must be tracked.

    If a cell admitted just one orientation, its bottom face would follow from
    the position alone and a model could skip the mental rotation entirely --
    which is why the tetrahedron is not offered as a variant.
    """
    import octahedron as oct_

    cells = {}
    stack = [(0, (0.0, 0.0))]
    seen = {(0, (0.0, 0.0))}
    while stack:
        orient, pos = stack.pop()
        cells.setdefault(pos, set()).add(oct_.bottom_value(orient))
        if abs(pos[0]) > 2.5 or abs(pos[1]) > 2.5:
            continue
        for step, nxt in oct_.ROLL_GRAPH[orient].items():
            q = (round(pos[0] + step[0], 2), round(pos[1] + step[1], 2))
            if (nxt, q) not in seen:
                seen.add((nxt, q))
                stack.append((nxt, q))

    per_cell = {len(v) for v in cells.values()}
    assert min(per_cell) > 1, (
        "some cell admits only one bottom face; position would give the answer away"
    )
    print(f"  ok: each octahedron cell admits {sorted(per_cell)} distinct bottom faces")


def test_octahedron_answers_reproducible():
    """A stored answer must be re-derivable by replaying its path."""
    import octahedron as oct_

    for seed in range(25):
        inst = oct_.generate_instance(seed=seed, num_rolls=5)
        replayed = oct_.replay(inst)
        stored = [t["bottom"] for t in inst["trace"]]
        assert replayed == stored, f"seed {seed}: {replayed} != {stored}"
        assert inst["answer"] == stored[-1]
    print("  ok: octahedron answers reproducible over 25 seeds")


def test_octahedron_path_stays_on_board():
    """Every cell the solid visits must exist on the generated board."""
    import octahedron as oct_

    for seed in range(15):
        inst = oct_.generate_instance(seed=seed, num_rolls=5)
        board = oct_.build_board(tuple(inst["board_radius"]))
        for entry in inst["trace"]:
            key = (round(entry["cell"][0], 2), round(entry["cell"][1], 2))
            assert key in board, f"seed {seed}: cell {key} is off the board"
    print("  ok: octahedron paths never leave the board")


def test_octahedron_resting_face_is_hidden():
    """The face resting on the board must never be listed as readable."""
    import octahedron as oct_

    for orient in range(oct_.NUM_ORIENTATIONS):
        resting = oct_.bottom_value(orient)
        assert resting not in oct_.visible_values(orient, oct_.CAMERA), (
            f"orientation {orient} exposes its resting face {resting}"
        )
    print("  ok: the octahedron's resting face is never readable")


def test_octahedron_shows_four_faces():
    """
    Every drawn face must carry its number, in every orientation.

    Four is the ceiling: an octahedron is convex, so exactly half its faces
    point away from any viewpoint. An earlier label rule also required a face
    to point upward, which silently blanked the lower-half faces and left 2-3
    of the 4 visible faces unnumbered.
    """
    import octahedron as oct_

    counts = {len(oct_.visible_values(o, oct_.CAMERA))
              for o in range(oct_.NUM_ORIENTATIONS)}
    assert counts == {4}, f"expected 4 readable faces in every pose, saw {counts}"

    for orient in range(oct_.NUM_ORIENTATIONS):
        values = oct_.visible_values(orient, oct_.CAMERA)
        assert len(set(values)) == 4, f"orientation {orient} repeats a value: {values}"
    print("  ok: all four camera-facing octahedron faces are numbered")


def test_octahedron_view_is_side_on():
    """
    The camera must look at the board from the side, not from above.

    A steep view shrinks the faces angled away from the camera until their
    numbers no longer fit; a view along the board flattens the lattice into an
    unreadable band. This pins the compromise.
    """
    import math
    import octahedron as oct_

    elevation = math.degrees(math.asin(oct_.CAMERA[2]))
    assert 12.0 <= elevation <= 25.0, (
        f"camera sits {elevation:.1f} degrees above the board; outside 12-25 "
        f"either the faces or the lattice stop reading"
    )
    print(f"  ok: camera looks from {elevation:.1f} degrees above the board")


def test_octahedron_solid_stays_on_board():
    """
    The solid's base must sit on the board in every frame.

    Only the base is checked: in this side-on view the body projects about four
    times a cell's screen height, so a die standing anywhere rises above the
    board behind it. Requiring the whole silhouette to overlap would force
    every path into the centre of a board large enough to swallow it.
    """
    import octahedron as oct_

    for seed in range(12):
        inst = oct_.generate_instance(seed=seed, num_rolls=5)
        board = oct_.build_board(tuple(inst["board_radius"]))
        for entry in inst["trace"]:
            cell = (round(entry["cell"][0], 2), round(entry["cell"][1], 2))
            assert oct_._silhouette_on_board(cell, entry["orientation"], board), (
                f"seed {seed}: solid at {cell} overhangs the board edge"
            )
    print("  ok: the octahedron's base is on the board in every frame")


def test_octahedron_route_is_always_drawn():
    """
    Every frame must show the whole route, not just the part already walked.

    Drawing only the travelled portion erased the remaining path after the
    first roll, which is the very thing the question asks the model to follow.
    """
    import octahedron as oct_

    inst = oct_.generate_instance(seed=7, num_rolls=5)
    trace = inst["trace"]
    for step in range(len(trace)):
        travelled = trace[:step + 1]
        remaining = trace[step:]
        assert len(travelled) + len(remaining) == len(trace) + 1, (
            f"step {step}: the two runs do not cover the trace"
        )
        assert travelled[-1] is remaining[0], (
            f"step {step}: travelled and remaining runs must meet at the die"
        )
    print("  ok: every frame draws the travelled and remaining route together")


def test_octahedron_board_crop_keeps_path_visible():
    """
    The drawn board is cropped to the path, so the route is not a scribble.

    Generation uses a lattice much larger than any path needs; rendering all of
    it left the route covering ~15% of the frame.
    """
    import numpy as np
    import octahedron as oct_

    for seed in range(8):
        inst = oct_.generate_instance(seed=seed, num_rolls=5)
        full = oct_.build_board(tuple(inst["board_radius"]))
        cropped = oct_._board_near_path(inst["trace"], tuple(inst["board_radius"]))
        assert len(cropped) < len(full), f"seed {seed}: crop kept the whole board"
        for entry in inst["trace"]:
            key = (round(entry["cell"][0], 2), round(entry["cell"][1], 2))
            assert key in cropped, f"seed {seed}: path cell {key} was cropped away"
            assert oct_._silhouette_on_board(key, entry["orientation"], cropped), (
                f"seed {seed}: cropping put the die's base off the board"
            )
    print("  ok: the board is cropped to the path without stranding the die")


if __name__ == "__main__":
    print("die kinematics")
    test_invariants()
    test_roll_is_reversible()
    test_four_rolls_cycle()
    test_known_case()
    test_perpendicular_axis_fixed()
    print("task semantics")
    test_bottom_sum_excludes_start()
    test_answers_consistent()
    test_blocked_cells_respected()
    print("rendering")
    test_projection_reads_as_a_cube()
    print("octahedron variant")
    test_octahedron_rotation_group()
    test_octahedron_rolls_reversible()
    test_octahedron_opposite_faces_sum_to_nine()
    test_octahedron_position_does_not_determine_bottom()
    test_octahedron_answers_reproducible()
    test_octahedron_path_stays_on_board()
    test_octahedron_resting_face_is_hidden()
    test_octahedron_shows_four_faces()
    test_octahedron_view_is_side_on()
    test_octahedron_solid_stays_on_board()
    test_octahedron_route_is_always_drawn()
    test_octahedron_board_crop_keeps_path_visible()
    print("\nall checks passed")
