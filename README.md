# Rolling Dice

A procedural generator for the **Rolling Dice** task from
[MIRA](https://arxiv.org/abs/2511.02779) (*When Visualizing is the First Step to
Reasoning*), built to the MentisOculi dataset conventions.

A standard die is tipped across a grid board one cell at a time. Tracking its
orientation requires maintaining a mental image of the cube through a sequence
of rotations — the paper's own example shows a model failing exactly this by
reasoning in text alone.

The paper ships no generator for the task (its 546 problems are hand-annotated),
so this reimplements it from the task descriptions, with every step rendered and
carrying its own reasoning text.

## Task variants

Three variants, matching the paper's three Rolling Dice tasks:

| Variant | Question | Answer |
|---------|----------|--------|
| `top` | What number is on top after the path? | one face value |
| `sum` | Total of the bottom faces touching the path at each step | an integer |
| `two` | Which of two equal-length paths gives the higher bottom sum? | `{winner, total}` |

Difficulty is the number of rolls: **level N = N rolls = N reasoning steps**,
following the MentisOculi convention.

## Layout

```
output/<variant>/level_XX/puzzle_XXXX/
    initial.png          question image (full path drawn)
    cot_00.png ...       one render per roll
    cot_reasoning.json   per-step reasoning + judge verdicts
    metadata.json        board, path, per-step trace, answer
```

Renders are isometric 3D, following the MIRA figures: a teal board with
receding grid lines, and the die drawn as a shaded cube on its current cell.
Three of the die's faces are visible, with the travelled path solid and the
remainder dashed.

Everything shown to a model is worded the way someone looking at the picture
would describe it. The die rolls `up-left`, `up-right`, `down-left` or
`down-right` — the board is drawn at an angle, so those are the diagonals you
actually see — and its side faces are named the same way (`up-left side`, and
so on). `top` and `bottom` keep their everyday names, since those are what the
questions ask about.

Compass points are deliberately avoided: in an isometric view "north" is not up
on the screen, so compass wording makes the model translate before it can even
start. The engine still uses N/S/E/W internally; `DIRECTION_NAMES` and
`FACE_NAMES` in [generator.py](generator.py) are the single translation point,
and their agreement with the projection is asserted rather than assumed.

Showing three faces rather than one gives enough of the cube to reason about;
the hidden three still have to be carried mentally, which is the capability the
task is meant to probe. An earlier version drew a six-face T/B/N/S/E/W readout
above the board, which let a model read the answer straight off the image and
defeated the point of the benchmark.

## Generate

```bash
uv sync
python main.py --instances 50 --min-level 1 --max-level 5
python main.py --variant top --level 3 --instances 20 --num-blocked 3
```

`--num-blocked` scatters impassable cells, which makes collisions possible and
gives the tool-use setting something to observe.

## Octahedron variant

An eight-faced die on a triangular grid, where each cell has three neighbours
instead of four:

```bash
python main.py --variant octahedron --level 5 --instances 20
```

Face values run 1-8 with opposite faces summing to 9, and the question asks
which number ends up face **down**. Chance is 1/8 rather than the cube's 1/6.

The view is deliberately side-on — the camera sits about **17 degrees** above
the board rather than looking down on it. Two measurements fix that angle: a
steeper view shrinks the faces angled away from the camera until their numbers
no longer fit, while a flatter one collapses the triangular lattice into an
unreadable band. **Four faces are visible**, and all four carry their number.
Four is the ceiling, not a shortfall: an octahedron is convex, so exactly half
its faces point away from any viewpoint — no angle shows more.

Each frame draws the **whole** route: the travelled portion solid, the rest
dashed, with the destination cell outlined. The board is cropped to the path's
neighbourhood — generation works on a much larger lattice so the die never
lands where its base would fall off an edge, but drawing all of it would leave
the route covering a sixth of the frame.

The choice of solid is forced. A solid only rolls consistently on a grid its
resting face tiles, which rules out the dodecahedron (pentagons do not tile the
plane) and leaves the triangular-faced solids. Of those, the **tetrahedron is
deliberately not offered**: its rotation group locks to the lattice's cell
parity, so exactly one orientation is reachable per cell and the bottom face
follows from the position alone — a model could answer without tracking the
solid at all. The octahedron keeps four distinct bottom faces per cell, which
`test_kinematics.py` asserts.

The kinematics are derived rather than hand-written: `octahedron.py` rolls a
real octahedron in 3D at import, rotating about the physical edge it tips over,
and records the resulting orientation graph.

The reasoning pipeline below works on this variant too — same generate → judge
loop, with the prompts restated for an eight-faced solid on a three-neighbour
lattice:

```bash
python sft_pipeline/generate_reasoning.py --output-dir output --variant octahedron
python sft_pipeline/judge_inline.py --output-dir output --variant octahedron
```

Faces are described to the model by what the picture shows — the bottom face,
the face opposite it, the visible sides, and the hidden sides — rather than by
index, since a bare `face_3` carries no spatial meaning to reason from.

The tool-use setting has an octahedron counterpart too:

```bash
python octahedron_tool_use.py --output-dir output --level 5 --limit 5
```

One thing differs from the cube's version. A cube always has four legal
directions; the octahedron rests on a triangle, so only **three** of the six
directions are possible at any moment, and which three depends on the
orientation. Every tool reply therefore lists `directions_available_now` rather
than leaving the model to infer them from a fixed enum — an enum offering all
six would invite calls that can never succeed.

Each reply also re-renders the board with the die at its current cell **and the
remainder of the route still drawn**. An earlier version drew only the current
cell; the model then lost the route after its first move and wandered, so the
setting measured nothing.

On a first run (`gemini-3.1-pro-preview`, level 5, n=5) the model scored 1/5,
and reached the wrong end cell in **all five** — including the one it answered
correctly, which was therefore luck. Rejected moves were near zero, so it is not
fighting the tool: it reads the route off the picture wrongly. That makes this
variant much harder than the cube's tool-use setting, where the rejection count
is the interesting signal.

## Blender renderer

`blender_render.py` is a drop-in alternative to the matplotlib renderers. It
reads the same `metadata.json` and writes the same `initial.png` /
`cot_NN.png` filenames into the same puzzle directories, so the rollout
scripts, the judges and `evaluate_responses.py` run against either renderer
unchanged.

```bash
python render_blender.py --variant top --level 5
python render_blender.py --variant octahedron --level 5
python render_blender.py --puzzle output/octahedron/level_05/puzzle_0001
```

`render_blender.py` locates the Blender binary and shells out to it; set
`BLENDER_PATH` or pass `--blender` if it is installed somewhere unusual. The
renderer itself only runs inside Blender:

```bash
blender --background --python blender_render.py -- --puzzle <dir>
```

Useful flags: `--suffix _blender` writes alongside the matplotlib images
instead of over them, `--engine eevee` trades some quality for speed,
`--samples` sets the Cycles sample count, and `--elevation` overrides the
octahedron camera height. `metadata.json` names `initial.png` and
`cot_NN.png`, so whichever renderer owns those filenames inside `output/` is
the one the rest of the pipeline reads.

### What the port has to preserve

These are benchmark stimuli, not illustration, so three properties are
load-bearing and the 3D version keeps each one:

- **Exactly three cube faces visible.** An orthographic camera on the board's
  near corner gives this by construction — the three are `top`, `south` and
  `west`, the same three `generator._draw_die_on_cell` drew. The 2D version got
  it by drawing only three polygons, so a fourth could have crept in through a
  drawing change. Orthographic matters on its own too: under perspective the
  same face would read differently depending on which cell the die stood on.
- **Values stay readable.** This is what the task asks a model to recover, so
  it constrains more of the renderer than anything else — see below.
- **Travelled path solid, remainder dashed.** Dashes are walked along the whole
  polyline rather than restarted per segment, so the rhythm stays continuous
  through corners.

`test_blender_scene.py` checks what only exists once a scene is built — the
camera, the die geometry, and the three-visible-faces invariant:

```bash
blender --background --python test_blender_scene.py
```

### Camera

The cube's camera is not tuned by eye. A parallel projection collapses exactly
the direction its matrix sends to zero, so the viewpoint each matplotlib basis
implies is that matrix's null space: **29.5 degrees** of elevation for the
cube. `--check-angles` prints it and runs outside Blender.

The octahedron is rendered from **26 degrees**, above the 16.86 its 2D basis
encodes. That angle was chosen for line work; in a shaded render it is too
shallow, and a lattice cell projects only 0.25 as tall as it is wide, so the
board collapses toward a band. At 26 degrees it projects 0.38 while the
smallest camera-facing face loses almost no area. The set of visible faces is
unchanged, verified step by step against `octahedron.visible_values`.

Raising it further is a real trade: the two faces angled away from the camera
keep only about 28% of their width at 26 degrees, and `--elevation 42` widens
them at the cost of looking down on the board rather than across it.

### Numbers on the octahedron

The cube carries its values as spheres sunk into the faces, and never had a
placement problem — a sphere looks the same from every angle. Numerals do not:
they have a top and a reading direction, and an octahedron's faces tilt in
three axes at once.

Placing each digit as its own 3D object meant solving orientation, handedness
and foreshortening by hand, and every variant left some face mirrored, rolled
or squashed to an unreadable sliver. A mirrored 5 reads as a 2, which would
silently corrupt the value the picture is supposed to show.

The digits are now **painted into the body's texture**: drawn once into an
atlas, one tile per face, and mapped on. A number cannot detach, mirror or roll
because it is part of the surface. The UV square is aligned to the camera's
axes rather than the face's edges — aligning to the edges is what a real die
does, and it was measured first: across all 24 orientations only 25% of visible
faces can present a level baseline that way, with a median 37 degrees off
level. Since reading the digits *is* the task, they are squared to the viewer.
Measured after the change, 96/96 visible faces land within a fraction of a
degree of upright.

The digit shapes are stroke paths rasterised into the atlas rather than text
rendered from a font file, so the result does not depend on which fonts a
machine happens to have — Blender's bundled font differs between builds. Each
digit is a set of independent strokes: chaining a bowl and its stem into one
polyline draws a connecting segment across the glyph, which is what once turned
the 6 into something that read as a `d`.

The glyph is sized against the **inradius** of the face's projected triangle,
not against its bounding box. A triangle covers only half its box, so a square
centred in the box overhangs two of the three edges — measured across all 24
orientations, half of every visible face had its digit escaping that way.

Ink is mixed toward a flat black shader by the texture's own darkness. The body
is lit, so digits on a shaded face rendered mid-grey (measured RGB 103) despite
being drawn black.

### Lighting

A three-point rig: an area key for a penumbral contact shadow, a broad sun fill,
and a low rim. The shading is the point of the port — a literal translation of
the 2D drawing would waste the renderer:

- **Contact shadow** tightening at the base is the strongest cue that the solid
  stands *on* the board rather than floating over it. The matplotlib version had
  none.
- **Bevelled edges** with hardened normals catch a highlight along every corner,
  so adjacent faces separate even where their tones are close. The 2D renderer
  faked that with three hardcoded shade factors.
- **A clear coat** gives the body the tight highlight of moulded plastic rather
  than a flat matte fill.
- **A graded environment**, brighter overhead than at the horizon, so faces at
  different angles differ even out of the key light.

The rim light is kept very low (0.15). A rim is aimed back at the camera, so a
face turned away from the viewer meets it almost head-on and clips long before
the faces being read do — at 1.6 the octahedron's rear-upper face rendered pure
white and lost both its colour and its number.

Two things deliberately stay flat. Grid rules and the start marker are
**emissive**: they annotate the scene rather than inhabit it, so they must read
identically over the lit and shadowed halves of the board. And the view
transform is forced to **Standard**, not Blender's default AgX, which would
desaturate the palette and turn the white board grey.

The board surface is emissive white so it renders exactly `#FFFFFF` everywhere,
with no shading falloff across it. An emissive surface cannot receive a shadow,
so the contact shadow is painted as its own soft decal — the floor stays white
except directly under the solid, which is the one place a shadow carries
information.

### Routes

Routes are flat ribbons built one segment at a time, with a round joint at each
corner. A single mitred strip is tidier but pushes its corner vertices out by
`1/cos(half-angle)`, and the triangular lattice turns 60 degrees at a time, so
a route that keeps turning the same way inflates until it folds through itself
and reads as broken.

On the octahedron the route is rendered as a **second pass and composited over
the scene**, but only on frames where the solid actually hides part of it — a
route point is behind the solid when it is further from the camera than the
body. Compositing unconditionally fixes the disappearing run and then creates a
worse problem: on frames where the route passes in front there was nothing to
fix, and the overlay draws an arrow straight across the body, which reads as
the route going through the die. The solid stands more than a cell tall in this view, so a route
running toward the camera passes behind it and disappears — routes running away
from the camera were fine, which is why only some puzzles looked broken.
Raising the route in 3D was tried and fails differently: under a parallel
projection height also shifts a point sideways, measured at 2.67 units for the
height needed to clear the apex, so it no longer lined up with the cells it
names. Compositing settles the occlusion and nothing else, which is what the 2D
renderer did by giving the route a higher zorder than the die.

The composite deliberately does not paint over dark pixels. The route must win
against the body or it breaks up; it must not win against the numbers, since a
dash laid across a digit can change what it reads as.

Geometry and kinematics are imported from `octahedron.py` rather than restated:
that module derives its orientation graph by rolling a real solid at import, and
a second copy here could drift from the ground truth the dataset was generated
against.

## Kinematics

The engine carries the die as an explicit six-face state; rolling permutes the
faces about the leading edge. `test_kinematics.py` verifies the invariants that
make the ground truth trustworthy:

```bash
python test_kinematics.py
```

It checks that the reachable orientation set is exactly **24** (the correct
count for a cube), that every roll is undone by its opposite, that four rolls in
one direction are the identity, that the axis perpendicular to travel is
preserved, and that every generated puzzle's stored answer is reproducible.

For the octahedron it checks that the reachable orientation set is exactly 24
(the rotation group of an octahedron), that every roll is reversed by the
opposite lattice step, that face values are 1-8 with opposites summing to 9,
that no cell admits a single bottom face, that generated paths stay on the
board, and that the resting face is never drawn as readable.

It also checks the cube projection itself: the die's vertical edges must project to
within 15% of its horizontal edge length, and no two of the cube's eight corners
may land on the same screen point. Those two constraints pull against each other
— the collapse happens exactly when the vertical rise equals the ground axes'
combined drop — so the test pins both rather than leaving the balance to a
comment.

## Setup

The model-calling scripts use Gemini via `google-genai`. Put your key in a
`.env` file in the project root (gitignored — see `.env.example`):

```
GEMINI_API_KEY=your-key-here
```

Every script loads it automatically by walking up from its own directory, the
same way the form-board pipeline does. An exported `GEMINI_API_KEY` takes
precedence over the file.

## Tool use

`tool_use_rollout.py` exposes the movements as Gemini function declarations
rather than asking the model to track the cube in its head:

- `roll_die(direction)` — tips the die one cell and returns the new six-face
  state, followed by **a freshly rendered image** of the board
- `observe()` — re-inspects the board without moving

Illegal rolls (off the board, into a blocked cell) come back with
`rejected: true` and an explicit `REJECTED: ...` message, the die left in place,
so the model observes the collision and re-plans instead of silently proceeding
from a corrupted state.

```bash
python tool_use_rollout.py --output-dir output --variant top --limit 20
python tool_use_rollout.py --puzzle output/top/level_03/puzzle_0001
```

Results record the predicted answer, tool-call count, and how many moves were
rejected — the rejection count is a direct measure of how well the model reads
the path off the image.

## Text-only baseline

`text_only_rollout.py` serializes the board, the die's three visible faces, and
the roll sequence to plain text, so a model with no vision can attempt the task
zero-shot:

```bash
python text_only_rollout.py --output-dir output --variant top --level 5
python text_only_rollout.py --model gemini-2.5-flash --dry-run
```

Only the three faces a viewer of the render would see are stated; the hidden
three still have to be inferred from the sum-to-7 rule and carried through the
rolls. `--dry-run` prints the prompts without calling a model.

This setting removes the perception half of the task — reading the path off an
isometric picture — so **its numbers are not comparable to the image-based
settings**. It measures the cube-tracking half alone, and so upper-bounds what a
text model can do once perception is free. `--thinking` leaves the reasoning
budget at the model's default; without it the budget is zeroed, which
thinking-only models reject outright rather than silently ignoring.

`claude_rollout.py` is the same baseline on Claude, importing the serialization
and answer parsing from `text_only_rollout.py` so the two differ only in which
model is called:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=your-key-here   # or add it to .env
python claude_rollout.py --output-dir output --variant top --level 5
python claude_rollout.py --effort low --no-thinking
```

`--effort` sets reasoning depth (`low` … `max`); `--no-thinking` is the true
zero-shot condition and is only valid at `high` or below, since Claude Opus 5
rejects disabled thinking above that.

### Checking the reasoning, not just the answer

A correct final number does not prove the cube was tracked correctly — a model
can make compensating errors, or guess from six options. `verify_steps.py`
reads whatever working a model volunteered and checks all six faces at every
step against the exact kinematics:

```bash
python verify_steps.py --results text_only_2p5flash_think.json
python verify_steps.py --results claude_opus5.json --show-bad
```

The prompt deliberately does **not** ask for a trace — asking would supply the
solution strategy and stop the setting being zero-shot. Both runners instead
pull whatever reasoning the API will return (`include_thoughts` on Gemini,
`display: "summarized"` on Claude) into a separate `reasoning` field, which
changes nothing about what the model was asked to do. Verification reads that
field alongside the visible answer.

The parser is therefore format-agnostic, accepting both named pairs
(`T=5 B=2 ...`) and positional tuples declared against a face order. Each
response lands in one of five states:

| Verdict | Meaning |
|---------|---------|
| `sound` | every stated state is on the true trajectory, steps 0..N all present |
| `UNSOUND` | a stated state is off the true trajectory — real reasoning error |
| `partial` | states shown are all correct, but the derivation is incomplete |
| `no trace` | answered without showing work |
| `narrates but states no values` | a reasoning summary describing the process without ever giving face values |

Only `UNSOUND` is evidence of faulty reasoning. The last three mean the check
has nothing to say — they are limits of the evidence, not failures of the model.
Gemini's thought summaries fall in the last category: they narrate ("I worked
through the five rolls") without stating intermediate faces, so a model that
keeps its visible answer terse cannot be verified in this setting at all.

`dump_traces.py` unpacks a run into readable files, one directory per puzzle,
each holding the prompt, the reasoning, the answer, and a step-by-step verdict
laying the model's stated states next to ground truth:

```bash
python dump_traces.py --results text_only_2p5flash_think.json
cat traces/text_only_2p5flash_think/summary.txt
```

**Results are not deterministic.** Re-running the same model on the same
puzzles changes which ones it gets wrong — the no-thinking condition scored 60%
in two separate runs, but failed a different pair each time. Treat any single
run at n=5 as a wide interval, not a measurement.

## Reasoning pipeline

The generate → judge → retry loop from the form-board pipeline, applied per
step:

```bash
python sft_pipeline/run_pipeline.py --output-dir output
```

1. **`generate_reasoning.py`** — writes a reasoning text for every roll, shown
   the before/after boards and the true faces, and instructed to reason forward
   from the rule rather than from the answer it was given.
2. **`judge_inline.py`** — a step passes only if its stated faces match ground
   truth (checked in code — the kinematics are exact, so there is no reason to
   ask a model), it contains no phrase revealing the answer was supplied, and a
   judge rates the explanation sound.
3. **`retry_bad.py`** — regenerates failures at higher effort, feeding the
   previous attempt's failure reason back in so a retry is not a blind resample.
4. **`build_dataset.py`** — collects verified chains into `dataset.jsonl`.

Every stage is idempotent, so the pipeline can be re-run or resumed.

## Evaluate

```bash
python evaluate_responses.py --responses results.json --dataset output
```

Parses fenced JSON, `{"answer": ...}` objects, and `<answer></answer>` tags,
and reports accuracy per variant and level against chance.

## Prompts

`prompts/` holds the four strategies used by MentisOculi — `simple`,
`tool_use`, `visual_cot`, and `generate_images` — each with the die-rolling rule
stated explicitly and the paper's own thinking-prompt wording for the CoT
variants.
