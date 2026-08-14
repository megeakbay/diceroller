"""
Launcher for the Blender renderer.

`blender_render.py` runs *inside* Blender and cannot import anything from a
normal interpreter, so this wrapper locates the binary and shells out to it:

    python render_blender.py --variant top --level 5 --limit 1
    python render_blender.py --puzzle output/top/level_05/puzzle_0001

Every unrecognised flag is passed straight through, so the two scripts share one
option set and cannot drift apart.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Where Blender installs itself, in order of preference. The macOS app bundle is
# first because that is the usual install there and it is not on PATH.
CANDIDATES = [
    "/Applications/Blender.app/Contents/MacOS/Blender",
    "/Applications/Blender/Blender.app/Contents/MacOS/Blender",
    "/usr/local/bin/blender",
    "/usr/bin/blender",
    "/snap/bin/blender",
    r"C:\Program Files\Blender Foundation\Blender\blender.exe",
]


def find_blender(explicit: str = None) -> str:
    """Locate the Blender binary, or exit with instructions for installing it."""
    if explicit:
        if Path(explicit).exists():
            return explicit
        sys.exit(f"No Blender at {explicit}")

    env = os.environ.get("BLENDER_PATH")
    if env and Path(env).exists():
        return env

    found = shutil.which("blender")
    if found:
        return found

    for candidate in CANDIDATES:
        if Path(candidate).exists():
            return candidate

    # Also try versioned macOS bundles, e.g. Blender 4.2.app
    for app in sorted(Path("/Applications").glob("Blender*.app"), reverse=True):
        binary = app / "Contents" / "MacOS" / "Blender"
        if binary.exists():
            return str(binary)

    sys.exit(
        "Blender not found.\n\n"
        "Install it from https://www.blender.org/download/ (or "
        "`brew install --cask blender` on macOS), then either put it on PATH "
        "or point this script at it:\n\n"
        "    python render_blender.py --blender /path/to/blender ...\n"
        "    BLENDER_PATH=/path/to/blender python render_blender.py ...\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render Rolling Dice puzzles through Blender",
        epilog="Unrecognised options are forwarded to blender_render.py; "
               "see `python render_blender.py -- --help` for the full set.",
    )
    parser.add_argument("--blender", type=str, default=None,
                        help="Path to the Blender binary")
    parser.add_argument("--gui", action="store_true",
                        help="Open Blender's window instead of rendering "
                             "headless, to inspect the scene")
    args, passthrough = parser.parse_known_args()

    blender = find_blender(args.blender)
    script = Path(__file__).resolve().parent / "blender_render.py"

    cmd = [blender]
    if not args.gui:
        cmd.append("--background")
    cmd += ["--python", str(script), "--"] + passthrough

    print(f"Blender: {blender}")
    print(f"Command: {' '.join(cmd)}\n")
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
