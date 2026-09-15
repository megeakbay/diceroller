"""Per-directory progress for the running render, counted by file age."""
import glob
import os
import subprocess
import time

ROWS = ["ds_train_oct_digits", "ds_test_oct_digits",
        "ds_train_oct_symbols", "ds_test_oct_symbols",
        "ds_train_cube_pin", "ds_test_cube_pin",
        "ds_train_cube_symbols", "ds_test_cube_symbols"]


def run_started() -> float:
    """
    When the current run began, from the oldest live worker's elapsed time.

    Files newer than that were written by this run. Counting by image size
    instead would report everything as done, since an earlier run had already
    produced the same 1000x1000 frames.
    """
    start = None
    try:
        out = subprocess.run(["ps", "-o", "etime=,command=", "-ax"],
                             capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            if "render_blender.py" not in line and "face_nets.py" not in line:
                continue
            parts = [int(x) for x in line.split()[0].replace("-", ":").split(":")]
            secs = 0
            for x in parts:
                secs = secs * 60 + x
            began = time.time() - secs
            start = began if start is None else min(start, began)
    except Exception:
        pass
    return start if start is not None else time.time() - 3600


start = run_started()
print(f"{'varyant':<24}{'yeni':>6}{'kare':>7}{'net':>6}{'net-yeni':>10}")
tn = tt = tnn = tnt = 0
for d in ROWS:
    if not os.path.isdir(d):
        continue
    frames = [f for f in glob.glob(f"{d}/*/level_*/puzzle_*/*.png")
              if not os.path.basename(f).startswith("net")]
    nets = glob.glob(f"{d}/*/level_*/puzzle_*/net*.png")
    fnew = sum(1 for f in frames if os.path.getmtime(f) > start)
    nnew = sum(1 for f in nets if os.path.getmtime(f) > start)
    print(f"{d:<24}{fnew:>6}{len(frames):>7}{len(nets):>6}{nnew:>10}")
    tn += fnew
    tt += len(frames)
    tnt += len(nets)
    tnn += nnew
print("-" * 53)
print(f"{'TOPLAM':<24}{tn:>6}{tt:>7}{tnt:>6}{tnn:>10}")

mins = (time.time() - start) / 60
if tn and mins > 0.5:
    rate = tn / mins
    print(f"\nhiz {rate:.0f} kare/dk   gecen {mins:.0f} dk   "
          f"kalan ~{(tt - tn) / rate:.0f} dk")
