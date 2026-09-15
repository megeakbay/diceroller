#!/bin/bash
# Live view of the CoT/judge run: what stage is running, per-variant counts,
# and the tail of the log. Ctrl+C stops watching; the run itself keeps going.
#
# Pass a log file to follow a specific one; with no argument the newest *.log
# in the project is used, so it keeps up when a run writes somewhere new.
cd "$(cd "$(dirname "$0")" && pwd)"
PY=~/miniconda3/bin/python3
LOG="${1:-}"

while true; do
    printf '\033[2J\033[H'
    # Pick the newest log each pass, unless one was named on the command line.
    if [ -z "$1" ]; then
        LOG=$(ls -t full_run*.log cot_run*.log judge_run*.log symbols_run*.log \
              rest_run*.log 2>/dev/null | head -1)
    fi

    echo "=== $(date +%H:%M:%S) ===   log: ${LOG:-yok}"

    # Match the worker scripts, not whatever wrapper started them.
    procs=$(ps -o command= -ax | grep -E "generate_reasoning\.py|judge_inline\.py|retry_bad\.py|build_dataset\.py" | grep -v grep)
    if [ -n "$procs" ]; then
        # Name the stage and the directory it is working on -- the counts below
        # only move while the matching stage runs, which otherwise reads as a stall.
        echo "durum: CALISIYOR"
        echo "$procs" | grep -Eo "(generate_reasoning|judge_inline|retry_bad|build_dataset)\.py --output-dir [^ ]*" \
            | sed -e 's/generate_reasoning.py --output-dir /   asama: REASONING  -> /' \
                  -e 's/judge_inline.py --output-dir /   asama: JUDGE      -> /' \
                  -e 's/retry_bad.py --output-dir /   asama: RETRY      -> /' \
                  -e 's/build_dataset.py --output-dir /   asama: DATASET    -> /' \
            | sort -u
    elif pgrep -f "run_pipeline.py|run_cot" >/dev/null; then
        echo "durum: CALISIYOR (asamalar arasi)"
    else
        echo "durum: calismiyor (bitti ya da durduruldu)"
    fi
    echo

    $PY - <<'PY'
import json, glob, os
rows = ["ds_train_cube_pin", "ds_train_cube_symbols", "ds_train_oct_digits",
        "ds_train_oct_symbols", "ds_test_cube_pin", "ds_test_cube_symbols",
        "ds_test_oct_digits", "ds_test_oct_symbols"]
print(f"{'varyant':<24}{'cot':>6}{'judge':>7}{'hedef':>7}")
tc = tj = th = 0
for d in rows:
    if not os.path.isdir(d):
        continue
    hedef = len(glob.glob(f"{d}/*/level_*/puzzle_*/metadata.json"))
    fs = glob.glob(f"{d}/*/level_*/puzzle_*/cot_reasoning.json")
    j = 0
    for f in fs:
        try:
            if any(s.get("verdict") for s in json.load(open(f)).get("steps", [])):
                j += 1
        except Exception:
            pass
    print(f"{d:<24}{len(fs):>6}{j:>7}{hedef:>7}")
    tc += len(fs); tj += j; th += hedef
print("-" * 44)
print(f"{'TOPLAM':<24}{tc:>6}{tj:>7}{th:>7}")
if th:
    print(f"\ncot %{100*tc/th:.1f}   judge %{100*tj/th:.1f}")
if os.path.exists("dataset.jsonl"):
    print(f"dataset.jsonl: {sum(1 for _ in open('dataset.jsonl'))} kayit")
PY

    echo
    echo "--- log (son 12 satir) ---"
    [ -n "$LOG" ] && tail -12 "$LOG" || echo "(log bulunamadi)"
    sleep 20
done
