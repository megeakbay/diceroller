#!/bin/bash
# Live view of a render/net run: what is running, how much of it is done, and
# the tail of the log. Ctrl+C stops watching; the run itself keeps going.
#
# Frames are counted by age, not by size. Size stopped telling the two apart
# once an earlier run had already squared everything: what matters now is
# whether a frame was written by the run currently going, so anything newer
# than the oldest running worker counts as done.
cd "$(cd "$(dirname "$0")" && pwd)"
PY=~/miniconda3/bin/python3
LOG="${1:-}"

while true; do
    printf '\033[2J\033[H'
    if [ -z "$1" ]; then
        LOG=$(ls -t rerender_all.log full_run*.log 2>/dev/null | head -1)
    fi
    echo "=== $(date +%H:%M:%S) ===   log: ${LOG:-yok}"

    procs=$(ps -o command= -ax | grep -E "render_blender\.py|face_nets\.py" | grep -v grep)
    if [ -n "$procs" ]; then
        echo "durum: CALISIYOR"
        echo "$procs" | grep -Eo "(render_blender|face_nets)\.py --output-dir [^ ]*" \
            | sed -e 's/render_blender.py --output-dir /   RENDER -> /' \
                  -e 's/face_nets.py --output-dir /   NET    -> /' | sort -u
    else
        echo "durum: calismiyor (bitti ya da durduruldu)"
    fi
    echo

    $PY watch_render_stats.py
    echo
    echo "--- log (son 8 satir) ---"
    [ -n "$LOG" ] && tail -8 "$LOG" || echo "(log bulunamadi)"
    sleep 20
done
