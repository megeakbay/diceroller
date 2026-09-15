#!/bin/bash
# Gemini CoT only -- no judging.
#
# The judge pass is optional: generate_reasoning.py writes cot_reasoning.json,
# and judge_inline.py only adds a "verdict" field to each step. Skipping it
# halves the API calls and leaves a complete, if unvetted, dataset. Judge later
# with run_cot.sh, which skips whatever is already done.
set -u
PY=~/miniconda3/bin/python3
cd "$(cd "$(dirname "$0")" && pwd)"
WORKERS=${WORKERS:-12}

log() { echo "[$(date +%H:%M:%S)] $*"; }

for split in train test; do
    for v in cube_pin cube_symbols oct_digits oct_symbols; do
        dir="ds_${split}_${v}"
        case $v in
            cube_*) variant=top ;;
            oct_*)  variant=octahedron ;;
        esac
        log "CoT: $dir"
        $PY sft_pipeline/generate_reasoning.py --output-dir "$dir" \
            --variant "$variant" --workers "$WORKERS" 2>&1 | tail -1
    done
done

log "bitti (judge yapilmadi)"
