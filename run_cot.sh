#!/bin/bash
# Gemini CoT + judge. Render bittikten sonra ayri calistirilir.
set -u
PY=~/miniconda3/bin/python3
cd "$(cd "$(dirname "$0")" && pwd)"
# Concurrent requests. Measured throughput: 3.5x at 8, 5.4x at 16.
WORKERS=${WORKERS:-12}

log() { echo "[$(date +%H:%M:%S)] $*"; }

# --- 5. Gemini CoT + judge ---------------------------------------------------
# En pahalı adım, en sona bırakıldı: buraya gelindiğinde görsellerin tamamı
# hazır, yani bu aşama kesilse bile veri setinin geri kalanı kullanılabilir.
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
        $PY sft_pipeline/judge_inline.py --output-dir "$dir" \
            --variant "$variant" --workers "$WORKERS" 2>&1 | tail -1
    done
done

log "bitti"
