#!/bin/bash
# 1200 örnek: 4 varyant x 5 seviye x (50 train + 10 test)
#
# Her aşama yeniden çalıştırılabilir: kendi çıktısını atlar, yarıda kesilirse
# kaldığı yerden devam eder.
set -u
PY=~/miniconda3/bin/python3
cd "$(cd "$(dirname "$0")" && pwd)"   # absolute, so a copy run from elsewhere still finds the project

TRAIN=50
TEST=10
LEVELS="1 2 3 4 5"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# --- 1. puzzle üretimi -------------------------------------------------------
# Küp ve sekizyüzlü için birer havuz; dört varyant bunları paylaşır, çünkü
# aralarındaki fark yalnızca işaretleme, yol ve tahta değil.
# One call per variant, covering every level.
#
# Calling main.py once per level meant each call started with an empty set of
# already-written puzzles, so a puzzle could repeat across levels and, worse,
# across the train/test split -- 118 of 200 test puzzles also appeared in
# train. Generating all levels in a single call lets one signature set cover
# them all. The seed advances inside main.py; the numbers here only keep the
# train and test pools far apart.
gen() {
    local variant=$1 out=$2 n=$3 seed=$4 exclude=${5:-}
    local args=(--variant "$variant" --min-level 1 --max-level 5
                --instances "$n" --seed "$seed" --output-dir "$out" --no-images)
    [ -n "$exclude" ] && args+=(--exclude "$exclude")
    $PY main.py "${args[@]}" >/dev/null 2>&1
}

log "puzzle üretiliyor (train)"
gen top        pool_train  $TRAIN  1000
gen octahedron pool_train  $TRAIN  2000
# Test excludes train: the signature set that keeps one run clean dies with
# the process, so test has to be told what train already used.
log "puzzle üretiliyor (test)"
gen top        pool_test   $TEST   500000  pool_train
gen octahedron pool_test   $TEST   600000  pool_train

# --- 2. dört varyanta kopyala -----------------------------------------------
copy_pool() {
    local pool=$1 src=$2 dest=$3
    find "$pool/$src" -name "*metadata.json" 2>/dev/null | while read -r f; do
        t="$dest/${f#$pool/}"
        mkdir -p "$(dirname "$t")" && cp "$f" "$t"
    done
}

for split in train test; do
    pool="pool_$split"
    copy_pool "$pool" top        "ds/${split}_cube_pin"
    copy_pool "$pool" top        "ds/${split}_cube_symbols"
    copy_pool "$pool" octahedron "ds/${split}_oct_digits"
    copy_pool "$pool" octahedron "ds/${split}_oct_symbols"
done
log "metadata dört varyanta kopyalandı"

# --- 3. render ---------------------------------------------------------------
render() {
    local dir=$1 variant=$2; shift 2
    log "render: $dir"
    $PY render_blender.py --output-dir "$dir" --variant "$variant" \
        --engine eevee --samples 64 "$@" 2>&1 | grep -E "^Done|Error"
}

for split in train test; do
    render "ds/${split}_cube_pin"     top
    render "ds/${split}_cube_symbols" top        --symbols
    render "ds/${split}_oct_digits"   octahedron --natural
    render "ds/${split}_oct_symbols"  octahedron --natural --symbols
done

# --- 4. net görselleri -------------------------------------------------------
for split in train test; do
    $PY face_nets.py --output-dir "ds/${split}_cube_pin"     --variant top >/dev/null 2>&1
    $PY face_nets.py --output-dir "ds/${split}_cube_symbols" --variant top \
        --symbols --filename net_symbols.png >/dev/null 2>&1
    $PY face_nets.py --output-dir "ds/${split}_oct_digits"   --variant octahedron >/dev/null 2>&1
    $PY face_nets.py --output-dir "ds/${split}_oct_symbols"  --variant octahedron \
        --symbols --filename net_symbols.png >/dev/null 2>&1
done
log "net görselleri yazıldı"

log "render bitti — CoT icin: bash run_cot.sh"
