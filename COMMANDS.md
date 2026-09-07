# Üretim komutları

Blender'ın Python'unda matplotlib yok, bu yüzden her komut miniconda ile
çalıştırılır. Kısayol için:

```bash
cd ~/Desktop/diceroller
alias py=~/miniconda3/bin/python3
```

Aşağıda `py` bunu kastediyor. `~/miniconda3/bin/python3` diye açık da yazabilirsin.

---

## Dört ana kombinasyon

Her biri: metadata kopyala → render → net. Metadata'yı kopyalamak gerekiyor
çünkü renderer her puzzle'ın tahtasını ve yolunu oradan okuyor.

### 6 yüzlü + rakamlar

```bash
mkdir -p out_cube_digits
find output/top -name "*metadata.json" | while read f; do
  t="out_cube_digits/${f#output/}"; mkdir -p "$(dirname "$t")" && cp "$f" "$t"
done

py render_blender.py --output-dir out_cube_digits --variant top \
    --digits --engine eevee --samples 64

py face_nets.py --output-dir out_cube_digits --variant top
```

### 6 yüzlü + semboller

```bash
mkdir -p out_cube_symbols
find output/top -name "*metadata.json" | while read f; do
  t="out_cube_symbols/${f#output/}"; mkdir -p "$(dirname "$t")" && cp "$f" "$t"
done

py render_blender.py --output-dir out_cube_symbols --variant top \
    --symbols --engine eevee --samples 64

py face_nets.py --output-dir out_cube_symbols --variant top \
    --symbols --filename net_symbols.png
```

### 8 yüzlü + rakamlar

```bash
mkdir -p out_oct_digits
find output/octahedron -name "*metadata.json" | while read f; do
  t="out_oct_digits/${f#output/}"; mkdir -p "$(dirname "$t")" && cp "$f" "$t"
done

py render_blender.py --output-dir out_oct_digits --variant octahedron \
    --natural --engine eevee --samples 64

py face_nets.py --output-dir out_oct_digits --variant octahedron
```

### 8 yüzlü + semboller

```bash
mkdir -p out_oct_symbols
find output/octahedron -name "*metadata.json" | while read f; do
  t="out_oct_symbols/${f#output/}"; mkdir -p "$(dirname "$t")" && cp "$f" "$t"
done

py render_blender.py --output-dir out_oct_symbols --variant octahedron \
    --natural --symbols --engine eevee --samples 64

py face_nets.py --output-dir out_oct_symbols --variant octahedron \
    --symbols --filename net_symbols.png
```

---

## Karışık işaretleme

Yüzlerin bir kısmı rakam, kalanı sembol. Oran `--symbol-ratio` ile verilir
(0.0 = hep rakam, 1.0 = hep sembol).

```bash
mkdir -p output_mix
find output_natural -name "*metadata.json" -path "*octahedron*" | while read f; do
  d="output_mix/${f#output_natural/}"; mkdir -p "$(dirname "$d")" && cp "$f" "$d"
done

py render_blender.py --output-dir output_mix --variant octahedron \
    --natural --symbol-ratio 0.5 --mix-seed 7 --engine eevee --samples 64

py face_nets.py --output-dir output_mix --variant octahedron \
    --symbol-ratio 0.5 --mix-seed 7
```

Hangi yüzlerin sembol olduğu puzzle'ın kendi seed'inden çıkar, yani her puzzle
farklı bir karışım alır ama bir puzzle'ın tüm kareleri aynı kalır. **İki komutta
da aynı `--symbol-ratio` ve `--mix-seed`** verilmeli, yoksa net ile zar ayrışır.

---

## Yeni puzzle üretmek

Yukarıdakiler mevcut puzzle'ları yeniden render eder. Yeni puzzle üretmek için:

```bash
py main.py --variant octahedron --level 5 --instances 20
py main.py --variant top --level 5 --instances 20 --num-blocked 3
```

Bu 2B matplotlib görsellerini yazar; Blender ile yeniden render etmek için
yukarıdaki komutları o klasöre yönlendir.

---

## Faydalı bayraklar

| Bayrak | Ne yapar |
|---|---|
| `--puzzle DIR` | Tek bir puzzle (denemek için hızlı) |
| `--initial-only` | Sadece soru görseli |
| `--limit N` | İlk N puzzle |
| `--suffix _x` | Üstüne yazmak yerine yanına yazar |
| `--engine cycles` | Daha kaliteli, daha yavaş (varsayılan) |
| `--engine eevee` | Hızlı önizleme |
| `--samples N` | Cycles örnek sayısı |
| `--elevation N` | Sekizyüzlü kamera yüksekliği (varsayılan 26°) |
| `--natural` | Sekizyüzlü: tek parça gövde, yüze hizalı işaretler, 29.5° kamera |
| `--transparent` | Şeffaf arka plan |

Tek puzzle'da denemek:

```bash
py render_blender.py --output-dir out_oct_symbols \
    --puzzle out_oct_symbols/octahedron/level_05/puzzle_0001 \
    --natural --symbols --engine eevee --samples 48
```

---

## Görünümü değiştirmek

| Ne | Nerede |
|---|---|
| Rakam boyutu | `blender_render.py` → `GLYPH_SCALE` (ters orantılı: büyütürsen rakam küçülür) |
| Doku çözünürlüğü | `blender_render.py` → `GLYPH_FRACTION` |
| Net'teki boyut | `face_nets.py` → `draw_digit(..., size=)` |
| Font | `bake_digits.py` → `FAMILY` / `WEIGHT`, sonra `py bake_digits.py` |
| Sembol şekilleri | `bake_symbols.py`, sonra `py bake_symbols.py` |
| Gövde rengi | `blender_render.py` → `DIE_BODY` |

Font veya sembol değiştirdikten sonra **yeniden pişirmek** gerekir — renderer
canlı fonttan değil, pişirilmiş konturlardan okur:

```bash
py bake_digits.py      # digit_outlines.json
py bake_symbols.py     # symbol_outlines.json
```

---

## Testler

```bash
py test_kinematics.py
blender --background --python test_blender_scene.py
```
