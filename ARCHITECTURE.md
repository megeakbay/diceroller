# Proje mimarisi

Rolling Dice: bir zar tahta üzerinde yuvarlanırken her adımı görselleştiren
veri seti üreteci. MIRA makalesindeki görevin yeniden uygulaması — bir modelin
zarın gizli yüzlerini zihinde takip edip etmediğini ölçmek için.

İki katı var: **küp** (6 yüz, MIRA'nın kendi görevi) ve **sekizyüzlü** (8 yüz,
üçgen kafeste, daha zor kardeşi).

---

## Neden bu tasarım

Her seçim bir kısıttan çıkıyor; bunları bilmeden kodu değiştirmek görevi sessizce
bozar.

**Katı seçimi zorunlu.** Bir katı, ancak taban yüzünün döşediği bir ızgarada
tutarlı yuvarlanır. Bu, çoğu Platonik katıyı eler — on iki yüzlünün beşgenleri
düzlemi döşemez. Üçgen yüzler döşer: dörtyüzlü, sekizyüzlü, yirmiyüzlü.
Dörtyüzlü kullanılamaz, çünkü dönme grubu hücre paritesine kilitleniyor: her
hücrede tek bir yönelim mümkün, yani model katıyı hiç takip etmeden konumdan
cevabı okuyabilir. Sekizyüzlü hücre başına dört farklı alt yüz bırakıyor.

**Üç görünür yüz bir tasarım kısıtı, çizim kazası değil.** Dördüncü bir yüz,
modelin zihinde taşıması gereken bilgiyi ona verir. Eski bir sürüm tahtanın
üstüne altı yüzlü bir okuma çizmiş ve cevabın doğrudan okunmasına izin vermişti;
bu kaldırıldı. 3B'de kısıtı kamera sağlıyor — küpün yakın köşesine bakan
ortografik bir kamera, üçten fazlasını gösteremez.

**Kinematik türetilmiş, elle yazılmamış.** `octahedron.py` import anında gerçek
bir sekizyüzlüyü 3B'de yuvarlıyor, fiziksel kenar etrafında döndürerek, ve çıkan
yönelim grafiğini kaydediyor. `test_kinematics.py` grubun mertebesinin 24
olduğunu doğruluyor.

---

## Dosyalar

| Dosya | Satır | Ne yapar |
|---|---|---|
| `generator.py` | 836 | Küp: kinematik, tahta/yol üretimi, 2B çizim |
| `octahedron.py` | 715 | Sekizyüzlü: aynısı, üçgen kafes üzerinde |
| `main.py` | 367 | Veri seti üreteci (2B) |
| `blender_render.py` | 2362 | 3B renderer — Blender içinde çalışır |
| `render_blender.py` | 96 | Blender'ı bulup yukarıdakini ona veren başlatıcı |
| `face_nets.py` | 423 | Zarın açınımı (`net.png`) |
| `bake_digits.py` | 83 | Fontu kontur verisine pişirir |
| `bake_symbols.py` | 167 | Sembolleri geometriden üretir |
| `test_kinematics.py` | 415 | Matematik testleri |
| `test_blender_scene.py` | 233 | Sahne testleri (Blender içinde) |

---

## Akış

```
main.py                    → output/<variant>/level_XX/puzzle_XXXX/
  ├─ generator.py             metadata.json + 2B görseller
  └─ octahedron.py

render_blender.py          → aynı klasörlere 3B görseller
  └─ blender_render.py        (metadata.json'u okur)
        ├─ digit_outlines.json
        └─ symbol_outlines.json

face_nets.py               → net.png / net_symbols.png
```

Renderer aynı `metadata.json`'u okuyup aynı dosya adlarını yazar, dolayısıyla
2B ve 3B birbirinin yerine geçebilir — rollout scriptleri, judge'lar ve
`evaluate_responses.py` hangisinin ürettiğini bilmez.

---

## İki Python

Bu, kodda birçok kararı açıklıyor:

| Ne için | Hangi Python | Neden |
|---|---|---|
| `face_nets.py`, testler, başlatıcı | miniconda 3.11 | matplotlib var |
| `blender_render.py` | Blender'ın gömülü 3.13 | `bpy` var, matplotlib **yok** |

Blender'ın Python'unda matplotlib olmadığı için:

- `octahedron.py` matplotlib'i **tembel** import eder (fonksiyon içinde), yoksa
  modül Blender'da import edilemez ve kinematik kaybolur
- Rakamlar render sırasında fonttan okunamaz → `bake_digits.py` konturları
  önceden `digit_outlines.json`'a yazar
- Ölçüm scriptleri Blender'da çalışamaz (PIL yok) → render'ı Blender alır,
  ölçümü dışarıda yaparım

---

## blender_render.py

En büyük dosya. Bölümleri:

### Kamera

Açılar tahmin edilmedi. Paralel projeksiyon, matrisinin sıfıra gönderdiği yönü
tam olarak çökertir; yani her 2B bazın ima ettiği bakış noktası o matrisin sıfır
uzayıdır:

| Sabit | Değer | Nerede |
|---|---|---|
| `CUBE_ELEVATION` | 29.5° | satır 131, bazdan türetilir |
| `OCT_ELEVATION` | 26.0° | satır 157 |
| `OCT_ORTHO_SCALE` | 8.90 | satır 172 |

`OCT_ORTHO_SCALE` sabit, puzzle başına hesaplanmıyor. Tahta her yolun etrafına
kırpıldığı için genişliği değişiyordu (ölçüm: 4.32–5.69), bu da ölçeği 7.12 ile
8.90 arasında oynatıp zarı karelerde farklı boyutta çiziyordu — karşılaştırılması
gereken görsellerde kabul edilemez.

**Bir tuzak:** `setup_camera` `rotation_euler` atar ama `matrix_world` ancak
depsgraph değerlendirince güncellenir. Sahne kurulurken kamerayı sorgulamak
birim matris döndürür ve görüşü +Z sanır. Bu yüzden yerleşim
`_CAMERA_PLACEMENT`'a kaydedilir; `view_direction()` oradan okur.

### İşaretler (rakam ve sembol)

Rakamlar gerçek fonttan (Avenir), önceden kontura pişirilmiş. Elle çizim
denendi ve bırakıldı: her rakamı elle şekillendirmek gerekiyordu ve her düzeltme
başka bir kusur getiriyordu — 6'nın kuyruğu kaseye açıyla değip kırık
görünüyordu, düzeltilince "b" gibi eğiliyordu.

Semboller ise **geometriden** üretiliyor (`bake_symbols.py`), fonttan değil:
font dingbat'ları makineden makineye değişir.

Bir sembol dönmeyle karışmaz — **6 ile 9** aynı şeklin döndürülmüşü, yuvarlanan
bir katıda bu gerçek bir risk.

| Sabit | Değer | Ne yapar |
|---|---|---|
| `GLYPH_SCALE` | 0.58 | Görünen boyut — **ters orantılı** |
| `GLYPH_FRACTION` | 0.30 | Doku çözünürlüğü |

Bu ikisi bir zamanlar tek sabitti ve ayar **işe yaramıyordu**: glifi karoda
küçültmek, yüzün eşlendiği karo bölgesini de küçültüyordu, oran sabit kalıyordu.

### Atlas ve UV

Sekiz işaret bir atlas dokusunda, her biri **atlamalı hücrelerde** — aralarında
boş karo var. Sebep ölçüldü: bir yüz kendi karosundan taşıyor (u'da −0.36…1.36),
bitişik karolar olsaydı komşu rakamı örnekler ve `EXTEND` + `Linear` filtreleme
onu yüze bulaştırırdı. Bu, dört yüzün buluştuğu köşede siyah bir kama olarak
görünüyordu; doku çıkarılınca 85 pikselden 0'a düşüyordu.

UV'ler **kameranın eksenlerine** hizalı, yüzün kenarlarına değil. Kenarlara
hizalamak gerçek bir zarın yaptığı şey ama ölçüldü: 24 yönelimde görünür
yüzlerin sadece %25'i dik bir taban sunabiliyor, medyan 37° eğik.

Küp ve sekizyüzlü **aynı fitting mantığını** kullanır: yüzün merkezine göre tek
bir ölçek. Küpte u ve v'yi ayrı ayrı normalize etmek denendi — yüzün oranını
yok ettiği için kareyi eşkenar dörtgene çeviriyordu.

### Aydınlatma

Üç ışık: alan key, güneş fill, alçak rim.

| Ayar | Değer | Neden |
|---|---|---|
| Ortam (sekizyüzlü) | 0.35 | 1.0'da yüzler 4 birim farkla aynı görünüyordu |
| Key enerji | 720 | Ortam kısılınca key şekillendirici olsun diye |
| Rim enerji | 0.15 | 1.6'da arka yüz beyaza doyuyordu |

Işıkları tek tek kapatarak teşhis edildi: key'i kapatmak ön yüzü 2 birim
değiştiriyordu, yani key katıyı hiç şekillendirmiyordu — işi ortam yapıyordu, o
da her yönden eşit geldiği için hiçbir şeyi ayıramaz.

**Tahta emisyonlu beyaz**, yani `#FFFFFF` render edilir. Ama emisyonlu yüzey
gölge alamaz, bu yüzden temas gölgesi ayrı bir decal olarak çiziliyor.

Dünya, kamera ışınlarına düz beyaz verir (Light Path düğümü), aydınlatmaya ise
gradyan — ikisi ayrı, yoksa ortamı kısmak sayfayı da karartırdı.

### Rotalar

Şeritler segment başına inşa edilir, köşelerde yuvarlak eklem. Tek bir mitre
şerit köşe noktalarını `1/cos` ile dışarı iter; üçgen kafes 60°'lik dönüşler
yapıyor, aynı yöne art arda dönünce şerit şişip kendi içine katlanıyordu.

Rota, zarın durduğu hücreye varmadan **%42'de durur**. Merkeze kadar gitmek,
okun ucunu gövdenin altından çıkarıp zara dayanmış bir blok gibi gösteriyordu.

---

## face_nets.py

Her puzzle'a bir `net.png`: zarın açınımı, hangi yüzün hangisine değdiğini
gösterir.

**Neden benchmark'ı bozmuyor:** net, zarın *yapısını* gösterir, o anki duruşunu
değil. Prompt'lar zaten karşıt yüz kuralını söylüyor, `face_layout` zaten gizli
değerleri veriyor. Net'in eklediği tek şey komşuluk — model hâlâ yuvarlanmaları
takip etmek zorunda.

Açınım **duran pozdan** üretilir: tahtadaki yüz merkezde, ona değenler etrafına
katlanır. Konumlar canlı komşuluk grafiğinden gelir, elle çizilmez.
Doğrulandı: sekiz yüz birer kez, çakışma yok, eşit alan, ve net'teki her kenar
katıda gerçek bir komşuluk.

---

## Testler

```bash
py test_kinematics.py
blender --background --python test_blender_scene.py
```

`test_kinematics.py` — ulaşılabilir yönelim kümesinin tam **24** olduğunu, her
yuvarlanmanın tersinir olduğunu, karşıt yüzlerin toplamını, ve üretilmiş her
puzzle'ın cevabının yeniden türetilebildiğini doğrular.

`test_blender_scene.py` — sadece sahne kurulunca var olanları: kamera, zar
geometrisi, ve **üç görünür yüz** değişmezi. Bu sonuncusu benchmark'ın
bağlı olduğu kısıt.

---

## Çıktı klasörleri

| Klasör | İçerik |
|---|---|
| `output/` | Ana veri seti (küp + sekizyüzlü) |
| `output_natural/` | Sekizyüzlü, `--natural` görünümü |
| `output_mix/` | Karışık işaretleme (rakam + sembol) |
| `out_oct_digits/` | 8 yüzlü + rakam |
| `out_oct_symbols/` | 8 yüzlü + sembol |
| `out_cube_symbols/` | 6 yüzlü + sembol |

Üretim komutları için `COMMANDS.md`.

---

## Kabul edilen sınırlar

Bunlar çözülmemiş sorunlar değil, ölçülüp kabul edilmiş değiş tokuşlar.

**Yan yüzlerdeki işaretler eğik.** Sekizyüzlünün kameraya eğik iki yüzü
genişliğinin ~%28'ini koruyor. Ezilmeyi telafi etmek denendi ve ölçüldü: iki
yüzü düzeltip diğerlerini bozuyor, 24 yönelimde görünür yüzlerin %38'ini
0.6–1.6 oran aralığının dışına çıkarıyordu. Düzeltilmemiş projeksiyon en azından
tutarlı — eğik yüzdeki sembol, yüz eğik olduğu için eziliyor, bu perspektif
olarak okunur.

**Sekizyüzlünün tahtası küpünkinden yayvan.** Üçgen kafes altıgen bir bölge
oluşturur, kare değil. Kamera açıları eşitlendi; kalan fark geometrinin kendisi.

**Küpte rakam yok.** Denendi ve kaldırıldı: üç görünür yüz üç farklı düzlemde
olduğu için rakamlar farklı yönlere dönüyordu. Küp nokta taşır — zaten doğrusu
bu — ve sembol taşıyabilir.
