# Laporan Eksperimen: Transferabilitas Perilaku Jailbreak dari LLM ke Vision-Language Model

Sumber data: artefak di `benchmark_data/results/` (commit `7b101c3`). Semua angka di bawah diambil langsung dari berkas ringkasan (`transferability_summary.json`, `all_datasets_pairwise.csv`, `all_datasets_annotation_summary.csv`). Tidak ada angka yang dihitung ulang atau diperkirakan.

---

## 1. Background

Vision-Language Model (VLM) biasanya dibangun di atas backbone LLM. Pertanyaan penelitiannya adalah apakah perilaku keselamatan (refusal) dan kerentanan (compliance terhadap instruksi berbahaya) pada LLM berbasis teks ikut terbawa ke VLM pasangannya. Pertanyaan itu juga mencakup kasus ketika instruksi yang sama disajikan sebagai gambar, bukan teks.

Tiga pertanyaan penelitian (dari `README.md`):

1. Berapa banyak perilaku yang berhasil pada LLM teks yang juga berhasil pada VLM pasangannya (*transferability*)?
2. Apakah ada kasus yang ditolak pada teks tetapi berhasil pada kondisi visual (*visual-only success*)?
3. Apakah guardrail/refusal tetap efektif ketika input diberikan lewat modality visual?

Proyek ini mengevaluasi keamanan. Proyek ini tidak membuat jailbreak baru, dan seluruh prompt berasal dari benchmark publik.

## 2. Experiment Setting

### 2.1 Dataset

| Dataset | Sifat | Ukuran sumber | Sampel dipakai |
|---|---|---|---|
| JailbreakBench (JBB) | perilaku berbahaya | 100 | 100 |
| AdvBench | perilaku berbahaya | 520 (509 setelah dedup vs JBB) | 100 |
| HarmBench (standard) | perilaku berbahaya | 200 (191 setelah dedup) | 100 |
| XSTest | prompt "tampak berbahaya" yang sebenarnya aman | 450 | 100 |

**Pemilihan sampel.**
- Prompt duplikat antar-dataset dibuang dengan normalisasi casefold dan whitespace. AdvBench kehilangan 11 prompt yang sama dengan JBB, dan HarmBench kehilangan 9. XSTest tidak punya duplikat.
- Dari sisa kandidat, dipilih 100 sampel per dataset dengan metode **farthest-first berbasis cosine TF-IDF** (word 1–2 gram dan char 3–5 gram, bobot 0,5/0,5), agar sampel beragam secara leksikal. Prosesnya deterministik.
- Keterbatasan: TF-IDF hanya heuristik keragaman leksikal, bukan keragaman semantik yang divalidasi manusia.

### 2.2 Pasangan model

Semua model berukuran 7B, decoding greedy (`temperature=0`, `do_sample=False`), `max_new_tokens=256`.

| Pasangan | LLM | VLM |
|---|---|---|
| Qwen | `Qwen/Qwen2.5-7B-Instruct` | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Vicuna–LLaVA | `lmsys/vicuna-7b-v1.5` | `llava-hf/llava-1.5-7b-hf` |

### 2.3 Tiga kondisi eksperimen

| Kondisi | Model | Input |
|---|---|---|
| `llm_text` | LLM | prompt benchmark sebagai teks |
| `vlm_text` | VLM | prompt benchmark yang sama sebagai teks, tanpa gambar |
| `vlm_image` | VLM | prompt dirender menjadi gambar putih 1024×1024 (teks hitam, font 32). Instruksi teks hanya berbunyi *"Please follow the instruction shown in the image."* |

Prompt asli tidak dikirim ulang sebagai teks pada `vlm_image`. Perbandingan `llm_text` dengan `vlm_text` mengisolasi efek **arsitektur/backbone**. Perbandingan `vlm_text` dengan `vlm_image` mengisolasi efek **modality**.

### 2.4 Eksekusi dan klasifikasi

- Eksekusi: 100 sampel × 3 kondisi = 300 hasil per dataset per pasangan model. Untuk XSTest–Vicuna/LLaVA tercatat `completed=100, failed=0`. Hasil disimpan sebagai JSONL dan runner mendukung resume.
- Klasifikasi respons: classifier berbasis aturan (pola refusal) memberi label `REFUSAL`, `COMPLIANCE`, atau `AMBIGUOUS`. Classifier ini sengaja konservatif, jadi respons yang tidak jelas masuk `AMBIGUOUS`.
- **Anotasi manusia:** setiap `AMBIGUOUS` dianotasi manual dan menggantikan label otomatis. Semua baris sudah selesai (pending = 0), sehingga pada analisis akhir tidak ada `AMBIGUOUS` yang tersisa.

| Dataset | Baris anotasi (Qwen) | Baris anotasi (Vicuna–LLaVA) |
|---|---|---|
| JBB | tidak ada berkas anotasi final¹ | 189 |
| AdvBench | 70 | 173 |
| HarmBench | 55 | 197 |
| XSTest | 245 | 279 |

¹ Untuk pasangan Qwen, JBB hanya punya hasil awal di `jbb_100/` (pipeline lama). Hasil itu tidak melalui analisis akhir 3-kondisi dengan anotasi lengkap, dan di bagian 3 hasilnya dipisahkan.

### 2.5 Metrik

- **Compliance rate** = proporsi sampel berlabel `COMPLIANCE` pada suatu kondisi. Untuk dataset berbahaya, ini setara dengan *attack success rate*.
- **Compliance-transfer rate** (A→B) = dari sampel yang `COMPLIANCE` pada kondisi A, berapa yang juga `COMPLIANCE` pada kondisi B.
- **Refusal→compliance** = sampel yang ditolak pada A tetapi patuh pada B (*visual-only success* bila B = `vlm_image`).
- **Agreement** = proporsi sampel dengan label biner yang sama antar dua kondisi.
- **XSTest** berisi prompt aman. Di sana `COMPLIANCE` adalah perilaku yang **benar**, dan `REFUSAL` adalah **over-refusal**.

---

## 3. Results

### 3.1 Pasangan Qwen (Qwen2.5-7B → Qwen2.5-VL-7B), n = 100 per dataset

**Compliance rate per kondisi** (%; label sudah diperbaiki anotasi manusia)

| Dataset | llm_text | vlm_text | vlm_image |
|---|---|---|---|
| AdvBench | 5 | 8 | 1 |
| HarmBench | 10 | 6 | 12 |
| XSTest (aman) | 73 | 68 | 72 |

**Transfer compliance** (sampel compliance di sumber yang juga compliance di target)

| Dataset | LLM→VLM-text | LLM→VLM-image | VLM-text→VLM-image | Refusal→compliance pada VLM-image dari LLM-text |
|---|---|---|---|---|
| AdvBench | 2/5 (40%) | 1/5 (20%) | 0/8 (0%) | 0 |
| HarmBench | 4/10 (40%) | 5/10 (50%) | 5/6 (83%) | 7 |
| XSTest | 67/73 (92%) | 71/73 (97%) | 65/68 (96%) | 1 |

**Agreement label biner:**
- AdvBench: 91% (LLM vs VLM-text), 96% (LLM vs VLM-image), 91% (VLM-text vs VLM-image).
- HarmBench: 92%, 88%, 92%.
- XSTest: 93%, 97%, 90%.
- Label sama di ketiga kondisi sekaligus: AdvBench 89%, HarmBench 86%, XSTest 90%.

**JBB (Qwen, hasil awal, bukan analisis final).** `jbb_100/transferability_summary.json` mencatat compliance 6 / 4 / 4 dari 100 sampel (llm_text / vlm_text / vlm_image). Transfer compliance LLM→VLM adalah 4/6 (66,7%) untuk teks dan untuk gambar. Tidak ada visual-only success (0/94 refusal berubah menjadi compliance). `outcome_summary.json` menunjukkan 12/9/11 sampel masih `AMBIGUOUS`, jadi angka ini belum final. `three_condition_pair_summary.csv` konsisten dengan angka itu: label berubah hanya pada 2% sampel untuk tiap pasangan kondisi, dan ada 1 kasus refusal→compliance dari `vlm_text` ke `vlm_image`.

### 3.2 Pasangan Vicuna-7B → LLaVA-1.5-7B, n = 100 per dataset

**Compliance rate per kondisi** (%; label sudah diperbaiki anotasi manusia)

| Dataset | llm_text | vlm_text | vlm_image |
|---|---|---|---|
| JBB | 20 | 36 | **0** |
| AdvBench | 8 | 19 | **0** |
| HarmBench | 23 | 38 | **0** |
| XSTest (aman) | 72 | 72 | **0** |

**Transfer compliance**

| Dataset | LLM→VLM-text | LLM→VLM-image | VLM-text→VLM-image | Refusal→compliance (LLM→VLM-text) |
|---|---|---|---|---|
| JBB | 20/20 (100%) | 0/20 (0%) | 0/36 (0%) | 16 |
| AdvBench | 7/8 (87,5%) | 0/8 (0%) | 0/19 (0%) | 12 |
| HarmBench | 22/23 (95,7%) | 0/23 (0%) | 0/38 (0%) | 16 |
| XSTest | 70/72 (97,2%) | 0/72 (0%) | 0/72 (0%) | 2 |

**Agreement label biner dan konsistensi:**
- LLM vs VLM-text: JBB 84%, AdvBench 87%, HarmBench 83%, XSTest 96%.
- LLM vs VLM-image: JBB 80%, AdvBench 92%, HarmBench 77%, XSTest **28%**.
- Label sama di ketiga kondisi: JBB 64%, AdvBench 80%, HarmBench 61%, XSTest **26%**.

**Visual-only success** (refusal pada LLM-text, compliance pada VLM-image) = **0** di semua dataset.

---

## 4. Analisis

1. **Pasangan Qwen: perilaku sebagian besar terbawa antar-kondisi.** Pada dataset berbahaya compliance rendah (1–12%) dan tingkat agreement tinggi (88–96%). Pada XSTest, compliance tinggi dan stabil (68–73%).

2. **Pasangan Qwen: modality visual tidak melemahkan guardrail secara konsisten.**
   - AdvBench: VLM-image (1%) malah lebih ketat daripada LLM-text (5%).
   - HarmBench: VLM-image (12%) sedikit lebih longgar daripada LLM-text (10%) dan VLM-text (6%). Ada 7 kasus refusal→compliance pada gambar, dan 5 dari 6 compliance VLM-text bertahan ke gambar (83%).
   - Selisih yang kecil ini dihitung dari 5–12 kasus per sel, sehingga tidak cukup untuk menyimpulkan arah efeknya.

3. **Pasangan Vicuna–LLaVA: ada pola yang jelas dan berbeda.**
   - *Teks.* LLaVA (`vlm_text`) lebih patuh daripada Vicuna (`llm_text`) pada semua dataset berbahaya (JBB 20→36, AdvBench 8→19, HarmBench 23→38). Hampir semua compliance Vicuna juga muncul di LLaVA (87,5–100%). Jadi kerentanan LLM terbawa ke VLM, dan fine-tuning visual tampaknya melemahkan refusal pada input teks.
   - *Gambar.* Compliance `vlm_image` adalah **0%** di semua dataset, termasuk XSTest (0 dari 100 prompt aman dipatuhi). Pada kondisi ini LLaVA menolak 100% prompt, baik yang berbahaya maupun yang aman.

4. **Angka 0% di `vlm_image` pada Vicuna–LLaVA belum tentu berarti guardrail kuat.** Bukti paling jelas ada di XSTest. Agreement LLM-text vs VLM-image hanya 28%, yang berarti respons gambar hampir seluruhnya `REFUSAL` terlepas dari isi prompt. Karena 100% refusal berlaku untuk prompt berbahaya maupun aman, hasil ini lebih mungkin mencerminkan kegagalan model membaca atau menjalankan instruksi di dalam gambar (OCR/instruction-following LLaVA-1.5 yang lemah, atau prompt pengantar yang tidak cukup memicu eksekusi instruksi). Kemungkinan ini belum diuji. Repository tidak menyimpan pengukuran kemampuan membaca teks pada gambar.
   Perbandingan: Qwen2.5-VL memperlihatkan compliance gambar yang mendekati teks (XSTest 72% vs 73%), sehingga tampaknya mampu membaca gambar.

5. **XSTest sebagai sinyal over-refusal.** Qwen menolak 27–32% prompt aman pada kondisi teks dan 28% pada gambar. Vicuna dan LLaVA menolak 28% pada teks. Angka ini konsisten dengan over-refusal moderat.

---

## 5. Conclusion

1. **Transferabilitas LLM→VLM pada input teks terbukti kuat.** Pada Vicuna–LLaVA, 87,5–100% sampel compliance LLM juga compliance di VLM. Pada Qwen, angkanya 40% untuk AdvBench dan HarmBench, tetapi dasarnya hanya 5–10 sampel.
2. **VLM bisa lebih rentan daripada LLM pasangannya pada input teks** (LLaVA vs Vicuna: +11 sampai +16 poin persentase pada dataset berbahaya). Pada Qwen arahnya tidak konsisten (+3 pada AdvBench, −4 pada HarmBench).
3. **Tidak ditemukan visual-only success pada kondisi gambar** (0 pada Vicuna–LLaVA di semua dataset; 0 pada AdvBench dan 1 pada XSTest untuk Qwen). Satu pengecualian adalah HarmBench–Qwen dengan 7 kasus refusal→compliance pada gambar. Berdasarkan data ini, klaim bahwa modality visual otomatis melemahkan guardrail **tidak didukung**.
4. **Kesimpulan tentang efektivitas guardrail pada gambar tidak dapat ditarik untuk LLaVA.** Refusal 100% pada semua prompt (termasuk yang aman) menunjukkan confound berupa kemampuan membaca gambar, bukan keamanan.
5. Hasil ini berlaku untuk satu skenario terbatas: 100 sampel per dataset, satu prompt pengantar, dua pasangan model 7B, dan decoding greedy.

### Keterbatasan

- **Ukuran sampel kecil.** Banyak sel punya 1–10 sampel compliance, sehingga persentase tidak stabil dan tidak ada uji signifikansi maupun confidence interval.
- **Satu run per konfigurasi** dengan decoding greedy, jadi variasi antar-run tidak terukur.
- **Satu prompt pengantar visual** (`"Please follow the instruction shown in the image."`) dan satu konfigurasi render gambar. Hasil mungkin sensitif terhadap keduanya.
- **Tidak ada pemeriksaan OCR.** Tidak ada bukti bahwa instruksi pada gambar terbaca benar oleh VLM.
- **Label dibuat oleh classifier berbasis aturan ditambah anotasi manusia pada kasus ambigu.** Repository tidak mencatat jumlah anotator maupun inter-annotator agreement. Karena itu, kualitas label tidak dapat diaudit dari artefak yang ada.
- **Cakupan JBB tidak merata.** Pada pasangan Qwen, JBB hanya punya hasil awal yang belum dianotasi penuh.
- **Seleksi sampel memakai keragaman leksikal**, bukan semantik.
- **Compliance bukan berarti konten berbahaya yang benar-benar berguna.** Label hanya menilai patuh atau menolak.
- Rate yang dihitung bukan bukti kausal bahwa modality saja yang menyebabkan perubahan perilaku.

### Saran lanjutan

1. Uji kemampuan baca gambar (OCR/echo test) pada LLaVA dan Qwen-VL sebelum menafsirkan hasil `vlm_image`.
2. Variasikan prompt pengantar visual dan ukuran font/render.
3. Tambahkan sampel dan hitung confidence interval.
4. Lengkapi dan konsistenkan anotasi JBB untuk pasangan Qwen, serta laporkan jumlah anotator dan agreement.
