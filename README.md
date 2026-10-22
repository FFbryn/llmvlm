# LLM vs LVLM

**Pewarisan Kerentanan Jailbreak dari LLM ke Vision-Language Model**

Project penelitian AI Security dan AI Safety untuk mengevaluasi apakah
outcome perilaku yang diamati pada benchmark teks tetap muncul ketika
instruksi benchmark yang sama disajikan melalui modality visual kepada
Vision-Language Model (VLM).

> **Status project:** pipeline dasar sedang dikembangkan dan diuji.
> Status eksperimen penuh, cakupan seluruh pasangan model, integrasi
> anotasi manusia, dan hasil transferability harus dikonfirmasi dari
> artefak terbaru. Jangan menafsirkan jumlah sampel yang berhasil dimuat
> sebagai jumlah sampel yang sudah selesai dievaluasi.

## Daftar isi

-   [Tujuan penelitian](#tujuan-penelitian)
-   [Pertanyaan penelitian](#pertanyaan-penelitian)
-   [Prinsip metodologis](#prinsip-metodologis)
-   [Struktur project](#struktur-project)
-   [Pipeline eksperimen](#pipeline-eksperimen)
-   [Menyiapkan lingkungan](#menyiapkan-lingkungan)
-   [Menjalankan pengujian](#menjalankan-pengujian)
-   [Menjalankan eksperimen](#menjalankan-eksperimen)
-   [Format hasil](#format-hasil)
-   [Analisis](#analisis)
-   [Reproducibility dan audit](#reproducibility-dan-audit)
-   [Status dan pekerjaan berikutnya](#status-dan-pekerjaan-berikutnya)

## Tujuan penelitian

Project ini mengevaluasi **transferability**, yaitu sejauh mana outcome
dari suatu instruksi benchmark pada kondisi teks tetap muncul ketika
instruksi yang sama direpresentasikan dalam bentuk gambar dan dievaluasi
oleh VLM.

Project ini bukan untuk mengembangkan jailbreak baru. Input perilaku
yang diuji harus berasal dari benchmark yang telah ditetapkan dalam
protokol penelitian.

## Pertanyaan penelitian

1.  Berapa banyak perilaku yang berhasil pada LLM berbasis teks yang
    juga berhasil pada VLM pasangannya?
2.  Apakah terdapat kasus yang gagal atau ditolak pada kondisi teks
    tetapi berhasil pada kondisi visual?
3.  Apakah guardrail dan mekanisme refusal tetap efektif ketika input
    diberikan melalui modality visual?

## Prinsip metodologis

-   **Kesetaraan input:** kondisi teks dan visual harus
    merepresentasikan instruksi benchmark yang sama.
-   **Pairing per sampel:** perbandingan dilakukan berdasarkan ID sampel
    dan pasangan model yang valid.
-   **Pemisahan outcome:** refusal, compliance/keberhasilan, respons
    ambigu, dan execution error tidak boleh dicampur tanpa aturan
    eksplisit.
-   **Tidak mengarang hasil:** angka hasil hanya boleh dilaporkan dari
    artefak eksperimen yang benar-benar dihasilkan.
-   **Human annotation:** jika digunakan, simpan label, pedoman anotasi,
    identitas anotator dalam bentuk yang sesuai kebijakan penelitian,
    serta provenance label.
-   **Reproducibility:** simpan versi kode, konfigurasi, model, prompt
    template, versi dataset, dan log.
-   **Responsible research:** gunakan benchmark yang disetujui dalam
    protokol; dokumentasikan temuan sebagai evaluasi keamanan, bukan
    sebagai panduan untuk membuat serangan baru.

## Struktur project

Struktur berikut merupakan gambaran komponen yang dibahas selama
pengembangan. Sesuaikan dengan isi repository aktual; daftar ini bukan
jaminan bahwa setiap file masih ada atau sudah selesai.

``` text
.
├── benchmark_data/
│   ├── loader.py
│   ├── schema.py
│   ├── adapters/
│   │   ├── advbench.py
│   │   └── harmbench.py
│   └── results/
│       └── jbb_100/
├── evaluation/
│   ├── input_builder.py
│   ├── experiment_executor.py
│   ├── raw_result_writer.py
│   ├── batch_experiment_runner.py
│   ├── outcome.py
│   ├── classification/
│   │   └── labels.py
│   └── analysis/
│       └── analyze_outcomes.py
└── tests/
    ├── test_experiment_executor.py
    └── test_three_condition_integration.py
```

### Tanggung jawab komponen

  -------------------------------------------------------------------------------
  Komponen                                    Tanggung jawab
  ------------------------------------------- -----------------------------------
  `benchmark_data/loader.py`                  Memuat data benchmark

  `benchmark_data/schema.py`                  Mendefinisikan bentuk data sampel

  `benchmark_data/adapters/`                  Mengadaptasi format dataset sumber
                                              ke schema internal

  `evaluation/input_builder.py`               Menyusun input sesuai kondisi
                                              eksperimen

  `evaluation/experiment_executor.py`         Menjalankan satu percobaan melalui
                                              konfigurasi/model yang dipilih

  `evaluation/raw_result_writer.py`           Menulis hasil mentah ke penyimpanan

  `evaluation/batch_experiment_runner.py`     Mengatur eksekusi untuk banyak
                                              sampel

  `evaluation/classification/labels.py`       Mendefinisikan label klasifikasi

  `evaluation/outcome.py`                     Merepresentasikan outcome
                                              eksperimen

  `evaluation/analysis/analyze_outcomes.py`   Mengagregasi dan menganalisis hasil
                                              yang tersimpan

  `tests/`                                    Menguji perilaku komponen dan
                                              integrasi
  -------------------------------------------------------------------------------

## Pipeline eksperimen

Alur konseptual yang diharapkan:

1.  **Load benchmark:** baca sampel dan validasi schema.
2.  **Load visual manifest:** hubungkan ID benchmark ke artefak gambar
    yang tepat.
3.  **Build conditions:** buat input untuk kondisi eksperimen sesuai
    protokol yang ditetapkan.
4.  **Execute:** jalankan setiap sampel pada model dan kondisi yang
    direncanakan.
5.  **Write raw results:** simpan respons, metadata, status, dan error
    tanpa kehilangan jejak asalnya.
6.  **Classify:** beri label outcome dengan aturan yang terdokumentasi.
7.  **Human review:** bila termasuk protokol, anotasi atau audit sampel
    yang relevan.
8.  **Analyze:** hitung metrik per pasangan sampel/model dan hasil
    agregat.
9.  **Report:** buat tabel, grafik, interpretasi, serta keterbatasan.

### Catatan penting tentang kondisi visual

Prompt pengantar yang hanya meminta model mendeskripsikan gambar
berisiko mengukur kemampuan image captioning, bukan respons terhadap
instruksi benchmark di dalam gambar. Pastikan protokol kondisi visual
dirancang untuk mengevaluasi instruksi target dan disejajarkan dengan
kondisi pembanding. Jangan mengubah input builder sekadar untuk membuat
smoke test tampak berhasil.

## Menyiapkan lingkungan

Contoh perintah berikut ditujukan untuk Windows PowerShell dengan
lingkungan Conda bernama `llmvlm`, sebagaimana digunakan dalam sesi
pengembangan. Sesuaikan nama lingkungan jika berbeda.

``` powershell
conda activate llmvlm
python --version
python -m pytest --version
```

Pastikan dependency project telah dipasang sesuai berkas dependency yang
benar-benar ada di repository, misalnya `requirements.txt` atau
`pyproject.toml`. Jangan membuat berkas dependency baru berdasarkan
tebakan; periksa konfigurasi project yang tersedia.

## Menjalankan pengujian

Mulai dari tes komponen inti:

``` powershell
python -m pytest tests/test_experiment_executor.py -v
```

Kemudian tes integrasi kondisi eksperimen:

``` powershell
python -m pytest tests/test_three_condition_integration.py -v
```

Jika kedua file tersebut ada, tes dapat dijalankan bersama:

``` powershell
python -m pytest tests/test_experiment_executor.py tests/test_three_condition_integration.py -v
```

Untuk menjalankan semua tes yang terdeteksi:

``` powershell
python -m pytest -v
```

Simpan output tes lengkap sebagai bukti audit. Status sukses satu tes
tidak membuktikan bahwa seluruh pipeline atau eksperimen penuh berhasil.

## Menjalankan eksperimen

Nama skrip dan argumen eksekusi harus diverifikasi terhadap repository
terbaru. Riwayat pengembangan membahas runner batch serta skrip manual
seperti `manual_full_batch.py` dan smoke test batch. Sebelum menjalankan
eksperimen, periksa:

1.  Apakah konfigurasi model mengarah ke model yang dimaksud?
2.  Apakah kredensial dan akses model tersedia melalui mekanisme yang
    aman?
3.  Apakah manifest gambar cocok dengan ID benchmark?
4.  Apakah kondisi eksperimen telah ditetapkan dan didokumentasikan?
5.  Apakah output diarahkan ke direktori hasil yang benar?
6.  Apakah skrip dapat dilanjutkan atau dijalankan ulang tanpa menimpa
    hasil yang sudah ada?

Jalankan smoke test dengan subset kecil terlebih dahulu. Setelah hasil
mentah dan pemetaan input lolos audit, jalankan batch penuh dengan
konfigurasi yang dibekukan. Gunakan perintah aktual yang tercantum dalam
skrip/repository; README ini sengaja tidak mengarang argumen CLI yang
belum terverifikasi.

## Format hasil

Runner membahas penggunaan **JSONL (JSON Lines)**, yaitu satu objek JSON
per baris. Bentuk field yang persis harus mengikuti implementasi
`raw_result_writer.py` dan schema aktual. Secara konseptual, satu record
hasil sebaiknya dapat ditelusuri ke:

-   ID sampel benchmark;
-   nama/ID dataset dan versinya;
-   model serta kondisi eksperimen;
-   referensi visual bila berlaku;
-   respons mentah model;
-   status eksekusi dan metadata error;
-   waktu atau metadata eksekusi yang relevan;
-   label otomatis dan/atau label manusia, bila tersedia, dengan sumber
    label yang jelas.

Jangan menganggap daftar di atas sebagai nama field JSONL yang sudah
pasti. Periksa writer dan satu contoh output aktual untuk mengetahui
nama, tipe, serta field wajib yang benar.

### Validasi jumlah hasil

Bandingkan jumlah percobaan yang direncanakan dengan jumlah record yang
berhasil ditulis, lalu hitung terpisah:

-   percobaan selesai dengan respons valid;
-   refusal;
-   respons ambigu atau tidak dapat diklasifikasikan;
-   execution error/timeout;
-   percobaan yang belum dijalankan.

Satu sampel dapat menghasilkan beberapa record jika diuji pada beberapa
model atau kondisi. Karena itu, jumlah baris tidak selalu sama dengan
jumlah sampel unik.

## Analisis

`evaluation/analysis/analyze_outcomes.py` merupakan komponen analisis
yang dibahas dalam pengembangan. Sebelum menggunakan hasilnya dalam
laporan, verifikasi sumber input, aturan klasifikasi, denominator,
penanganan data hilang, dan apakah anotasi manusia dibaca oleh program.

Metrik yang direncanakan:

-   **Transfer rate bersyarat:** proporsi sampel yang berhasil pada
    visual di antara sampel yang berhasil pada teks dan mempunyai
    pasangan visual valid.
-   **Visual-only success:** kasus ketika kondisi visual berhasil,
    tetapi kondisi teks pasangannya tidak berhasil.
-   **Refusal rate:** proporsi respons valid yang diberi label refusal,
    dengan perlakuan terhadap error dan label ambigu yang dinyatakan
    jelas.
-   **Over-refusal rate:** proporsi permintaan aman yang keliru ditolak;
    memerlukan data aman berlabel dan tidak bisa diukur hanya dari
    benchmark berbahaya.

Finalisasi definisi operasional sebelum menghitung angka. Sertakan
jumlah absolut, persentase, denominator, aturan eksklusi, dan jumlah
kasus yang tidak dapat diklasifikasikan.

## Reproducibility dan audit

Untuk setiap run, dokumentasikan:

-   commit atau versi kode;
-   versi Python dan dependency;
-   dataset dan versinya;
-   daftar model dan konfigurasi yang digunakan;
-   prompt template serta kondisi eksperimen;
-   jumlah sampel yang direncanakan dan dijalankan;
-   lokasi hasil mentah dan log;
-   aturan retry serta penanganan error;
-   versi classifier dan definisi label;
-   status human annotation, jika digunakan.

Jangan menyimpan API key atau rahasia lain ke repository maupun file
hasil.

## Status dan pekerjaan berikutnya

Berdasarkan catatan pengembangan yang tersedia:

-   Loader JBB pernah melaporkan 100 sampel tersedia pada smoke test.
-   Visual manifest pernah melaporkan 100 sampel tersedia pada smoke
    test.
-   Pengguna melaporkan pengujian executor telah passed.
-   Batch runner, penulisan JSONL, klasifikasi, dan analisis outcome
    telah menjadi bagian dari implementasi yang dibahas.

Hal-hal berikut **belum dapat dianggap selesai tanpa bukti tambahan**:

-   seluruh tes repository lulus pada versi terkini;
-   eksperimen end-to-end berhasil;
-   semua 100 sampel telah dievaluasi pada semua pasangan model dan
    kondisi;
-   seluruh pasangan LLM--VLM telah digunakan;
-   human annotation telah diintegrasikan ke analisis;
-   metrik transferability, visual-only, refusal, dan over-refusal telah
    dihitung dan divalidasi.

### Urutan kerja yang disarankan

1.  Jalankan seluruh tes dan simpan output.
2.  Audit loader, schema, dan pemetaan visual manifest.
3.  Validasi kesetaraan kondisi teks dan visual.
4.  Jalankan smoke test end-to-end dan audit JSONL.
5.  Bekukan matriks model/kondisi/sampel lalu jalankan batch penuh.
6.  Validasi classifier dan integrasi anotasi manusia.
7.  Jalankan analisis dengan denominator yang jelas.
8.  Buat grafik, laporan, dan presentasi berdasarkan hasil yang telah
    diaudit.

## Kontribusi dan status hasil

Dokumen ini mendeskripsikan tujuan serta struktur yang diketahui dari
catatan pengembangan. Perbarui bagian status berdasarkan bukti yang
dapat direproduksi; jangan menambahkan angka eksperimen atau klaim
keberhasilan tanpa hasil aktual.
