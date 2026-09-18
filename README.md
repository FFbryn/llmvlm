# LLM vs LVLM

## README — Model Installation & Integration

**Project:** Pewarisan Kerentanan Jailbreak dari LLM ke Vision-Language Model
**Working Name:** LLM vs LVLM
**Research Area:** AI Security / AI Safety
**Current Phase:** Model Layer — LLM & VLM Integration

---

# 1. Konteks Project

Project ini merupakan penelitian akademik dengan tujuan mengetahui apakah perilaku/kerentanan jailbreak yang berhasil pada **Large Language Model (LLM)** berbasis teks tetap berhasil ketika perilaku yang sama direpresentasikan dalam bentuk visual dan dievaluasi menggunakan **Vision-Language Model (VLM/LVLM)**.

Pertanyaan penelitian utama:

1. Berapa banyak behavior yang berhasil pada LLM berbasis teks yang juga berhasil pada VLM pasangannya?
2. Apakah terdapat kasus yang gagal atau ditolak pada input teks tetapi berhasil ketika input yang sama direpresentasikan secara visual?
3. Apakah mekanisme guardrail dan refusal pada model berbasis teks tetap efektif ketika input diberikan melalui modality visual?

---

# 2. Prinsip Penting Penelitian

Project ini **bukan project untuk membuat jailbreak baru**.

Behavior berbahaya untuk eksperimen harus berasal dari benchmark/dataset yang telah ditentukan dalam project.

Jangan:

* membuat harmful behavior baru secara manual;
* membuat jailbreak prompt baru;
* mengoptimalkan prompt agar model melewati guardrail;
* mengubah benchmark secara langsung;
* menggunakan hasil smoke test sebagai hasil eksperimen penelitian.

Fokus penelitian adalah:

> **Transferability** — kemampuan suatu behavior yang berhasil pada modality teks untuk tetap menghasilkan outcome yang sama ketika direpresentasikan melalui modality visual.

---

# 3. Benchmark yang Digunakan

Benchmark yang telah ditentukan dalam project:

1. JBB-Behaviors
2. AdvBench
3. HarmBench
4. XSTest

Benchmark akan digunakan pada tahap eksperimen setelah model layer dan pipeline dasar selesai.

Pada tahap **model installation/integration**, gunakan prompt benign untuk smoke test.

---

# 4. Model yang Digunakan

## 4.1 LLM

### Qwen2.5-7B-Instruct

Model ID:

```text
Qwen/Qwen2.5-7B-Instruct
```

Implementasi:

```text
models/llm/qwen.py
```

Class:

```python
QwenLLM
```

---

### Vicuna-1.5-7B

Model ID:

```text
lmsys/vicuna-7b-v1.5
```

Implementasi:

```text
models/llm/vicuna.py
```

Class:

```python
VicunaLLM
```

---

# 5. VLM

## 5.1 Qwen2.5-VL-7B-Instruct

Model ID:

```text
Qwen/Qwen2.5-VL-7B-Instruct
```

Implementasi:

```text
models/vlm/qwen_v1.py
```

Class:

```python
QwenVLM
```

Komponen utama:

```python
AutoProcessor
Qwen2_5_VLForConditionalGeneration
```

---

## 5.2 LLaVA-1.5-7B

Checkpoint Hugging Face yang digunakan:

```text
llava-hf/llava-1.5-7b-hf
```

Implementasi:

```text
models/vlm/llava.py
```

Class:

```python
LLaVAVLM
```

Komponen utama:

```python
AutoProcessor
LlavaForConditionalGeneration
```

---

# 6. Arsitektur Model Layer

Struktur abstraction:

```text
                    BaseModel
                       |
             +---------+---------+
             |                   |
          BaseLLM              BaseVLM
             |                   |
       +-----+-----+       +-----+------+
       |           |       |            |
     Qwen       Vicuna   Qwen-VL      LLaVA
```

Tujuan abstraction:

Pipeline eksperimen tidak boleh bergantung langsung pada implementasi internal masing-masing model.

Pipeline cukup menggunakan interface umum:

```python
model.load()

response = model.generate(
    prompt=prompt,
    image_path=image_path,
    sample_id=sample_id,
)

model.unload()
```

---

# 7. Struktur Repository Saat Ini

Struktur model layer:

```text
models/
├── base.py
│
├── llm/
│   ├── __init__.py
│   ├── base.py
│   ├── qwen.py
│   └── vicuna.py
│
└── vlm/
    ├── __init__.py
    ├── base.py
    ├── qwen_v1.py
    └── llava.py
```

Test:

```text
tests/
├── fixtures/
│   ├── dummy_dataset.csv
│   └── vlm_test_image.png
│
├── test_llm_base.py
├── test_qwen.py
├── test_vicuna.py
├── test_vlm_base.py
├── test_qwen_vlm.py
└── test_llava.py
```

Manual smoke tests:

```text
tests/
├── manual_qwen_smoke.py
├── manual_vicuna_smoke.py
├── manual_qwen_vlm_smoke.py
└── manual_llava_smoke.py
```

---

# 8. `BaseModel`

File:

```text
models/base.py
```

Menyediakan abstraction dasar untuk semua model.

Komponen utama:

```python
ModelResponse
BaseModel
```

`ModelResponse` memiliki struktur:

```python
@dataclass
class ModelResponse:
    text: str
    model_name: str
    model_type: str
    sample_id: Optional[str] = None
    image_path: Optional[Path] = None
    metadata: Optional[dict] = None
```

Tujuan:

Response dari LLM dan VLM memiliki struktur yang konsisten.

Contoh:

```text
LLM response
    |
    +-- text
    +-- model_name
    +-- model_type = llm
    +-- sample_id
    +-- image_path = None
```

Sedangkan:

```text
VLM response
    |
    +-- text
    +-- model_name
    +-- model_type = vlm
    +-- sample_id
    +-- image_path
```

---

# 9. `BaseLLM`

File:

```text
models/llm/base.py
```

`BaseLLM` merupakan subclass dari:

```python
BaseModel
```

LLM hanya menerima input text.

Jika:

```python
image_path != None
```

maka `BaseLLM.generate()` harus menolak input tersebut dengan `ValueError`.

Interface:

```python
model.generate(
    prompt=prompt,
    sample_id=sample_id,
)
```

Subclass mengimplementasikan:

```python
_generate_text()
```

---

# 10. `BaseVLM`

File:

```text
models/vlm/base.py
```

`BaseVLM` merupakan subclass dari:

```python
BaseModel
```

VLM dapat menerima:

```text
text
```

atau:

```text
image + text
```

Interface:

```python
model.generate(
    prompt=prompt,
    image_path=image_path,
    sample_id=sample_id,
)
```

Subclass mengimplementasikan:

```python
_generate_multimodal()
```

Untuk eksperimen visual utama, VLM harus menerima image representation.

---

# 11. Qwen LLM

File:

```text
models/llm/qwen.py
```

Class:

```python
QwenLLM
```

Model:

```text
Qwen/Qwen2.5-7B-Instruct
```

Komponen:

```python
AutoTokenizer
AutoModelForCausalLM
```

Alur:

```text
prompt
   |
   v
AutoTokenizer
   |
   v
Qwen2.5-7B-Instruct
   |
   v
generated tokens
   |
   v
decoded text
   |
   v
ModelResponse
```

Generation default:

```python
max_new_tokens=256
temperature=0.0
do_sample=False
```

Untuk smoke test digunakan konfigurasi lebih kecil:

```python
max_new_tokens=32
do_sample=False
```

---

# 12. Vicuna LLM

File:

```text
models/llm/vicuna.py
```

Class:

```python
VicunaLLM
```

Model:

```text
lmsys/vicuna-7b-v1.5
```

Komponen:

```python
AutoTokenizer
AutoModelForCausalLM
```

Alur:

```text
prompt
   |
   v
Vicuna conversation format
   |
   v
Tokenizer
   |
   v
Vicuna
   |
   v
generated tokens
   |
   v
ModelResponse
```

Smoke test menggunakan prompt benign.

---

# 13. Qwen VLM

File:

```text
models/vlm/qwen_v1.py
```

Class:

```python
QwenVLM
```

Model:

```text
Qwen/Qwen2.5-VL-7B-Instruct
```

Komponen utama:

```python
AutoProcessor
Qwen2_5_VLForConditionalGeneration
```

Alur:

```text
             prompt
                |
                |
image ----------+
                |
                v
         AutoProcessor
                |
                v
       Qwen2.5-VL-7B
                |
                v
          text response
                |
                v
          ModelResponse
```

`image_path` harus tersedia pada multimodal generation.

Jika image tidak diberikan:

```python
ValueError
```

Jika image tidak ditemukan:

```python
FileNotFoundError
```

---

# 14. LLaVA VLM

File:

```text
models/vlm/llava.py
```

Class:

```python
LLaVAVLM
```

Model:

```text
llava-hf/llava-1.5-7b-hf
```

Komponen utama:

```python
AutoProcessor
LlavaForConditionalGeneration
```

Alur:

```text
prompt
   +
image
   |
   v
conversation
   |
   v
apply_chat_template()
   |
   v
AutoProcessor
   |
   v
LLaVA
   |
   v
generated tokens
   |
   v
ModelResponse
```

---

# 15. Lifecycle Model

Semua adapter model harus memiliki lifecycle:

```text
create
  |
  v
load()
  |
  v
generate()
  |
  v
unload()
```

Jangan mengasumsikan model sudah loaded setelah constructor.

Contoh:

```python
model = QwenLLM()

model.load()

response = model.generate(
    prompt="...",
    sample_id="..."
)

model.unload()
```

Jika `generate()` dipanggil sebelum `load()`, adapter harus memberikan `RuntimeError`.

---

# 16. Smoke Test

Smoke test adalah pengujian sederhana untuk memastikan model dan adapter dapat berjalan.

Smoke test BUKAN eksperimen jailbreak.

Gunakan prompt benign seperti:

```text
Explain what machine learning is in one short sentence.
```

atau untuk VLM:

```text
Describe this image in one short sentence.
```

---

# 17. Test Files

## Qwen LLM

```text
tests/test_qwen.py
```

Run:

```bash
pytest tests/test_qwen.py -v -m slow
```

Manual:

```bash
python tests/manual_qwen_smoke.py
```

---

## Vicuna

```text
tests/test_vicuna.py
```

Run:

```bash
pytest tests/test_vicuna.py -v -m slow
```

Manual:

```bash
python tests/manual_vicuna_smoke.py
```

---

## BaseVLM

```text
tests/test_vlm_base.py
```

Run:

```bash
pytest tests/test_vlm_base.py -v
```

---

## Qwen VLM

```text
tests/test_qwen_vlm.py
```

Run:

```bash
pytest tests/test_qwen_vlm.py -v -m slow
```

Manual:

```bash
python tests/manual_qwen_vlm_smoke.py
```

---

## LLaVA

```text
tests/test_llava.py
```

Run:

```bash
pytest tests/test_llava.py -v -m slow
```

Manual:

```bash
python tests/manual_llava_smoke.py
```

---

# 18. Regression Test

Untuk menjalankan seluruh test yang tidak membutuhkan model besar:

```bash
pytest -v -m "not slow"
```

Test yang membutuhkan loading model ditandai:

```python
@pytest.mark.slow
```

Sehingga test cepat dan test model dapat dipisahkan.

---

# 19. Fixture Image

Untuk smoke test VLM digunakan:

```text
tests/fixtures/vlm_test_image.png
```

Image ini hanya digunakan untuk testing adapter.

Image fixture:

* bukan benchmark;
* bukan harmful content;
* bukan data eksperimen;
* tidak digunakan untuk menghitung transferability.

Tujuannya hanya memastikan image pipeline berfungsi.

---

# 20. Status Penelitian

Pada awal conversation baru, jangan menganggap semua test berikut sudah benar-benar dijalankan.

Status awal berdasarkan implementasi yang telah dibuat:

```text
BaseModel              IMPLEMENTED
BaseLLM                IMPLEMENTED
QwenLLM                IMPLEMENTED
VicunaLLM              IMPLEMENTED
BaseVLM                IMPLEMENTED
QwenVLM                IMPLEMENTED
LLaVAVLM               IMPLEMENTED
```

Status eksekusi test harus dianggap:

```text
UNKNOWN
```

kecuali user memberikan output terminal aktual.

Jangan mengubah `UNKNOWN` menjadi `PASSED` tanpa bukti.

---

# 21. Hal yang Harus Diverifikasi Jika Ada Error

Jika model gagal dijalankan, periksa secara berurutan:

### 1. Python environment

```bash
python --version
```

### 2. PyTorch

```bash
python -c "import torch; print(torch.__version__)"
```

### 3. CUDA

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 4. Transformers

```bash
python -c "import transformers; print(transformers.__version__)"
```

### 5. Import model

Qwen:

```bash
python -c "from models.llm.qwen import QwenLLM; print('OK')"
```

Vicuna:

```bash
python -c "from models.llm.vicuna import VicunaLLM; print('OK')"
```

Qwen VLM:

```bash
python -c "from models.vlm.qwen_v1 import QwenVLM; print('OK')"
```

LLaVA:

```bash
python -c "from models.vlm.llava import LLaVAVLM; print('OK')"
```

---

# 22. Prinsip Debugging

Jika terjadi error, jangan langsung mengubah seluruh arsitektur.

Identifikasi terlebih dahulu kategori error:

```text
Import Error
     |
     +-- package
     +-- class
     +-- path

Model Loading Error
     |
     +-- model ID
     +-- network/cache
     +-- tokenizer/processor
     +-- memory

CUDA Error
     |
     +-- VRAM
     +-- CUDA
     +-- dtype
     +-- device

Generation Error
     |
     +-- input format
     +-- processor
     +-- generation parameters

Image Error
     |
     +-- path
     +-- file
     +-- format
     +-- PIL
```

Perbaiki masalah yang paling spesifik terlebih dahulu.

---

# 23. Hal yang Jangan Dilakukan

Jangan:

```text
❌ mengubah benchmark untuk memperbaiki hasil
❌ membuat jailbreak prompt baru
❌ memasukkan harmful prompt secara manual
❌ menganggap response tertentu sebagai jailbreak tanpa classifier
❌ menyimpulkan transferability dari smoke test
❌ membandingkan model berdasarkan satu prompt
❌ mengubah model architecture hanya karena output berbeda
```

Smoke test hanya menjawab:

> "Apakah adapter/model dapat berjalan?"

Smoke test tidak menjawab:

> "Apakah behavior berhasil ditransfer?"

---

# 24. Perbedaan Smoke Test dan Eksperimen

## Smoke Test

Tujuan:

```text
Apakah model dapat berjalan?
```

Input:

```text
Benign prompt
```

Output:

```text
ModelResponse
```

---

## Eksperimen

Tujuan:

```text
Apakah behavior dari benchmark
memiliki outcome yang sama pada
LLM dan VLM?
```

Input:

```text
Benchmark sample
```

Output:

```text
LLM outcome
VLM outcome
Paired comparison
```

Eksperimen baru dilakukan setelah pipeline dan model layer stabil.

---

# 25. Target Arsitektur Setelah Model Layer Selesai

```text
                         BENCHMARK
                             |
                             v
                    Dataset / Sample
                             |
                             v
                    Normalization Layer
                             |
               +-------------+-------------+
               |                           |
               v                           v
          Text Representation       Visual Representation
               |                           |
               v                           v
              LLM                         VLM
               |                           |
        +------+-------+             +-----+------+
        |              |             |            |
      Qwen           Vicuna       Qwen-VL       LLaVA
        |              |             |            |
        +------+-------+             +-----+------+
               |                           |
               v                           v
          LLM Response                VLM Response
               |                           |
               v                           v
        Response Classifier         Response Classifier
               |                           |
               +-------------+-------------+
                             |
                             v
                    Paired Comparison
                             |
                             v
                   Transferability Analysis
```

---

# 26. Langkah Berikutnya

Setelah semua adapter model selesai, **jangan langsung menjalankan benchmark**.

Langkah berikutnya:

```text
1. Model Registry / Factory
2. Unified Model Runner
3. Configuration management
4. Model execution result schema
5. Experiment runner
6. Dataset → Model pipeline
7. Response classification
8. Paired comparison
9. Transferability analysis
```

Urutan yang direkomendasikan:

```text
BaseModel
   ↓
BaseLLM / BaseVLM
   ↓
4 Model Adapter
   ↓
Model Factory / Registry
   ↓
Unified Runner
   ↓
Experiment Pipeline
```

---

# 27. Instruksi untuk Conversation Berikutnya

Jika README ini digunakan sebagai konteks pada conversation baru, assistant harus:

1. Membaca README terlebih dahulu.
2. Menganggap project berada pada tahap **Model Layer — LLM & VLM Integration**.
3. Tidak mengulang penjelasan dasar yang sudah ada kecuali diperlukan.
4. Tidak langsung menjalankan benchmark harmful.
5. Tidak membuat jailbreak baru.
6. Menggunakan benchmark yang telah ditentukan.
7. Memisahkan:

   * fakta,
   * hasil test aktual,
   * interpretasi,
   * hipotesis,
   * asumsi,
   * keterbatasan.
8. Jika membutuhkan informasi model/API yang dapat berubah, verifikasi sumber resmi terlebih dahulu.
9. Jika user memberikan error terminal, diagnosis error tersebut terlebih dahulu sebelum melanjutkan arsitektur.
10. Jika adapter model sudah stabil, lanjutkan ke **Model Factory / Registry**.

---

# 28. Expected Next Task

Task berikutnya:

## Implementasi Model Factory / Registry

Target:

```text
models/
├── base.py
├── factory.py          ← NEXT
│
├── llm/
│   ├── base.py
│   ├── qwen.py
│   └── vicuna.py
│
└── vlm/
    ├── base.py
    ├── qwen_v1.py
    └── llava.py
```

Tujuan:

Pipeline dapat melakukan:

```python
model = create_model("qwen_llm")
```

atau:

```python
model = create_model("vicuna_llm")
```

atau:

```python
model = create_model("qwen_vlm")
```

atau:

```python
model = create_model("llava_vlm")
```

tanpa perlu meng-import setiap model secara manual pada experiment runner.

Setelah Factory selesai, lanjutkan ke **Unified Model Runner**.

---

# 29. Catatan Reproducibility

Semua eksperimen nantinya harus mencatat setidaknya:

```text
sample_id
benchmark
behavior/category
model_name
model_type
input modality
prompt/input representation
image_path jika ada
generation configuration
response
classification
timestamp
```

Jika parameter generation digunakan:

```text
max_new_tokens
temperature
do_sample
dtype
device
```

juga perlu dicatat.

Tujuannya agar hasil eksperimen dapat direproduksi dan dibandingkan secara konsisten.

---

# 30. Prinsip Utama Project

Jangan mengubah pertanyaan:

> "Apakah behavior yang sudah tersedia pada benchmark dapat berpindah dari text modality ke visual modality?"

menjadi:

> "Bagaimana cara membuat jailbreak berhasil?"

Project ini adalah **evaluasi transferability**, bukan pengembangan jailbreak.

Fokus utama:

```text
Benchmark
   ↓
Controlled Representation
   ↓
LLM vs VLM
   ↓
Response Classification
   ↓
Paired Comparison
   ↓
Transferability
```

**End of README**
