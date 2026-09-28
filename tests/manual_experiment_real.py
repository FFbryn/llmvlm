from pathlib import Path

from benchmark_data.loader import DatasetLoader
from benchmark_data.schema import BenchmarkSample

from config.experiment_config import ExperimentConfig
from config.model_config import GenerationConfig
from config.model_pair import ModelPairConfig

from evaluation.experiment_executor import ExperimentExecutor
from evaluation.model_runner_builder import ModelRunnerBuilder

from preprocessing.visual_manifest_loader import VisualManifestLoader


# ============================================================
# CONFIGURATION
# ============================================================

VISUAL_MANIFEST_PATH = Path(
    "benchmark_data/generated/visual_manifest_jbb.jsonl"
)

LLM_MODEL_KEY = "qwen_llm"
VLM_MODEL_KEY = "qwen_vlm"

MAX_NEW_TOKENS = 64

NEUTRAL_INTRO_PROMPT = (
    "Please describe the content of the image."
)


# ============================================================
# LOAD REAL JBB DATASET
# ============================================================

def load_first_jbb_sample() -> BenchmarkSample:
    """
    Memuat dataset JBB menggunakan DatasetLoader yang
    sudah menjadi entry point resmi project.

    Hanya satu sample yang digunakan untuk smoke test.
    """

    print("[DATASET] Loading JBB...")

    loader = DatasetLoader(
        benchmark="jbb"
    )

    samples = loader.load()

    if not samples:
        raise RuntimeError(
            "Dataset JBB berhasil dimuat tetapi "
            "tidak memiliki sample."
        )

    sample = samples[0]

    print(
        f"[DATASET] Loaded {len(samples)} JBB samples."
    )

    print(
        f"[DATASET] Selected sample: "
        f"{sample.sample_id}"
    )

    return sample


# ============================================================
# BUILD EXPERIMENT EXECUTOR
# ============================================================

def build_executor(
    dataset_samples: list[BenchmarkSample],
) -> ExperimentExecutor:

    if not VISUAL_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "Visual manifest tidak ditemukan:\n"
            f"{VISUAL_MANIFEST_PATH.resolve()}"
        )

    print(
        "[MANIFEST] Using:"
        f" {VISUAL_MANIFEST_PATH}"
    )

    config = ExperimentConfig(
        model_pair=ModelPairConfig(
            pair_id="qwen-qwen",
            llm_model_key=LLM_MODEL_KEY,
            vlm_model_key=VLM_MODEL_KEY,
        ),
        benchmark="JBB-Behaviors",
        generation=GenerationConfig(
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
            do_sample=False,
        ),
        visual_manifest_path=VISUAL_MANIFEST_PATH,
        neutral_intro_prompt=NEUTRAL_INTRO_PROMPT,
    )

    visual_manifest_loader = VisualManifestLoader(
        VISUAL_MANIFEST_PATH
    )

    executor = ExperimentExecutor(
        config=config,
        dataset_samples=dataset_samples,
        model_runner_builder=ModelRunnerBuilder(),
        visual_manifest_loader=visual_manifest_loader,
    )

    return executor


# ============================================================
# DISPLAY RESULT
# ============================================================

def display_result(result) -> None:
    """
    Menampilkan hasil dari tiga condition.

    Tahap ini hanya menampilkan raw response.
    Belum melakukan classification atau analysis.
    """

    print()
    print("=" * 70)
    print("REAL EXPERIMENT RESULT")
    print("=" * 70)

    # --------------------------------------------------------
    # LLM TEXT
    # --------------------------------------------------------

    print("\n[LLM_TEXT]")

    print(
        f"model      : "
        f"{result.llm_text.model_name}"
    )

    print(
        f"model_type : "
        f"{result.llm_text.model_type}"
    )

    print(
        f"modality   : "
        f"{result.llm_text.input_modality}"
    )

    print("response:")

    print(result.llm_text.response)

    # --------------------------------------------------------
    # VLM TEXT
    # --------------------------------------------------------

    print("\n[VLM_TEXT]")

    print(
        f"model      : "
        f"{result.vlm_text.model_name}"
    )

    print(
        f"model_type : "
        f"{result.vlm_text.model_type}"
    )

    print(
        f"modality   : "
        f"{result.vlm_text.input_modality}"
    )

    print("response:")

    print(result.vlm_text.response)

    # --------------------------------------------------------
    # VLM IMAGE
    # --------------------------------------------------------

    print("\n[VLM_IMAGE]")

    print(
        f"model      : "
        f"{result.vlm_image.model_name}"
    )

    print(
        f"model_type : "
        f"{result.vlm_image.model_type}"
    )

    print(
        f"modality   : "
        f"{result.vlm_image.input_modality}"
    )

    print(
        f"image_path : "
        f"{result.vlm_image.image_path}"
    )

    print("response:")

    print(result.vlm_image.response)

    print()
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("REAL THREE-CONDITION EXPERIMENT SMOKE TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    print("\n[1/4] Loading real JBB dataset...")

    sample = load_first_jbb_sample()

    print(
        f"sample_id : {sample.sample_id}"
    )

    print(
        f"benchmark : {sample.benchmark}"
    )

    print(
        f"category  : {sample.category}"
    )

    print(
        "prompt    :"
    )

    print(sample.prompt)

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    print(
        "\n[2/4] Building real ExperimentExecutor..."
    )

    executor = build_executor(
        dataset_samples=[sample]
    )

    print(
        "[EXECUTOR] Successfully created."
    )

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    print(
        "\n[3/4] Running three conditions..."
    )

    print(
        "  - LLM_TEXT"
    )

    print(
        "  - VLM_TEXT"
    )

    print(
        "  - VLM_IMAGE"
    )

    print()

    result = executor.run_sample(
        sample.sample_id
    )

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    print(
        "\n[4/4] Validating result..."
    )

    if result.sample_id != sample.sample_id:
        raise AssertionError(
            "Result sample_id tidak sesuai."
        )

    if not result.complete:
        raise AssertionError(
            "ThreeConditionResult tidak lengkap."
        )

    if result.llm_text is None:
        raise AssertionError(
            "LLM_TEXT result tidak tersedia."
        )

    if result.vlm_text is None:
        raise AssertionError(
            "VLM_TEXT result tidak tersedia."
        )

    if result.vlm_image is None:
        raise AssertionError(
            "VLM_IMAGE result tidak tersedia."
        )

    if not result.llm_text.response.strip():
        raise AssertionError(
            "LLM_TEXT menghasilkan response kosong."
        )

    if not result.vlm_text.response.strip():
        raise AssertionError(
            "VLM_TEXT menghasilkan response kosong."
        )

    if not result.vlm_image.response.strip():
        raise AssertionError(
            "VLM_IMAGE menghasilkan response kosong."
        )

    if result.vlm_image.image_path is None:
        raise AssertionError(
            "VLM_IMAGE tidak memiliki image_path."
        )

    if not Path(
        result.vlm_image.image_path
    ).exists():
        raise AssertionError(
            "Image yang digunakan VLM_IMAGE "
            "tidak ditemukan:\n"
            f"{result.vlm_image.image_path}"
        )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    display_result(result)

    print()
    print("=" * 70)
    print(
        "REAL THREE-CONDITION EXPERIMENT "
        "SMOKE TEST PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()