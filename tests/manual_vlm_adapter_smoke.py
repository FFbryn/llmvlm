from pathlib import Path

from models.vlm.qwen_v1 import QwenVLM
from preprocessing.visual_manifest_loader import VisualManifestLoader


MANIFEST_PATH = Path(
    "benchmark_data/generated/visual_manifest_jbb.jsonl"
)


def get_first_visual_sample():
    """
    Mengambil satu sample visual dari manifest JBB.
    """

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest tidak ditemukan: {MANIFEST_PATH}"
        )

    loader = VisualManifestLoader(
        MANIFEST_PATH
    )

    samples = loader.load()

    if not samples:
        raise RuntimeError(
            "Visual manifest tidak memiliki sample."
        )

    return samples[0]


def test_qwen_vlm_image():
    print("=" * 60)
    print("QWEN VLM - IMAGE SMOKE TEST")
    print("=" * 60)

    visual_sample = get_first_visual_sample()

    print("Sample ID :", visual_sample.sample_id)
    print("Benchmark :", visual_sample.benchmark)
    print("Image     :", visual_sample.image_path)

    if not visual_sample.image_path.exists():
        raise FileNotFoundError(
            "Image dari manifest tidak ditemukan: "
            f"{visual_sample.image_path}"
        )

    model = QwenVLM(
        max_new_tokens=32,
        temperature=0.0,
        do_sample=False,
    )

    try:
        print()
        print("Loading model...")

        model.load()

        print("Generating...")

        response = model.generate(
            prompt="Please describe the image briefly.",
            image_path=visual_sample.image_path,
            sample_id=visual_sample.sample_id,
        )

        print()
        print("Model name :", response.model_name)
        print("Model type :", response.model_type)
        print("Sample ID  :", response.sample_id)
        print("Image path :", response.image_path)
        print("Response   :", response.text)

        assert response.model_type == "vlm"

        assert (
            response.sample_id
            == visual_sample.sample_id
        )

        assert (
            response.image_path
            == visual_sample.image_path
        )

        assert response.text.strip()

        print()
        print("Qwen VLM image: PASS")

    finally:
        print()
        print("Unloading model...")
        model.unload()


if __name__ == "__main__":
    test_qwen_vlm_image()