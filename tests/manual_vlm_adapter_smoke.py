from pathlib import Path

from models.vlm.qwen_v1 import QwenVLM
from models.vlm.llava import LLaVAVLM


TEST_PROMPT = (
    "Please respond briefly with the word "
    "'READY' if you can process this text."
)

TEST_IMAGE = Path(
    "data/visual_dataset/sample.png"
)


def test_qwen_vlm_text():
    print("=" * 60)
    print("QWEN VLM - TEXT ONLY")
    print("=" * 60)

    model = QwenVLM(
        max_new_tokens=32,
        temperature=0.0,
        do_sample=False,
    )

    try:
        print("Loading model...")
        model.load()

        print("Generating...")
        response = model.generate(
            prompt=TEST_PROMPT,
            image_path=None,
            sample_id="smoke_qwen_text",
        )

        print()
        print("Model name :", response.model_name)
        print("Model type :", response.model_type)
        print("Sample ID  :", response.sample_id)
        print("Image path :", response.image_path)
        print("Response   :", response.text)

        assert response.model_type == "vlm"
        assert response.sample_id == "smoke_qwen_text"
        assert response.image_path is None
        assert response.text.strip()

        print()
        print("Qwen VLM text-only: PASS")

    finally:
        print("Unloading model...")
        model.unload()


def test_qwen_vlm_image():
    print("=" * 60)
    print("QWEN VLM - IMAGE")
    print("=" * 60)

    if not TEST_IMAGE.exists():
        raise FileNotFoundError(
            f"Test image tidak ditemukan: {TEST_IMAGE}"
        )

    model = QwenVLM(
        max_new_tokens=32,
        temperature=0.0,
        do_sample=False,
    )

    try:
        print("Loading model...")
        model.load()

        print("Generating...")
        response = model.generate(
            prompt="Please describe the image briefly.",
            image_path=TEST_IMAGE,
            sample_id="smoke_qwen_image",
        )

        print()
        print("Model name :", response.model_name)
        print("Model type :", response.model_type)
        print("Sample ID  :", response.sample_id)
        print("Image path :", response.image_path)
        print("Response   :", response.text)

        assert response.model_type == "vlm"
        assert response.sample_id == "smoke_qwen_image"
        assert response.image_path == TEST_IMAGE
        assert response.text.strip()

        print()
        print("Qwen VLM image: PASS")

    finally:
        print("Unloading model...")
        model.unload()


def test_llava_vlm_text():
    print("=" * 60)
    print("LLAVA VLM - TEXT ONLY")
    print("=" * 60)

    model = LLaVAVLM(
        max_new_tokens=32,
        temperature=0.0,
        do_sample=False,
    )

    try:
        print("Loading model...")
        model.load()

        print("Generating...")
        response = model.generate(
            prompt=TEST_PROMPT,
            image_path=None,
            sample_id="smoke_llava_text",
        )

        print()
        print("Model name :", response.model_name)
        print("Model type :", response.model_type)
        print("Sample ID  :", response.sample_id)
        print("Image path :", response.image_path)
        print("Response   :", response.text)

        assert response.model_type == "vlm"
        assert response.sample_id == "smoke_llava_text"
        assert response.image_path is None
        assert response.text.strip()

        print()
        print("LLaVA VLM text-only: PASS")

    finally:
        print("Unloading model...")
        model.unload()


def test_llava_vlm_image():
    print("=" * 60)
    print("LLAVA VLM - IMAGE")
    print("=" * 60)

    if not TEST_IMAGE.exists():
        raise FileNotFoundError(
            f"Test image tidak ditemukan: {TEST_IMAGE}"
        )

    model = LLaVAVLM(
        max_new_tokens=32,
        temperature=0.0,
        do_sample=False,
    )

    try:
        print("Loading model...")
        model.load()

        print("Generating...")
        response = model.generate(
            prompt="Please describe the image briefly.",
            image_path=TEST_IMAGE,
            sample_id="smoke_llava_image",
        )

        print()
        print("Model name :", response.model_name)
        print("Model type :", response.model_type)
        print("Sample ID  :", response.sample_id)
        print("Image path :", response.image_path)
        print("Response   :", response.text)

        assert response.model_type == "vlm"
        assert response.sample_id == "smoke_llava_image"
        assert response.image_path == TEST_IMAGE
        assert response.text.strip()

        print()
        print("LLaVA VLM image: PASS")

    finally:
        print("Unloading model...")
        model.unload()


if __name__ == "__main__":
    test_qwen_vlm_text()
    test_qwen_vlm_image()

    test_llava_vlm_text()
    test_llava_vlm_image()

    print()
    print("=" * 60)
    print("ALL VLM ADAPTER SMOKE TESTS COMPLETED")
    print("=" * 60)